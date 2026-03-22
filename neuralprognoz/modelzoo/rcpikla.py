from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

try:
    from kan import KAN
except ImportError:
    raise ImportError("Для использования RCPIKLA нужно установить pykan: pip install pykan")

from neuralprognoz.modelzoo.basemodel import BaseModel
from neuralprognoz.modelzoo.inputlayer import InputLayer
from neuralprognoz.modelzoo.head import get_head
from neuralprognoz.utils.config import Config


class SnowpackLayer(nn.Module):
    """Физическая модель снежного покрова (Zhou & Liu 2025, Eq. 14-16).

    Отслеживает накопление снега и таяние по временному ряду.
    fm = 0.5 мм/°C/день × 30 дней/мес = 15 мм/°C/мес (месячный шаг).
    """

    def __init__(self, fm: float = 15.0):
        super().__init__()
        self.fm = fm

    def forward(self, precip: torch.Tensor, temp: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        precip : [batch, seq_len]
        temp   : [batch, seq_len]

        Returns
        -------
        melt_seq : [batch, seq_len]  — фактическое таяние на каждом шаге
        """
        B, T = precip.shape
        swe = torch.zeros(B, device=precip.device)
        melt_list = []
        for t in range(T):
            snow_in = precip[:, t] * (temp[:, t] < 0).float()
            melt_rate = self.fm * F.relu(temp[:, t])
            melt_act = torch.minimum(swe, melt_rate)
            swe = F.relu(swe + snow_in - melt_act)
            melt_list.append(melt_act)
        return torch.stack(melt_list, dim=1)  # [B, T]


class GlobalAttention(nn.Module):
    """Глобальный механизм внимания по всей последовательности (Eq. 10-13).

    e_t = v^T tanh(W_a h_t)
    α_t = softmax(e_t)
    C   = Σ α_t h_t
    Q̂  = W_c C
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.W_a = nn.Linear(hidden_size, hidden_size)
        self.v   = nn.Linear(hidden_size, 1, bias=False)
        self.W_c = nn.Linear(hidden_size, hidden_size)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        lstm_out : [batch, seq_len, hidden]

        Returns
        -------
        context : [batch, hidden]
        """
        # e_t = v^T tanh(W_a h_t)  →  [batch, seq_len, 1]
        e = self.v(torch.tanh(self.W_a(lstm_out)))
        # α_t = softmax по seq_len
        alpha = torch.softmax(e, dim=1)          # [batch, seq_len, 1]
        # C = Σ α_t h_t
        context = (alpha * lstm_out).sum(dim=1)   # [batch, hidden]
        return self.W_c(context)


class TemporalSelfAttention(nn.Module):
    """Вариант C: временна́я self-attention с покадровыми предсказаниями (Вариант C).

    В отличие от GlobalAttention, который сворачивает всю последовательность
    в один контекстный вектор (и затем expand обратно), TemporalSelfAttention
    сохраняет per-timestep представление через nn.MultiheadAttention.

    Это позволяет модели учить разные веса внимания для разных выходных шагов,
    что критично для predict_last_n=3 (июль/август/сентябрь имеют разные паттерны).

    Архитектура:
        MHA(Q=h_t, K=h_1..T, V=h_1..T)  →  [B, T, H]  — каждый шаг видит всю историю
        + residual connection + LayerNorm
    """

    def __init__(self, hidden_size: int, num_heads: int = 4):
        super().__init__()
        # num_heads должен делить hidden_size нацело
        while hidden_size % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        self.mha  = nn.MultiheadAttention(hidden_size, num_heads=num_heads,
                                           batch_first=True, dropout=0.0)
        self.norm = nn.LayerNorm(hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        lstm_out : [batch, seq_len, hidden]

        Returns
        -------
        out : [batch, seq_len, hidden]  — временна́я структура сохранена
        """
        attn_out, _ = self.mha(lstm_out, lstm_out, lstm_out)
        out = self.norm(lstm_out + attn_out)
        return self.proj(out)


class KANBlock(nn.Module):
    """KAN-аппроксимация через FC+GELU+FC+LayerNorm (Eq. 1-3).

    Заменяет полноценный pyKAN при флаге use_linear_kan=True,
    что ускоряет обучение при больших hidden_size.
    """

    def __init__(self, in_features: int, hidden: int, out_features: int):
        super().__init__()
        self.expand   = nn.Linear(in_features, hidden)
        self.compress = nn.Linear(hidden, out_features)
        self.norm     = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.compress(F.gelu(self.expand(x))))


class PIKLAStage(nn.Module):
    """Один каскад PIKLA: KAN → LSTM → Attention → head.

    Используется дважды в RCPIKLA (Stage1=физика, Stage2=residual).

    use_temporal_attn=True (Вариант C): TemporalSelfAttention — сохраняет
    per-timestep представление, каждый выходной шаг видит всю историю.
    use_temporal_attn=False (исходный): GlobalAttention — один контекстный
    вектор для всей последовательности.
    """

    def __init__(self, cfg: Config, input_size: int, hidden_size: int,
                 kan_hidden: int, with_physics: bool,
                 precip_idx: int = 0, temp_idx: int = 1,
                 use_temporal_attn: bool = True):
        super().__init__()
        self.with_physics = with_physics
        self.precip_idx   = precip_idx
        self.temp_idx     = temp_idx
        self._use_temporal_attn = use_temporal_attn

        # KAN через pyKAN (как в PIKLA) или линейный блок
        try:
            self.kan = KAN(
                width=[input_size, kan_hidden, hidden_size],
                grid=cfg.kan_grid_size,
                k=cfg.kan_spline_order,
            )
            self._use_linear_kan = False
        except Exception:
            self.kan = KANBlock(input_size, kan_hidden, hidden_size)
            self._use_linear_kan = True

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
        )

        if use_temporal_attn:
            # Вариант C: per-timestep self-attention, сохраняет временну́ю структуру
            self.attn = TemporalSelfAttention(hidden_size, num_heads=cfg.attention_heads)
        else:
            # Исходный: глобальный вектор контекста
            self.attn = GlobalAttention(hidden_size)

        self.drop = nn.Dropout(cfg.output_dropout)
        self.head = nn.Linear(hidden_size, 1)

        if with_physics:
            self.snow  = SnowpackLayer()
            self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x_kan: torch.Tensor,
                x_raw: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Parameters
        ----------
        x_kan : [batch, seq_len, input_size]  — нормализованные признаки
        x_raw : [batch, seq_len, n_features]  — сырые данные (нужны для физики)

        Returns
        -------
        q     : [batch, seq_len, 1]
        melt  : [batch, seq_len] или None
        """
        B, T, n_feat = x_kan.shape

        # KAN по каждому временному шагу
        k = self.kan(x_kan.reshape(B * T, n_feat)).reshape(B, T, -1)

        # LSTM
        lstm_out, _ = self.lstm(k)                # [B, T, H]

        # Attention
        if self._use_temporal_attn:
            # Вариант C: temporal self-attention → [B, T, H] (временна́я структура сохранена)
            ctx_seq = self.drop(self.attn(lstm_out))  # [B, T, H]
        else:
            # Исходный: глобальный вектор → expand
            ctx = self.drop(self.attn(lstm_out))           # [B, H]
            ctx_seq = ctx.unsqueeze(1).expand(-1, T, -1)   # [B, T, H]

        q = self.head(ctx_seq)                    # [B, T, 1]

        melt = None
        if self.with_physics and x_raw is not None:
            precip = x_raw[:, :, self.precip_idx]
            temp   = x_raw[:, :, self.temp_idx]
            melt   = self.snow(precip, temp)      # [B, T]
            melt_s = melt.unsqueeze(-1)           # [B, T, 1]
            q = F.relu(q + self.scale * melt_s)
        else:
            q = F.relu(q)

        return q, melt


class RCPIKLA(BaseModel):
    """Residual Compensated Physics-Informed KAN-LSTM with Attention.

    Реализует полную архитектуру Zhou & Liu (2025):

    Stage 1 (физика):
        KAN → LSTM → GlobalAttn → Q̂₁  (с SnowpackLayer)

    Stage 2 (residual):
        KAN → LSTM → GlobalAttn → R̂   (без физики)

    Q_final = Q̂₁ + R̂   (Eq. 20)

    Конфигурационные ключи (дополнительно к PIKLA):
        precip_feature_idx : int  — индекс precipitation в dynamic_inputs (default 0)
        temp_feature_idx   : int  — индекс temperature_mean (default 1)

    CosineAnnealingLR реализуется через learning_rate как dict:
        learning_rate:
          0:  0.001
          20: 0.0005
          40: 0.0002
          60: 0.0001
          70: 0.00005
    """

    module_parts = ['input_layer', 'stage1', 'stage2']

    def __init__(self, cfg: Config):
        super(RCPIKLA, self).__init__(cfg=cfg)

        self.input_layer = InputLayer(cfg)
        inp = self.input_layer.output_size

        pi = cfg.precip_feature_idx   # индекс в тензоре после InputLayer
        ti = cfg.temp_feature_idx

        h1 = cfg.hidden_size
        h2 = cfg.hidden_size // 2
        kh = cfg.kan_hidden_features
        kh2 = max(kh // 2, 1)

        # Stage1 (физика, июль) — GlobalAttention: единый контекстный вектор,
        #   стабильный прогноз первого шага (июль).
        # Stage2 (residual) — TemporalSelfAttention: per-timestep MHA,
        #   корректирует ошибки для шагов +2/+3 (август/сентябрь).
        self.stage1 = PIKLAStage(cfg, inp, h1, kh,
                                 with_physics=True,
                                 precip_idx=pi, temp_idx=ti,
                                 use_temporal_attn=False)
        self.stage2 = PIKLAStage(cfg, inp, h2, kh2,
                                 with_physics=False,
                                 use_temporal_attn=True)

        # Финальная проекция: объединяем h1 и h2 → output_size
        self.fusion = nn.Linear(2, 1)   # смешивает Q1 и R по последней оси
        self.head   = get_head(cfg=cfg, n_in=cfg.hidden_size, n_out=self.output_size)

        self.melting_factor = 0.5       # для совместимости с PIKLA

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # 1. InputLayer
        x_proc = self.input_layer(data)           # [seq, batch, features]
        x_proc = x_proc.transpose(0, 1)          # [batch, seq, features]

        # 2. Тригонометрическое кодирование месяца (как в PIKLA)
        dates = data['date']
        month = np.array([pd.to_datetime(d).month
                          for d in dates.flatten()]).reshape(dates.shape)
        month_t = torch.from_numpy(month).float().to(x_proc.device).unsqueeze(-1)
        month_sin = torch.sin(2 * np.pi * month_t / 12.0)
        month_cos = torch.cos(2 * np.pi * month_t / 12.0)
        x_kan = torch.cat([x_proc, month_sin, month_cos], dim=-1)  # [B, T, F+2]

        # 3. Сырые данные для SnowpackLayer (ненормализованные)
        temp_var = self.cfg.temperature_variable
        prcp_var = self.cfg.precipitation_variable
        if temp_var not in data['x_d']:
            raise KeyError(f"'{temp_var}' не найдена в data['x_d']. Доступно: {list(data['x_d'].keys())}")
        if prcp_var not in data['x_d']:
            raise KeyError(f"'{prcp_var}' не найдена в data['x_d']. Доступно: {list(data['x_d'].keys())}")

        temp = data['x_d'][temp_var]   # [B, T, 1]
        prcp = data['x_d'][prcp_var]   # [B, T, 1]
        B, T, _ = x_kan.shape

        # Стек сырых данных в порядке precip_idx / temp_idx
        # Stage1 будет брать индексы из x_raw
        n_raw = max(self.cfg.precip_feature_idx, self.cfg.temp_feature_idx) + 1
        x_raw = torch.zeros(B, T, n_raw, device=x_kan.device)
        x_raw[:, :, self.cfg.precip_feature_idx] = prcp.squeeze(-1)
        x_raw[:, :, self.cfg.temp_feature_idx]   = temp.squeeze(-1)

        # 4. Stage 1: physics-informed
        q1, melt = self.stage1(x_kan, x_raw)     # [B, T, 1], [B, T]

        # 5. Stage 2: residual compensation
        r, _ = self.stage2(x_kan, None)           # [B, T, 1]

        # 6. Q_final = Q1 + R  (Eq. 20)
        q_final = q1 + r                          # [B, T, 1]

        out = {'y_hat': q_final}
        if melt is not None:
            out['snow_melt'] = melt.unsqueeze(-1)
        return out
