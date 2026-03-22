from typing import Dict
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

try:
    from kan import KAN
except ImportError:
    raise ImportError("Для использования модели PIKLA/RCPIKLA нужно установить библиотеку pykan: pip install pykan")

from neuralprognoz.modelzoo.basemodel import BaseModel
from neuralprognoz.modelzoo.inputlayer import InputLayer
from neuralprognoz.modelzoo.head import get_head
from neuralprognoz.utils.config import Config


class PIKLA(BaseModel):
    """
    Physics-Informed KAN-LSTM with Attention.
    Эта модель реализует основную часть архитектуры из статьи Zhou & Liu (2025),
    но без двухэтапной компенсации остатков для упрощения интеграции.
    """
    # Части модели для возможной тонкой настройки
    module_parts = ['input_layer', 'kan_feature_extractor', 'lstm', 'attention', 'head']

    def __init__(self, cfg: Config):
        super(PIKLA, self).__init__(cfg=cfg)

        # Слой для предобработки входных данных
        self.input_layer = InputLayer(cfg)

        # KAN как экстрактор нелинейных признаков
        # Размеры KAN - задать в config.yml
        self.kan_feature_extractor = KAN(
            width=[self.input_layer.output_size, cfg.kan_hidden_features, cfg.hidden_size],
            grid=cfg.kan_grid_size,
            k=cfg.kan_spline_order
        )

        # LSTM для обработки временных последовательностей
        self.lstm = nn.LSTM(input_size=cfg.hidden_size, hidden_size=cfg.hidden_size, batch_first=True)

        # Механизм внимания для взвешивания скрытых состояний LSTM
        self.attention = nn.MultiheadAttention(embed_dim=cfg.hidden_size, num_heads=cfg.attention_heads, batch_first=True)
        self.attention_fc = nn.Linear(cfg.hidden_size, cfg.hidden_size) # Для агрегации выходов внимания

        # Выходной слой (голова)
        self.head = get_head(cfg=cfg, n_in=cfg.hidden_size, n_out=self.output_size)
        
        # Параметры для физической модели снега
        # Коэффициент таяния (melting factor) в мм/°C/день
        self.melting_factor = 0.5

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Прямой проход модели.
        """
        # 1. Предобработка входных данных
        x_d_processed = self.input_layer(data) # shape: [seq_len, batch_size, features]
        x_d_processed = x_d_processed.transpose(0, 1) # shape: [batch_size, seq_len, features]

        # 2. Тригонометрическое кодирование
        # Получение дат и создание тензоров месяца на том же устройстве, что и входные данные.
        dates = data['date'] # shape: [batch_size, seq_len]
        month = np.array([pd.to_datetime(d).month for d in dates.flatten()]).reshape(dates.shape)
        # Использование .device от существующего тензора для создания новых тензоров.
        month_tensor = torch.from_numpy(month).float().to(x_d_processed.device).unsqueeze(-1)
        
        month_sin = torch.sin(2 * np.pi * month_tensor / 12.0)
        month_cos = torch.cos(2 * np.pi * month_tensor / 12.0)
        
        # Объединение с основными входными данными
        x_d_final = torch.cat([x_d_processed, month_sin, month_cos], dim=-1)

        # 3. KAN-экстрактор признаков
        batch_size, seq_len, n_features = x_d_final.shape
        x_kan = self.kan_feature_extractor(x_d_final.reshape(batch_size * seq_len, n_features))
        x_kan = x_kan.view(batch_size, seq_len, self.cfg.hidden_size)

        # 4. Физический модуль снега
        temp_var = self.cfg.temperature_variable
        prcp_var = self.cfg.precipitation_variable

        if temp_var not in data['x_d']:
            raise KeyError(f"Переменная для температуры '{temp_var}' не найдена. Доступные ключи: {list(data['x_d'].keys())}")
        if prcp_var not in data['x_d']:
            raise KeyError(f"Переменная для осадков '{prcp_var}' не найдена. Доступные ключи: {list(data['x_d'].keys())}")
            
        # Убираем .to(self.device), так как тензоры уже на нужном устройстве
        temp = data['x_d'][temp_var] 
        prcp = data['x_d'][prcp_var]

        # Создаем тензор snowpack на том же устройстве, что и входные данные
        snowpack = torch.zeros(batch_size, 1, device=x_d_final.device)
        melt_t_list = []

        for t in range(seq_len):
            snowfall = torch.where(temp[:, t, :] < 0, prcp[:, t, :], torch.zeros_like(prcp[:, t, :]))
            potential_melt = torch.clamp(temp[:, t, :], min=0) * self.melting_factor
            snowpack += snowfall
            actual_melt = torch.min(snowpack, potential_melt)
            melt_t_list.append(actual_melt)
            snowpack -= actual_melt
            
        melt_t = torch.cat(melt_t_list, dim=1).unsqueeze(-1)

        # 5. LSTM
        lstm_output, _ = self.lstm(x_kan)
        
        # 6. Внимание
        attention_output, _ = self.attention(lstm_output, lstm_output, lstm_output)
        
        # 7. Комбинация с физикой и выходной слой
        q_lstm = self.head(attention_output)['y_hat']
        q_final = torch.relu(q_lstm + melt_t.expand_as(q_lstm))
        
        return {
            'y_hat': q_final,
            'snow_melt': melt_t
        }