from typing import Dict

try:
    from mamba_ssm import Mamba as Mamba_SSM
except ModuleNotFoundError:
    Mamba_SSM = None
import torch
import torch.nn as nn

from neuralprognoz.modelzoo.inputlayer import InputLayer
from neuralprognoz.modelzoo.head import get_head
from neuralprognoz.modelzoo.basemodel import BaseModel
from neuralprognoz.utils.config import Config


class Mamba(BaseModel):
    """Класс модели Mamba, основанный на https://github.com/state-spaces/mamba/tree/main.

    Этот класс реализует Mamba SSM с объединенной головой модели, как указано в файле конфигурации, и переходным
    слоем для обеспечения соответствия входных размеров спецификациям mamba_ssm. Пожалуйста, прочитайте документацию mamba
    для лучшего понимания требуемых гиперпараметров.

    Параметры
    ----------
    cfg : Config
        Конфигурация запуска.
    """
    # Укажите подмодули модели, которые впоследствии можно использовать для тонкой настройки. Имена должны соответствовать атрибутам класса
    module_parts = ['embedding_net', 'transition_layer', 'mamba', 'head']

    def __init__(self, cfg: Config):
        super(Mamba, self).__init__(cfg=cfg)

        self.embedding_net = InputLayer(cfg)

        # использование линейного слоя для перемещения из emdedded_layer dims в указанный скрытый размер
        self.transition_layer = nn.Linear(self.embedding_net.output_size, self.cfg.hidden_size)

        if Mamba_SSM is None:
            raise ModuleNotFoundError(
                f"mamba_ssm, и зависимости, требуются. Пожалуйста, выполните: pip install mamba_ssm causal-conv1d>=1.1.0"
            )
        else:
            self.mamba = Mamba_SSM(
                d_model=self.cfg.hidden_size,
                d_state=self.cfg.mamba_d_state,
                d_conv=self.cfg.mamba_d_conv,
                expand=self.cfg.mamba_expand,
            )

        self.dropout = nn.Dropout(p=cfg.output_dropout)

        self.head = get_head(cfg=cfg, n_in=cfg.hidden_size, n_out=self.output_size)

    def forward(self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Выполняет прямой проход по модели CudaLSTM.

        Параметры
        ----------
        data : dict[str, torch.Tensor | dict[str, torch.Tensor]]
            Словарь, содержащий входные признаки в виде пар ключ-значение.

        Возвращает
        -------
        Dict[str, torch.Tensor]
            Словарь, содержащий выходные данные модели и промежуточные состояния.
                - `y_hat`: предсказания модели размерностью [batch size, sequence length, number of target variables].
                - `h_n`: скрытое состояние в последнем временном шаге последовательности размерностью [batch size, 1, hidden size].
                - `c_n`: состояние ячейки в последнем временном шаге последовательности размерностью [batch size, 1, hidden size].
        """
        # возможно, передать динамические и статические входы через embedding слои, затем объединить их
        x_d = self.embedding_net(data)

        # изменение размерности в соответствии с требованиями mamba:
        x_d_transition = self.transition_layer(x_d)

        mamba_output = self.mamba(x_d_transition)

        # изменение размерности до [batch_size, seq, n_hiddens]
        mamba_output = mamba_output.transpose(0, 1)

        pred = {'y_hat': mamba_output}
        pred.update(self.head(self.dropout(mamba_output)))
        return pred
