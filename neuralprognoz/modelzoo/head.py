import logging
from typing import Dict

import torch
import torch.nn as nn

from neuralprognoz.utils.config import Config

LOGGER = logging.getLogger(__name__)


def get_head(cfg: Config, n_in: int, n_out: int) -> nn.Module:
    """Получить конкретный модуль head, в зависимости от конфигурации запуска.

    Параметры
    ----------
    cfg : Config
        Конфигурация запуска.
    n_in : int
        Количество входных признаков.
    n_out : int
        Количество выходных признаков.

    Возвращает
    -------
    nn.Module
        Модуль head, как указано в конфигурации запуска.
    """
    if cfg.head.lower() == "regression":
        head = Regression(n_in=n_in, n_out=n_out, activation=cfg.output_activation)
    elif cfg.head.lower() == "gmm":
        head = GMM(n_in=n_in, n_out=n_out)
    elif cfg.head.lower() == "umal":
        head = UMAL(n_in=n_in, n_out=n_out)
    elif cfg.head.lower() == "cmal":
        head = CMAL(n_in=n_in, n_out=n_out)
    elif cfg.head.lower() == "":
        raise ValueError(f"В конфигурации не указан 'head', но он требуется для {cfg.model}")
    else:
        raise NotImplementedError(f"{cfg.head} не реализован или не связан в `get_head()`")

    return head


class Regression(nn.Module):
    """Однослойный regression head с различными выходными активациями.
    
    Параметры
    ----------
    n_in : int
        Количество входных нейронов.
    n_out : int
        Количество выходных нейронов.
    activation : str, optional
        Функция активации выхода. Может быть указана в конфигурации с помощью аргумента `output_activation`. Поддерживаются
        {'linear', 'relu', 'softplus'}. Если не указана (или указана неподдерживаемая функция активации), будет
        использована 'linear' активация по умолчанию.
    """

    def __init__(self, n_in: int, n_out: int, activation: str = "linear"):
        super(Regression, self).__init__()

        # TODO: Добавить поддержку многослойности
        layers = [nn.Linear(n_in, n_out)]
        if activation != "linear":
            if activation.lower() == "relu":
                layers.append(nn.ReLU())
            elif activation.lower() == "softplus":
                layers.append(nn.Softplus())
            else:
                LOGGER.warning(f"Игнорируется выходная активация {activation} и используется 'linear' вместо нее.")
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Выполнить прямой проход Regression head.
        
        Параметры
        ----------
        x : torch.Tensor

        Возвращает
        -------
        Dict[str, torch.Tensor]
            Dictionary containing the model predictions in the 'y_hat' key.
        """
        return {'y_hat': self.net(x)}


class GMM(nn.Module):
    """Сеть Гауссовой Смеси

    Сеть плотности смеси с гауссовыми распределениями в качестве компонентов. Хорошие ссылки - [#]_ и [#]_. Последняя 
    лежит в основе нашей реализации. Как таковая, мы также используем два слоя в head, чтобы предоставить ей 
    дополнительную гибкость, экспоненциальную активацию для оценок дисперсии и softmax для весов.  

    Параметры
    ----------
    n_in : int
        Количество входных нейронов.
    n_out : int
        Количество выходных нейронов. Соответствует 3-кратному количеству компонентов.
    n_hidden : int
        Размер скрытого слоя.
    
    Ссылки
    ----------
    .. [#] C. M. Bishop: Mixture density networks. 1994.
    .. [#] D. Ha: Mixture density networks with tensorflow. blog.otoro.net, 
           URL: http://blog.otoro.net/2015/11/24/mixture-density-networks-with-tensorflow, 2015.
    """

    def __init__(self, n_in: int, n_out: int, n_hidden: int = 100):
        super(GMM, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)
        self._eps = 1e-5

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Выполнить прямой проход GMM head.

        Параметры
        ----------
        x : torch.Tensor
            Выход предыдущей части модели. Он предоставляет основные скрытые переменные для вычисления компонентов GMM.

        Возвращает
        -------
        Dict[str, torch.Tensor]
            Словарь, содержащий параметры смеси и веса; где ключ 'mu' хранит средние значения, ключ
            'sigma' - дисперсии, а ключ 'pi' - веса.
        """
        h = torch.relu(self.fc1(x))
        h = self.fc2(h)

        # разделение выхода на средние значения, дисперсии и веса
        mu, sigma, pi = h.chunk(3, dim=-1)

        return {'mu': mu, 'sigma': torch.exp(sigma) + self._eps, 'pi': torch.softmax(pi, dim=-1)}


class CMAL(nn.Module):
    """Счетная смесь асимметричных распределений Лапласа

    Смесь плотности с лапласовскими распределениями в качестве компонентов.

    CMAL-head использует дополнительный скрытый слой, чтобы придать ему большую выразительность (как и GMM-head).
    CMAL лучше подходит для многих гидрологических условий, так как он легче справляется с асимметриями. Однако он также
    более хрупок, чем GMM, и может чаще вызывать исключения. Подробности о CMAL можно найти в [#]_.

    Параметры
    ----------
    n_in : int
        Количество входных нейронов.
    n_out : int
        Количество выходных нейронов. Соответствует 4-кратному количеству компонентов.
    n_hidden : int
        Размер скрытого слоя.
        
    Ссылки
    ----------
    .. [#] D.Klotz, F. Kratzert, M. Gauch, A. K. Sampson, G. Klambauer, S. Hochreiter, and G. Nearing: 
        Uncertainty Estimation with Deep Learning for Rainfall-Runoff Modelling. arXiv preprint arXiv:2012.14295, 2020.
    """

    def __init__(self, n_in: int, n_out: int, n_hidden: int = 100):
        super(CMAL, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)

        self._softplus = torch.nn.Softplus(2)
        self._eps = 1e-5

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Выполнить прямой проход CMAL head.

        Параметры
        ----------
        x : torch.Tensor
            Выход предыдущей части модели. Он предоставляет основные скрытые переменные для вычисления компонентов CMAL.

        Возвращает
        -------
        Dict[str, torch.Tensor]
            Словарь, содержащий параметры смеси и веса; где ключ 'mu' хранит средние значения,
            ключ 'b' - параметры масштаба, ключ 'tau' - параметры асимметрии, и ключ 'pi' - веса.
        """
        h = torch.relu(self.fc1(x))
        h = self.fc2(h)

        m_latent, b_latent, t_latent, p_latent = h.chunk(4, dim=-1)

        # навязывание свойств параметрам и весам компонентов:
        m = m_latent  # без ограничений (в зависимости от настроек m>0 может быть полезно)
        b = self._softplus(b_latent) + self._eps  # масштаб > 0 (softplus хорошо работал в тестах)
        t = (1 - self._eps) * torch.sigmoid(t_latent) + self._eps  # 0 > tau > 1
        p = (1 - self._eps) * torch.softmax(p_latent, dim=-1) + self._eps  # sum(pi) = 1 & pi > 0

        return {'mu': m, 'b': b, 'tau': t, 'pi': p}


class UMAL(nn.Module):
    """Несчетная смесь асимметричных распределений Лапласа.

    Неявное приближение к сети плотности смеси с распределениями Лапласа, которое не требует предварительного
    указания количества компонентов. Дополнительный скрытый слой используется для придания голове большей выразительности.
    Общие сведения об UMAL можно найти в [#]_. Основное отличие между их реализацией 
    и нашей заключается в функции привязки для параметра масштаба (b). Масштаб должен быть ограничен снизу. Оригинальная реализация UMAL 
    использует привязку на основе elu. Однако в нашем эксперименте это привело к недостоверным прогнозам
    (слишком большие дисперсии). Поэтому мы выбрали специальную функцию привязки, которая ограничивает масштаб снизу и 
    сверху с помощью сигмоиды. Весьма вероятно, что это необходимо адаптировать для ненормализованных выходов.   

    Параметры
    ----------
    n_in : int
        Количество входных нейронов.
    n_out : int
        Количество выходных нейронов. Соответствует 2-кратному размеру выхода, так как предсказываются и параметры масштаба.
    n_hidden : int
        Размер скрытого слоя.

    Ссылки
    ----------
    .. [#] A. Brando, J. A. Rodriguez, J. Vitria, and A. R. Munoz: Modelling heterogeneous distributions 
        with an Uncountable Mixture of Asymmetric Laplacians. Advances in Neural Information Processing Systems, 
        pp. 8838-8848, 2019.
    """

    def __init__(self, n_in: int, n_out: int, n_hidden: int = 100):
        super(UMAL, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)
        self._upper_bound_scale = 0.5  # этот параметр был найден эмпирически при тестировании UMAL на ограниченном наборе бассейнов
        self._eps = 1e-5

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Выполнить прямой проход UMAL head.

        Параметры
        ----------
        x : torch.Tensor
            Выход предыдущей части модели. Он предоставляет основные скрытые переменные для вычисления компонентов UMAL.

        Возвращает
        -------
        Dict[str, torch.Tensor]
            Dictionary containing the means ('mu') and scale parameters ('b') to parametrize the asymmetric Laplacians.
        """
        h = torch.relu(self.fc1(x))
        h = self.fc2(h)

        m_latent, b_latent = h.chunk(2, dim=-1)

        # навязывание свойств параметрам и весам компонентов:
        m = m_latent  # без ограничений (в зависимости от настроек m>0 может быть полезно)
        b = self._upper_bound_scale * torch.sigmoid(b_latent) + self._eps  # ограничение масштаба с двух сторон.
        return {'mu': m, 'b': b}
