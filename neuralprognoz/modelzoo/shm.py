import torch
from typing import Dict, Union
from neuralprognoz.modelzoo.baseconceptualmodel import BaseConceptualModel
from neuralprognoz.utils.config import Config


class SHM(BaseConceptualModel):
    """Модифицированная версия гидрологической модели SHM [#]_ с динамической параметризацией.
    
    The SHM получает динамические параметры, заданные моделью глубокого обучения. Этот класс имеет два свойства, которые 
    определяют начальные условия внутренних состояний модели (buckets) и диапазоны, в которых модель 
    параметры разрешено варьировать во время оптимизации.

    Параметры
    ----------
    cfg : Config
        Конфигурация запуска.

    Ссылки
    ----------
    .. [#] Ehret, U., van Pruijssen, R., Bortoli, M., Loritz, R., Azmi, E., and Zehe, E.: Adaptive clustering: reducing
        the computational costs of distributed (hydrological) modelling by exploiting time-variable similarity among 
        model elements, Hydrology and Earth System Sciences, 24, 4389–4411, https://doi.org/10.5194/hess-24-4389-2020, 
        2020.
    """

    def __init__(self, cfg: Config):
        super(SHM, self).__init__(cfg=cfg)

    def forward(self, x_conceptual: torch.Tensor,
                lstm_out: torch.Tensor) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Выполняет прямой проход на модели SHM. 
        
        В этом прямом проходе все элементы пакета обрабатываются параллельно.

        Параметры
        ----------
        x_conceptual: torch.Tensor
            Тензор размера [batch_size, time_steps, n_inputs]. Размер batch_size связан с определенным бассейном и определенным
            периодом прогнозирования. Параметр time_steps обозначает количество временных шагов (например, дней), 
            в течение которых будет запускаться наша концептуальная модель. 
            Параметр n_inputs обозначает количество динамических параметров, используемых для запуска концептуальной модели 
            (например, осадки, температура...)

        lstm_out: torch.Tensor
            Тензор размера [batch_size, time_steps, n_parameters]. Тензор поступает из модели глубокого обучения и будет
            использован для получения динамической параметризации концептуальной модели.

        Возвращает
        -------
        Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]
            - y_hat: torch.Tensor
                Суммарный сток
            - parameters: Dict[str, torch.Tensor]
                Динамическая параметризация концептуальной модели
            - internal_states: Dict[str, torch.Tensor]]
                Временная эволюция внутренних состояний концептуальной модели
        """
        # Получаем параметры модели
        parameters = self._get_dynamic_parameters_conceptual(lstm_out=lstm_out)

        # Инициализируем структуры для хранения информации
        states, out = self._initialize_information(conceptual_inputs=x_conceptual)

        # Инициализируем константы
        zero = torch.tensor(0.0, dtype=torch.float32, device=x_conceptual.device)
        one = torch.tensor(1.0, dtype=torch.float32, device=x_conceptual.device)
        klu = torch.tensor(0.90, requires_grad=False, dtype=torch.float32,
                           device=x_conceptual.device)  # коэффициент коррекции землепользования [-]

        # Вспомогательные векторы для ускорения выполнения гидрологической модели
        t_mean = (x_conceptual[:, :, 2] + x_conceptual[:, :, 3]) / 2
        temp_mask = t_mean < 0
        snow_melt = t_mean * parameters['dd']
        snow_melt[temp_mask] = zero
        # Жидкие осадки:
        liquid_p = x_conceptual[:, :, 0].clone()
        liquid_p[temp_mask] = zero
        # Твердые осадки (снег):
        snow = x_conceptual[:, :, 0].clone()
        snow[~temp_mask] = zero
        # Точка устойчивого завядания, используемая в ET:
        pwp = torch.tensor(0.8, dtype=torch.float32, device=x_conceptual.device) * parameters['sumax']

        # Резервуары
        ss = torch.tensor(self.initial_states['ss'], dtype=torch.float32,
                          device=x_conceptual.device).repeat(x_conceptual.shape[0])
        sf = torch.tensor(self.initial_states['sf'], dtype=torch.float32,
                          device=x_conceptual.device).repeat(x_conceptual.shape[0])
        su = torch.tensor(self.initial_states['su'], dtype=torch.float32,
                          device=x_conceptual.device).repeat(x_conceptual.shape[0])
        si = torch.tensor(self.initial_states['si'], dtype=torch.float32,
                          device=x_conceptual.device).repeat(x_conceptual.shape[0])
        sb = torch.tensor(self.initial_states['sb'], dtype=torch.float32,
                          device=x_conceptual.device).repeat(x_conceptual.shape[0])

        # Запускаем гидрологическую модель для каждого временного шага
        for j in range(x_conceptual.shape[1]):
            # Модуль снега --------------------------
            qs_out = torch.minimum(ss, snow_melt[:, j])
            ss = ss - qs_out + snow[:, j]
            qsp_out = qs_out + liquid_p[:, j]

            # Разделяем талый снег + дождь на приток в резервуар быстрого потока и ненасыщенный резервуар ------
            qf_in = torch.maximum(zero, qsp_out - parameters['f_thr'][:, j])
            qu_in = torch.minimum(qsp_out, parameters['f_thr'][:, j])

            # Модуль быстрого потока ----------------------
            sf = sf + qf_in
            qf_out = sf / parameters['kf'][:, j]
            sf = sf - qf_out

            # Ненасыщенная зона----------------------
            psi = (su / parameters['sumax'][:, j])**parameters['beta'][:, j]  # [-]
            su_temp = su + qu_in * (1 - psi)
            su = torch.minimum(su_temp, parameters['sumax'][:, j])
            qu_out = qu_in * psi + torch.maximum(zero, su_temp - parameters['sumax'][:, j])  # [mm]
            # Испарение -------------------
            ktetha = su / parameters['sumax'][:, j]
            et_mask = su <= pwp[:, j]
            ktetha[~et_mask] = one
            ret = x_conceptual[:, j, 1] * klu * ktetha  # [мм]
            su = torch.maximum(zero, su - ret)  # [мм]

            # Резервуар межпластового стока ------------------
            qi_in = qu_out * parameters['perc'][:, j]  # [мм]
            si = si + qi_in  # [мм]
            qi_out = si / parameters['ki'][:, j]  # [мм]
            si = si - qi_out  # [мм]

            # Резервуар грунтового стока -------------------
            qb_in = qu_out * (1.0 - parameters['perc'][:, j])  # [мм]
            sb = sb + qb_in  # [мм]
            qb_out = sb / parameters['kb'][:, j]  # [мм]
            sb = sb - qb_out

            # Сохраняем временную эволюцию внутренних состояний
            states['ss'][:, j] = ss
            states['sf'][:, j] = sf
            states['su'][:, j] = su
            states['si'][:, j] = si
            states['sb'][:, j] = sb

            # Суммарный сток
            out[:, j, 0] = qf_out + qi_out + qb_out  # [мм]

        return {'y_hat': out, 'parameters': parameters, 'internal_states': states}

    @property
    def initial_states(self):
        return {'ss': 0.0, 'sf': 1.0, 'su': 5.0, 'si': 10.0, 'sb': 15.0}

    @property
    def parameter_ranges(self):
        return {
            'dd': [0.0, 10.0],
            'f_thr': [10.0, 60.0],
            'sumax': [20.0, 700.0],
            'beta': [1.0, 6.0],
            'perc': [0.0, 1.0],
            'kf': [1.0, 20.0],
            'ki': [1.0, 100.0],
            'kb': [10.0, 1000.0]
        }
