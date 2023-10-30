from typing import Union

from ..first_order_model import FirstOrderModel
from .base import BaseModelIdentification
from ...utils.data import DataUtils
from ...utils.data_input import DataInputUtils


class SundaresanKrishnaswamyModelIdentification(BaseModelIdentification):
    """
    Método de Identicação Sundaresan Krishnaswamy.

    Este é um método clássico da literatura para identificação de modelos dinâmicos de processos industriais.
    Desenvolvido por Sundaresan e Krishnaswamy em 1977 :footcite:p:`CoelhoChapter4`.

    O processo de identificação de modelo pelo método de Sundaresan e Krishnaswamy envolve a análise da resposta ao
    degrau de um sistema, com base em uma função de transferência de primeira ordem com tempo morto (FOPDT).
    Os parâmetros chave a serem determinados são o ganho estático (:math:`K`), a constante de tempo (:math:`\\tau`),
    e o atraso de transporte (:math:`\\theta`) :footcite:p:`CoelhoChapter4`.


    Os parâmetros são calculados conforme indicado pela figura a seguir:

    .. figure:: ../image_resources/sd_kr_ident_meth.png
        :align: center

        Método de Sundaresan e Krishnaswamy para a modelagem de processos de primeira ordem
        :footcite:p:`CoelhoChapter4`.

    No metodo Sundaresan Krishnaswamy são obtidas as constantes de tempo :math:`t_1` e o instante :math:`t_2`,
    que correspondem as passagens da resposta pelos pontos :math:`y(t) = y(0) + 0.353y(\\infty)` e
    :math:`y(t) = y(0) + 0.853y(\\infty)`, respectivamente.
    As constantes podem então ser utilizadas para o cálculo da constante de tempo :math:`\\tau` e
    do valor de :math:`\\theta`, conforme as seguintes equações :footcite:p:`CoelhoChapter4`:

    .. math::

        \\tau = 0.67*(t_2 - t_1)

    .. math::

        \\theta = 1.3t_1 - 0.29t_2

    Por fim o valor de :math:`K` pode ser obtido a través da equação,

    .. math::

        K = \\frac{\\Delta y}{\\Delta u}

    Ou seja :math:`y(0) + y(\\infty)` dividido pelo valor do sinal degrau :footcite:p:`CoelhoChapter4`.

    Detalhes sobre a implementação do cálculo dos parâmetros podem ser visualizados na documentação do método
    :meth:`get_model`.

    Os parâmetros obtidos são utilizados para instanciar um objeto da classe :class:`FirstOrderModel`, que se
    especializa em modelos como o gerado por esse método, e possibilita o uso de métodos de aproximação de ganhos de
    controlador PID também especializados neste tipo de modelo.

    Notes
    -----
    - .. include:: ../shared/estim_params_obs.rst

    - .. include:: ../shared/non_lin_obs.rst

    Referências:
        .. footbibliography::


    Examples
    --------
    >>> file_path = r'path/to/layout/data_input.csv'
    >>> model = SundaresanKrishnaswamyModelIdentification.get_model(file_path)
    >>> model.view.print_tf()

    .. math::
        \\frac{ 168.26 }{ 8.71s + 1 }e^{-1.28s}

    >>> model.view.print_model_step_response_data()
    {'Overshoot': 0,
     'Peak': 168.0659721995617,
     'PeakTime': 60.166548479934534,
     'RiseTime': 19.15051301335547,
     'SettlingMax': 168.2608695652174,
     'SettlingMin': 151.47519007461054,
     'SettlingTime': 35.36784618542485,
     'SteadyStateValue': 168.2608695652174,
     'Undershoot': 0.3157730699025619}

    >>> model.view.plot_model_step_response_graph()

    .. image:: ../image_resources/sn_kr_ident_plot.png
    """
    @classmethod
    def get_model(
            cls,
            path: str,
            sample_time: Union[None, float] = None,
            step_signal: Union[None, float] = None,
            ignore_delay_threshold: Union[None, float] = 0.5,
            settling_time_threshold: float = 0.02

    ) -> FirstOrderModel:
        df = DataInputUtils.get_model_data_default(path, sample_time, step_signal)
        tf_data, step_signal = DataUtils.setup_data_default(df, sample_time, step_signal)

        idx_vreg, vreg = DataUtils.get_vreg(tf_data, settling_time_threshold=settling_time_threshold)

        t1 = tf_data[tf_data == min(tf_data[tf_data >= vreg*0.353])].index[0]
        t2 = tf_data[tf_data == min(tf_data[tf_data >= vreg*0.853])].index[0]

        K = vreg / step_signal
        tau = 0.67*(t2 - t1)
        theta = 1.3 * t1 - 0.29 * t2

        if ignore_delay_threshold is not None and theta < ignore_delay_threshold:
            theta = 0

        return FirstOrderModel(K, tau, theta, source_data=tf_data)
