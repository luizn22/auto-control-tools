from typing import Union

from ..first_order_model import FirstOrderModel
from .base import BaseModelIdentification
from ...utils.data import DataUtils
from ...utils.data_input import DataInputUtils


class ZieglerNicholsModelIdentification(BaseModelIdentification):
    """
    Método de Identicação Ziegler Nichols.

    Este é um método clássico da literatura para identificação de modelos dinâmicos de processos industriais.
    Desenvolvido por Ziegler e Nichols em 1942 :footcite:p:`CoelhoChapter4`.

    O processo de identificação de modelo pelo método de Ziegler-Nichols envolve a análise da resposta ao
    degrau de um sistema, com base em uma função de transferência de primeira ordem com tempo morto (FOPDT).
    Os parâmetros chave a serem determinados são o ganho estático (:math:`K`), a constante de tempo (:math:`\\tau`),
    e o atraso de transporte (:math:`\\theta`) :footcite:p:`CoelhoChapter4`.


    Os parâmetros são calculados conforme indicado pela figura a seguir:

    .. figure:: ../image_resources/zn_hg_ident_meth.png
        :align: center

        Figura 4.3 - Métodos de ZN e HAG para a modelagem de processos de primeira ordem :footcite:p:`CoelhoChapter4`.

    A reta traçada corresponde à tangente no ponto de máxima inclinação da curva de reação.

    A constante de tempo :math:`\\tau` é determinada pelo intervalo de tempo entre :math:`t_1`, e o instante
    :math:`t_3`, onde a reta tangente toca o eixo :math:`t`, e onde cruza com a reta :math:`y(t)` = :math:`y_f`,
    respectivamente.
    O valor de  :math:`\\theta` é considerado como :math:`t_1`, o intervalo entre a aplicação do sinal degrau e o
    momento em que a reta tangente toca o eixo :math:`t`. Por fim o valor de :math:`K` pode ser obtido a través da
    equação,

    .. math::

        K = \\frac{\\Delta y}{\\Delta u}

    Ou seja :math:`y_f` dividido pelo valor do sinal degrau :footcite:p:`CoelhoChapter4`.

    Detalhes sobre a implementação do cálculo dos parâmetros podem ser visualizados na documentação do método
    :meth:`get_model`.

    Os parâmetros obtidos são utilizados para instanciar um objeto da classe :class:`FirstOrderModel`, que se
    especializa em modelos como o gerado por esse método, e possibilita o uso de métodos de aproximação de ganhos de
    controlador PID também especializados neste tipo de modelo.

    Notes
    -----
    - .. include:: ../shared/zn_hag_noise_obs.rst

    - .. include:: ../shared/estim_params_obs.rst

    - .. include:: ../shared/non_lin_obs.rst


    Referências:
        .. footbibliography::


    Examples
    --------
    >>> file_path = r'path/to/layout/data_input.csv'
    >>> model = ZieglerNicholsModelIdentification.get_model(file_path)
    >>> model.view.print_tf()

    .. math::
        \\frac{ 168.26 }{ 2.63s + 1 }e^{-4s}

    >>> model.view.print_model_step_response_data()
    {'Overshoot': 0,
     'Peak': 167.49042890412443,
     'PeakTime': 18.16101421851958,
     'RiseTime': 5.826658728441698,
     'SettlingMax': 168.2608695652174,
     'SettlingMin': 151.5076223670495,
     'SettlingTime': 14.30179869708417,
     'SteadyStateValue': 168.2608695652174,
     'Undershoot': 3.166596639477343}

    >>> model.view.plot_model_step_response_graph()

    .. image:: ../image_resources/zn_ident_plot.png
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
        """
        Calcula um modelo baseado nos dados fornecidos.


        * Utiliza :meth:`DataInputUtils.get_model_data_default` para obter a tabela (pandas.DataFrame) com os dados de
          resposta do modelo.
        * Então obtém a pandas.Series :attr:`tf_data` (lista cujo index representa o tempo e os valores a saida)
          e o valor do sinal degrau através do método :meth:`DataUtils.setup_data_default`.
        * Obtém o valor de regime e o momento de entrada em valor de regime através do método
          :meth:`DataUtils.get_vreg`.
        * Obtém o valor da inclinação da reta tangente e o momento em que a reta encosta na curva :math:`y(t)`
          através do método :meth:`DataUtils.get_max_tan`.
        * Encontra o valor de :math:`y(t)` quando :math:`t` é igual ao momento em que a reta encosta na curva
          :math:`y(t)`.
        * Utiliza a inclinação da reta tangente e a localização do ponto de encontro dela com a curva :math:`y(t)`
          para encontrar os valores de :math:`t_1` e :math:`t_3` através do método
          :meth:`DataUtils.get_time_from_inclination` e dos valores de :math:`y(t)` referentes aos pontos desejados,
          :math:`y(t) = 0` e :math:`y(t) = y_f`, respectivamente.
        * Obtém :math:`K` dividindo o valor de regime, :math:`y_f`, pelo valor do sinal degrau.
        * Com os valores de :math:`t_1` e :math:`t_3` em mãos calcula o valor de :attr:`tau` e :attr:`theta`,
          sendo :math:`\\tau = t_3 - t_1` e :math:`\\theta = t_1`.
        * Dependendo do valor de :attr:`theta` e de :paramref:`ignore_delay_threshold` zera o valor de :attr:`theta`.
        * Instancia um objeto da classe :class:`FirstOrderModel` com os valores obtidos.

        Parameters
        ----------
        path : str
            Caminho até o arquivo a ser lido. O leiaute pode ser obtido através de :meth:`get_data_input_layout`.
        sample_time : float, optional
            Valor do invervalo de amostragem. Caso informado, o intervalo de amostragem é considerado constante e
            igual ao valor fornecido.
        step_signal : float, optional
            Valor do sinal degrau de entrada. Se informado é considerado que o sinal está ativo em todos os momentos
            nos dados recebidos.
        ignore_delay_threshold : float, optional
            Valor mínimo de theta para que ele não seja zerado.
        settling_time_threshold : float, optional
            Percentual de desvio do valor de regime considerado do cálculo do tempo de acomodação.

        Returns
        -------
        Objeto :class:`FirstOrderModel` referente ao modelo gerado pelo método.
        """
        df = DataInputUtils.get_model_data_default(path, sample_time, step_signal)
        tf_data, step_signal = DataUtils.setup_data_default(df, sample_time, step_signal)

        idx_vreg, vreg = DataUtils.get_vreg(tf_data, settling_time_threshold=settling_time_threshold)
        idx_tan, tan = DataUtils.get_max_tan(tf_data)
        tan_point_value = tf_data.loc[tf_data.index == idx_tan].iloc[0]

        t1 = DataUtils.get_time_from_inclination(idx_tan, tan_point_value, tan_point_value, 0)
        t3 = DataUtils.get_time_from_inclination(idx_tan, tan_point_value, tan_point_value, vreg)

        K = vreg / step_signal
        tau = t3 - t1
        theta = t1

        if ignore_delay_threshold is not None and theta < ignore_delay_threshold:
            theta = 0

        return FirstOrderModel(K, tau, theta, source_data=tf_data)
