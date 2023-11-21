from typing import Union

from ..first_order_model import FirstOrderModel
from .base import BaseModelIdentification
from ...utils.data import DataUtils
from ...utils.data_input import DataInputUtils


class SmithModelIdentification(BaseModelIdentification):
    """
    Método de Identicação Smith.

    Este é um método clássico da literatura para :term:`Identificação` de modelos dinâmicos de processos industriais.
    Desenvolvido por Smith em 1985 :footcite:p:`CoelhoChapter4`.

    O processo de identificação de :term:`Modelo` pelo método de Smith envolve a análise da resposta ao
    degrau de um :term:`Sistema`, com base em uma :term:`Função de Transferência` de primeira ordem com tempo morto
    (FOPDT).
    Os parâmetros chave a serem determinados são o ganho estático (:math:`K`), a constante de tempo (:math:`\\tau`),
    e o atraso de transporte (:math:`\\theta`) :footcite:p:`CoelhoChapter4`.


    Os parâmetros são calculados conforme indicado pela figura a seguir:

    .. figure:: ../image_resources/sm_ident_meth.png
        :align: center

        Método de Smith para a modelagem de processos de primeira ordem :footcite:p:`CoelhoChapter4`.

    No metodo Smith são obtidas as constantes de tempo :math:`t_1` e o instante :math:`t_2`, que correspondem as
    passagens da resposta pelos pontos :math:`y(t) = y(0) + 0.283y(\\infty)` e :math:`y(t) = y(0) + 0.632y(\\infty)`,
    respectivamente. As constantes podem então ser utilizadas para o cálculo da constante de tempo :math:`\\tau` e
    do valor de :math:`\\theta`, conforme as seguintes equações :footcite:p:`CoelhoChapter4`:

    .. math::

        \\tau = 1.5*(t_2 - t_1)

    .. math::

        \\theta = t_2 - \\tau

    Por fim o valor de :math:`K` pode ser obtido a través da equação,

    .. math::

        K = \\frac{\\Delta y}{\\Delta u}

    Ou seja :math:`y(0) + y(\\infty)` dividido pelo valor do sinal degrau :footcite:p:`CoelhoChapter4`.

    Detalhes sobre a implementação do cálculo dos parâmetros podem ser visualizados na documentação do método
    :meth:`get_model`.

    Os parâmetros obtidos são utilizados para instanciar um objeto da classe :class:`FirstOrderModel`, que se
    especializa em modelos como o gerado por esse método, e possibilita o uso de métodos de :term:`Aproximação de Ganhos` de
    :term:`Controlador PID` também especializados neste tipo de modelo.

    Notes
    -----
    - .. include:: ../shared/estim_params_obs.rst

    - .. include:: ../shared/non_lin_obs.rst

    Referências:
        .. footbibliography::


    Examples
    --------
    >>> file_path = r'path/to/layout/data_input.csv'
    >>> model = act.SmithModelIdentification.get_model(file_path)
    >>> model.view.print_tf()

    .. math::
        \\frac{ 168.26 }{ 10.5s + 1 }e^{-0.5s}

    >>> model.view.print_model_step_response_data()
    {'Overshoot': 0,
     'Peak': 168.08440243680673,
     'PeakTime': 72.53143042931237,
     'RiseTime': 23.06960879828099,
     'SettlingMax': 168.2608695652174,
     'SettlingMin': 151.43956563550034,
     'SettlingTime': 41.58333258859957,
     'SteadyStateValue': 168.2608695652174,
     'Undershoot': 0.10239175640599915}

    >>> model.view.plot_model_step_response_graph()

    .. image:: ../image_resources/sm_ident_plot.png
    """

    @classmethod
    def get_model(
            cls,
            path: str,
            sample_time: Union[None, float] = None,
            step_signal: Union[None, float] = None,
            ignore_delay_threshold: Union[None, float] = 0.5,
            settling_time_threshold: float = 0.02,
            use_lin_filter: bool = False,
            linfilter_smoothness: int = 5

    ) -> FirstOrderModel:
        """
        Calcula um modelo baseado nos dados fornecidos.


        * Utiliza :meth:`DataInputUtils.get_model_data_default` para obter a tabela (pandas.DataFrame) com os dados de
          resposta do modelo.
        * Então obtém a pandas.Series :attr:`tf_data` (lista cujo index representa o tempo e os valores a saida)
          e o valor do sinal degrau através do método :meth:`DataUtils.setup_data_default`.
        * Obtém o valor de regime e o momento de entrada em valor de regime através do método
          :meth:`DataUtils.get_vreg`.
        * Calcula :math:`t_1` e :math:`t_2` baseado no instante em que a curva atinge 28.3% e 63.2% do valor de regime,
          respectivamente.
        * Obtém :math:`K` dividindo o valor de regime, pelo valor do sinal degrau.
        * Com os valores de :math:`t_1` e :math:`t_2` em mãos, calcula o valor de :attr:`tau` e :attr:`theta`,
          sendo :math:`\\tau = 1.5*(t_2 - t_1)` e :math:`\\theta = t_2 - \\tau`.
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
        use_lin_filter : bool, optional
            Em casos cujo sinal de saida é muito ruidoso, um filtro linear, para reduzir a oscilação dos dados é
            necessário para realizar a identificação. Se esse parâmetro for informado como verdadeiro, um filtro linear
            :meth:`DataUtils.linfilter` será aplicado sobre os dados.
        linfilter_smoothness : bool, optional
            Valor inteiro referente a intensidade do filtro linear. Valores maiores reduzem mais o ruído, contudo
            tornam os dados mais distantes da realidade.

        Returns
        -------
        Objeto :class:`FirstOrderModel` referente ao modelo gerado pelo método.
        """
        df = DataInputUtils.get_model_data_default(path, sample_time, step_signal)
        tf_data, step_signal = DataUtils.setup_data_default(
            df, sample_time, step_signal,
            use_lin_filter=use_lin_filter,
            linfilter_smoothness=linfilter_smoothness
        )

        idx_vreg, vreg = DataUtils.get_vreg(tf_data, settling_time_threshold=settling_time_threshold)

        t1 = tf_data[tf_data == min(tf_data[tf_data >= vreg * 0.283])].index[0]
        t2 = tf_data[tf_data == min(tf_data[tf_data >= vreg * 0.632])].index[0]

        K = vreg / step_signal
        tau = 1.5 * (t2 - t1)
        theta = t2 - tau

        if ignore_delay_threshold is not None and theta < ignore_delay_threshold:
            theta = 0

        return FirstOrderModel(K, tau, theta, source_data=tf_data, step_signal=step_signal)
