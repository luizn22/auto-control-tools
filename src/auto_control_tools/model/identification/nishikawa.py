from typing import Union

import numpy as np

from ..first_order_model import FirstOrderModel
from .base import BaseModelIdentification
from ...utils.data import DataUtils
from ...utils.data_input import DataInputUtils


class NishikawaModelIdentification(BaseModelIdentification):
    """
    Método de Identicação Nishikawa.

    Este é um método clássico da literatura para identificação de modelos dinâmicos de processos industriais.
    Desenvolvido por Nishikawa, em 1984 :footcite:p:`CoelhoChapter4`.

    O processo de identificação de modelo pelo método de Nishikawa envolve a análise da resposta ao
    degrau de um sistema, com base em uma função de transferência de primeira ordem com tempo morto (FOPDT).
    Os parâmetros chave a serem determinados são o ganho estático (:math:`K`), a constante de tempo (:math:`\\tau`),
    e o atraso de transporte (:math:`\\theta`) :footcite:p:`CoelhoChapter4`.


    Os parâmetros são calculados conforme indicado pela figura a seguir:

    .. figure:: ../image_resources/ni_ident_meth.png
        :align: center

        Método de Nishikawa para a modelagem de processos de primeira ordem :footcite:p:`CoelhoChapter4`.

    No metodo Nishikawa os parâmetros são obtidos pelo cálculo das áreas :math:`A_0` e :math:`A_1` e do tempo t_0,
    ou seja,

    .. math::

        A_0 = \\int_{0}^{\\infty} \\{ \\Delta y(\\infty) - \\Delta y(t) \\} dt

    .. math::

        A_1 = \\int_{0}^{t_0} \\Delta y(t) dt \\;\\; ; \\;\\; t_0 = \\frac{A_0}{\\Delta y(\\infty)}

    Conforme ilustrado na figura. De posse dessas áreas, os parâmetros do modelo são dados por

    .. math::

        \\tau = \\frac{A_1}{0.368\\Delta y(\\infty)}

    .. math::

        \\theta = t_0 - \\tau

    E por fim o valor de :math:`K` pode ser obtido a través da equação,

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
    >>> model = act.NishikawaModelIdentification.get_model(file_path)
    >>> model.view.print_tf()

    .. math::
        \\frac{ 168.26 }{ 7.71s + 1 }e^{-2.22s}

    >>> model.view.print_model_step_response_data()
    {'Overshoot': 0,
     'Peak': 168.03545714435953,
     'PeakTime': 52.5444563372188,
     'RiseTime': 16.672760183925195,
     'SettlingMax': 168.2608695652174,
     'SettlingMin': 151.45062634600725,
     'SettlingTime': 31.998226615614012,
     'SteadyStateValue': 168.2608695652174,
     'Undershoot': 0.6269575101889905}

    >>> model.view.plot_model_step_response_graph()

    .. image:: ../image_resources/ni_ident_plot.png
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
        * Calcula :math:`A_0` como a área de um retângulo de lados :math:`\\Delta y(\\infty)` e o momento de entrada em
          valor de regime menos a integral da curva de :math:`t = 0` até momento de entrada em valor de regime.
        * Calcula :math:`t_0` como :math:`A_0` dividido pelo valor de regime.
        * Calcula :math:`A_1` como a integral da curva de :math:`t = 0` até :math:`t = t_0`.
        * Obtém :math:`K` dividindo o valor de regime, pelo valor do sinal degrau.

        * Com os valores de :math:`A_0`, :math:`A_1` e :math:`t_0` em mãos, calcula o valor de :attr:`tau` e
          :attr:`theta`, sendo :math:`\\tau` igual a :math:`A_0` dividido por 0.368 vezes o valor de regime e
          :math:`\\theta = t_0 - \\tau`.
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

        Notes
        -----
        As integrais calculadas utilizaram a função `numpy.trapz
        <https://numpy.org/doc/stable/reference/generated/numpy.trapz.html>`_
        que utiliza a regra dos trapezoides compostos para realizar a integração de dados discretos.

        Returns
        -------
        Objeto :class:`FirstOrderModel` referente ao modelo gerado pelo método.
        """
        df = DataInputUtils.get_model_data_default(path, sample_time, step_signal)
        tf_data, step_signal = DataUtils.setup_data_default(df, sample_time, step_signal)

        idx_vreg, vreg = DataUtils.get_vreg(tf_data, settling_time_threshold=settling_time_threshold)

        A0 = vreg*idx_vreg - np.trapz(tf_data[:idx_vreg], tf_data[:idx_vreg].index)  # type: ignore

        t0 = A0/vreg

        A1 = np.trapz(tf_data.loc[tf_data.index <= t0], tf_data.loc[tf_data.index <= t0].index)

        K = vreg / step_signal

        tau = A1/(0.368*vreg)
        theta = t0 - tau

        if ignore_delay_threshold is not None and theta < ignore_delay_threshold:
            theta = 0

        return FirstOrderModel(K, tau, theta, source_data=tf_data, step_signal=step_signal)
