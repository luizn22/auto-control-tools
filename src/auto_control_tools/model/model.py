from copy import copy
from typing import Union, List, Dict, Any, Tuple

import control
import pandas as pd
import sympy as sp

from ..utils.plot import PlotUtils


class Model:
    """
    Classe representativa do :term:`Modelo` matemático de uma planta de sistemas de controle.

    Típicamente o modelo matemático de uma planta no domínio da frequenica pode ser definido por um numerador e um
    denominador em potências de :math:`s`. Ex:

    .. math::

        P(s) = \\frac{ s + 1 }{ s^2 + s + 1 }


    Essa classe funciona guardando um objeto de :term:`Função de Transferência` (:attr:`tf`) da biblioteca de sistemas
    de controle do python (:mod:`control`), representando o modelo
    matemático de uma planta no domínio da frequenica, além de guardar alguns outros metadados sobre a função de
    transferência, que poderão ser utilizados para :term:`Identificação` de modelo posteriormente, por exemplo.


    Além disso o atributo :attr:`view` possibilita a visualização de dados, estatísticas e gráficos referentes a
    função de trasnferência.


    Parameters
    ----------
    tf : ~control.TransferFunction or tuple[list[float], list[float]]]
        :term:`Função de Transferência` que representa o processo da planta,
        pode ser tanto as listas dos coeficientes,
        quanto com um objeto de :term:`Função de Transferência` da bilbioteca de controle
        (:class:`control.TransferFunction`).

    source_data : pd.Series, optional
        Conjunto de dados representando a variação da saida em relação ao tempo.
        Deve um objeto do tipo :class:`pandas.Series` da bilbioteca
        :mod:`pandas`, sendo os valores representativos da saida
        e os valores de index representativos do tempo.

        Essa classe não faz nenhuma análise desses dados por si só, porém eles ficam salvos, e podem ser utilizados
        posteriormente, para plotagem de gráficos, por exemplo.

    Attributes
    ----------
    tf : control.TransferFunction
        :term:`Função de Transferência` que representa o processo da planta (:class:`control.TransferFunction`).

        Utilizada pelas classes de visualização de dados (:class:`ModelView` e :class:`ControllerView`) e pelas
        classes de aproximação de :term:`Modelo` para :term:`Aproximação de Ganhos` (:ref:`gain-aprox`).

    view : ModelView
        Utilizado para visualização dos dados, estatísticas e gráficos referentes ao :term:`Modelo`.

        Verificar a classe :class:`ModelView` para os métodos de visualização de dados.

    source_data : pandas.Series, optional
        Dados originários do :term:`Modelo`.

    Examples
    --------
    Definindo :term:`Modelo` com listas de numerador e denominador:

    >>> num = [1, 2, 3]
    >>> den = [4, 5, 6, 7]
    >>> model = act.Model((num, den))
    >>> model.view.print_tf()

    .. math::

        \\frac{ s^2 + 2s + 3 }{ 4s^3 + 5s^2 + 6s + 7 }

    Definindo :term:`Modelo` com uma :term:`Função de Transferência` da biblioteca
    :mod:`control`:

    >>> num = [1, 2, 3]
    >>> den = [1, 1, 1, 0]
    >>> tf = control.TransferFunction(num, den)
    >>> model = act.Model(tf)
    >>> model.view.print_tf()

    .. math::

        \\frac{ s^2 + 2s + 3 }{ s^3 + s^2 + s }

    """
    def __init__(
            self,
            tf: Union[control.TransferFunction, Tuple[List[float], List[float]]],
            source_data: Union[pd.Series, None] = None,
            step_signal: float = 1,
    ):
        """Instancia :term:`Modelo` matemático de uma planta"""
        if isinstance(tf, control.TransferFunction):
            self.tf = tf
        else:
            self.tf = control.TransferFunction(*tf)

        self.num = self.tf.num[0][0].tolist()
        self.den = self.tf.den[0][0].tolist()

        s = sp.symbols('s')
        num_symbolic = sum([coefficient * s ** i for i, coefficient in enumerate(reversed(self.num))])
        den_symbolic = sum([coefficient * s ** i for i, coefficient in enumerate(reversed(self.den))])
        self.tf_symbolic = num_symbolic / den_symbolic

        self.order = self._identify_model_order()

        self.source_data = source_data
        self.step_signal = step_signal
        self.pade = None

        self.view: ModelView = ModelView(self)

    def _identify_model_order(self) -> int:
        """Identifies the order of self.tf and returns its value"""
        return int(self.tf.den[0][0].size) - 1

    def get_simulation_time(self) -> Union[float, None]:
        """
        Retorna a duração dos dados discretos

        Returns
        -------
        float | None
            Valor da duração dos dados discretos, se existirem, se não, None
        """
        return self.source_data.index[-1] if self.source_data is not None else None


class ModelView:
    """
    Classe utilizada pra visualização de dados de um objeto da classe :class:`Model`.

    Parameters
    ----------

    model : Model
        Um objeto da classe :class:`Model` para o qual deseja-se visualizar os dados.

    Examples
    --------
    >>> num = [1]
    >>> den = [1, 1]
    >>> model = act.Model((num, den))
    >>> model.view.print_tf()

    .. math::
        \\frac{ 1 }{ s + 1 }

    >>> model.view.get_model_step_response_data()
    {'Overshoot': 0,
     'Peak': 0.9989999999999999,
     'PeakTime': 6.907755278982137,
     'RiseTime': 2.1630344812974367,
     'SettlingMax': 1.0,
     'SettlingMin': 0.9,
     'SettlingTime': 3.9771924333533515,
     'SteadyStateValue': 1.0,
     'Undershoot': 0}

    >>> model.view.plot_model_step_response_graph()

    .. image:: ../image_resources/model_view_plot_model_graph.png

    """
    def __init__(self, model: Model):
        """Instancia :class:`ModelView`"""
        self.model = model

    def plot_model_step_response_graph(self, plot_discrete_data: bool = True, settling_time_threshold: float = 0.02,
                                       upscale_model: Union[bool, None] = None,
                                       simulation_time: Union[float, None] = None):
        """
        Apresenta gráfico de resposta degrau do :term:`Modelo`.

        Utiliza :meth:`PlotUtils.plot_tf` para plotar o gráfico da resposta a sinal degrau do :term:`Modelo`, bem como
        as retas de tempo de acomodaçao, sobressinal, e valor de regime e os dados
        discretos (:attr:`Model.source_data`) caso tenham sido informados.

        Parameters
        ----------
        plot_discrete_data : bool, optional
            Plotar ou não os dados discretos juntamente ao :term:`Modelo` calculado.
        settling_time_threshold : float, optional
            Percentual de desvio do valor de regime considerado do cálculo do tempo de acomodação.
        upscale_model : bool, optional
            Fazer ou não *upscale* dos outputs de model pelo valor do setpoint dos dados discretos.
            Facilita comparação visual entre o :term:`Modelo` e os dados discretos.
        simulation_time : float, optional
            Unidade de tempo que a simulação deve durar. Calculado automaticamente se não for fornecido.
        """
        upscale_model = True if plot_discrete_data and upscale_model is None else upscale_model

        PlotUtils.plot_tf(
            tf=self.model.tf,
            discrete_data=self.model.source_data,
            pade=self.model.pade,
            settling_time_threshold=settling_time_threshold,
            scale=self.model.step_signal if upscale_model else 1,
            simulation_time=self.model.get_simulation_time() if simulation_time is None else simulation_time,
        )

    def get_model_step_response_data(self, settling_time_threshold: float = 0.02) -> Dict[str, Any]:
        """
        Retorna dados de resposta a sinal degrau do :term:`Modelo`.

        Utiliza :func:`control.step_info`
        para obtenção dos dados de resposta a sinal degrau do :term:`Sistema` no formato de dicionário (:class:`dict`).

        Parameters
        ----------
        settling_time_threshold : float, optional
            Percentual de desvio do valor de regime considerado do cálculo do tempo de acomodação.

        Notes
        -----
        .. list-table:: Detalhamento dos termos retornados
            :header-rows: 1

            * - Chave
              - Nome
              - Descrição
            * - RiseTime
              - Tempo de Subida
              - Tempo de 10% a 90% do valor de regime.
            * - SettlingTime
              - Tempo de Acomodação
              - Tempo para até entrada em 2% de erro do valor de regime.
            * - SettlingMin
              - Valor Mínimo de Acom.
              - Valor mínimo após o Tempo de Subida.
            * - SettlingMax
              - Valor Máximo de Acom.
              - Valor máximo após o Tempo de Subida.
            * - Overshoot
              - Sobressinal
              - Percentual do Pico em relação ao valor de regime.
            * - Undershoot
              - Subsinal
              - Percentual de Subsinal em relação ao valor de regime.
            * - Peak
              - Pico
              - Valor absoluto do pico.
            * - PeakTime
              - Tempo do Pico
              - Tempo do pico.
            * - SteadyStateValue
              - Valor de regime
              - Valor de regime do :term:`Sistema`.

        """

        tf = copy(self.model.tf)
        if self.model.pade is not None:
            tf = tf * self.model.pade  # type: ignore

        return dict(control.step_info(tf, T=self.model.get_simulation_time(),
                                      SettlingTimeThreshold=settling_time_threshold))

    def print_model_step_response_data(self, settling_time_threshold: float = 0.02):
        """
        *Pretty Print* dos dados de resposta a sinal de grau do :term:`sistema`.

        Utiliza :meth:`PlotUtils.pprint_dict` para fazer o print dos dados.

        Parameters
        ----------
        settling_time_threshold : float, optional
            Percentual de desvio do valor de regime considerado do cálculo do tempo de acomodação.
        """

        PlotUtils.pprint_dict(self.get_model_step_response_data(settling_time_threshold=settling_time_threshold))

    def print_tf(self):
        """
        Imprime a :term:`Função de Transferência` na tela.

        Utiliza o método :meth:`PlotUtils.print_tf`
        Caso esteja em um ambiente `jupyter <https://jupyter.org/>`_, a função :func:`~IPython.display.display`
        é chamada para que a função
        de transferência seja mostrada com formatação matemática.
        """
        PlotUtils.print_tf(self.model.tf_symbolic)
