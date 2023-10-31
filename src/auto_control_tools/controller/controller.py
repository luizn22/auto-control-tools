import copy
from typing import Dict, Any, Union

import control
import sympy as sp

from ..model.model import Model
from ..utils.plot import PlotUtils


class Controller:
    """
    Classe base para controle PID

    Esta é uma classe representativa do modelo matemático de uma planta de sistemas de controle em malha fechada com
    controlador PID (Proporcional-Itegrador-Derivativo). Esse tipo de controlador é muito difundido na indústria,
    possuindo inúmeras aplicações.

    Para um processo :math:`P(s)` o fechamento de malha simples com controlador PID denota-se pelas seguintes equações:

    .. math::

        C(s) = K_d s + K_p + \\frac{K_i}{s}

    .. math::

        F(s) = \\frac{P(s)C(s)}{1 + P(s)C(s)}


    Essa classe se destina a armazenar os valores de ganhos de controlador, bem como realizar o fechamento da manha
    de controle utilizando os parâmetros PID (:paramref:`kp`, :paramref:`ki`, :paramref:`kd`) e o objeto de modelo
    (:paramref:`model`). Para isso, atua encima do objeto de função de transferência da biblioteca de controle
    :attr:`Model.tf` (`TransferFunction
    <https://python-control.readthedocs.io/en/latest/generated/control.TransferFunction.html>`_) presente no
    :attr:`Model` fornecido.


    Além disso o atributo :attr:`view` possibilita a visualização de dados, estatísticas e gráficos referentes ao
    resultado do controle (função de transfferência resultante do fechamento da malha com controlador).

    Parameters
    ----------
    model : Model
        Modelo matemático de uma planta a ser controlada. Salvo como atributo :attr:`model`.
    ki : float, optional
        Valor referente ao ganho Integrador, :math:`K_i`, do controlador, caso não informado o ganho é considerado
        zero. implicando que este método não faz parte deste controlador. Salvo como atributo :attr:`ki`.
    kp : float, optional
        Valor referente ao ganho Proporcional, :math:`K_p` do controlador, caso não informado o ganho é considerado
        zero. implicando que este método não faz parte deste controlador. Salvo como atributo :attr:`kp`.
    kd : float, optional
        Valor referente ao ganho Derivativo, :math:`K_d` do controlador, caso não informado o ganho é considerado zero.
        implicando que este método não faz parte deste controlador. Salvo como atributo :attr:`kd`.

    Attributes
    ----------
    model : Model
        Modelo matemático de uma planta a ser controlada. Utilizado por métodos de outras classes para obter informações
        sobre o modelo.
    ki : float, optional
        Valor referente ao ganho Integrador, :math:`K_i`, do controlador, utilizado no cálculo de :attr:`tf` e
        :attr:`tf_symbolic`.
    kp : float, optional
        Valor referente ao ganho Proporcional, :math:`K_p` do controlador, utilizado no cálculo de :attr:`tf` e
        :attr:`tf_symbolic`.
    kd : float, optional
        Valor referente ao ganho Derivativo, :math:`K_d` do controlador, utilizado no cálculo de :attr:`tf` e
        :attr:`tf_symbolic`.
    tf : `TransferFunction <https://python-control.readthedocs.io/en/latest/generated/control.TransferFunction.html>`_
        Função de transferência que representa o processo da planta em malha fechada com controlador (`TransferFunction
        <https://python-control.readthedocs.io/en/latest/generated/control.TransferFunction.html>`_).

        Utilizada pelas classe de visualização de dados :class:`ControllerView`.

        Calculada a partir do fechamento da malha do controlador junto ao processo.
    tf_symbolic : `Expr <https://docs.sympy.org/latest/modules/core.html#module-sympy.core.expr>`_
        Guarda representação simbolica
        (`Expr <https://docs.sympy.org/latest/modules/core.html#module-sympy.core.expr>`_) de :attr:`tf`.

        Atributos da class control.TransferFunction funcionam apenas com as listas dos coeficientes de s de seus
        numeradores e denominadores, expressões mais complexas, como exponenciais, não podem ser representadas de
        forma direta (Ex: :attr:`FirstOrderModel.pade`), para tal, :attr:`tf_symbolic` guarda a representação do modelo
        matemático do qual :attr:`tf` tenta se aproximar.
    view : ModelView
        Utilizado para visualização dos dados, estatísticas e gráficos referentes ao Controlador.

        Verificar a classe :class:`ControllerView` para os métodos de visualização de dados.

    Examples
    --------
    Usando Controller:

    >>> model = FirstOrderModel(K=1.95, tau=8.33, theta=1)
    >>> controller = Controller(model, ki=3.393417131440939, kp=3.9766689766689765, kd=0.5213405222539381)
    >>> controller.view.print_tf()

    .. math::

        \\frac{1.95 \\left(0.52s + 3.98 + \\frac{3.39}{s}\\right) \\exp(-1.48s)}{\\left(1 + \\frac{1.95
        \\left(0.52s + 3.98 + \\frac{3.39}{s}\\right) \\exp(-1.48s)}{8.33s + 1.0}\\right) \\left(8.33s + 1.0\\right)}
    """

    def __init__(
            self,
            model: Model,
            ki: float = 0,
            kp: float = 0,
            kd: float = 0
    ):
        """Instancia Controlador"""
        self.model = model
        self.ki = ki
        self.kp = kp
        self.kd = kd

        s = sp.symbols('s')
        Kp, Ki, Kd = sp.symbols('Kp Ki Kd')
        pid_symbolic = kp + (ki / s) + (kd * s)

        self.tf_symbolic = model.tf_symbolic * pid_symbolic / (1 + model.tf_symbolic * pid_symbolic)
        self.tf_symbolic = sp.trigsimp(self.tf_symbolic, tolerance=1e-4, rational=True)

        s = control.TransferFunction.s
        pid = kd * s + kp + ki / s
        # pid = control.TransferFunction([kd, kp, ki], [1, 0])
        self.tf = (model.tf * pid) / (1 + model.tf * pid)
        if (self.tf.num[0][0][-1] == 0) and (self.tf.den[0][0][-1] == 0):
            self.tf = control.TransferFunction(self.tf.num[0][0][:-1], self.tf.den[0][0][:-1])
        # self.tf = model.tf.feedback(pid)

        self.view = ControllerView(self)


class ControllerView:
    """
    Classe utilizada pra visualização de dados de um objeto da classe :class:`Controller`.

    Parameters
    ----------

    controller : Controller
        Um objeto da classe :class:`Controller` para o qual deseja-se visualizar os dados.

    Examples
    --------
    Usando Controller:

    >>> model = FirstOrderModel(K=1.95, tau=8.33, theta=1)
    >>> controller = Controller(model, ki=3.393417131440939, kp=3.9766689766689765, kd=0.5213405222539381)
    >>> controller.view.print_tf()

    .. math::

        \\frac{1.95 \\left(0.52s + 3.98 + \\frac{3.39}{s}\\right) \\exp(-1.48s)}{\\left(1 + \\frac{1.95
        \\left(0.52s + 3.98 + \\frac{3.39}{s}\\right) \\exp(-1.48s)}{8.33s + 1.0}\\right) \\left(8.33s + 1.0\\right)}

    >>> controller.view.print_controller_step_response_data()
    {'Kd': 0.5213405222539381,
     'Ki': 3.393417131440939,
     'Kp': 3.9766689766689765,
     'Overshoot': 20.40652189650112,
     'Peak': 1.2040652189650112,
     'PeakTime': 4.478154203547828,
     'RiseTime': 1.3154577972921744,
     'SettlingMax': 1.2040652189650112,
     'SettlingMin': 0.9001418058725997,
     'SettlingTime': 9.823950784033048,
     'SteadyStateValue': 1.0,
     'Undershoot': 10.87681610040138}

    >>> controller.view.plot_controller_step_response_graph()

    .. image:: ../image_resources/ctrl_view_plot_ctrl_graph.png

    """

    def __init__(self, controller: Controller):
        self.controller = controller

    def plot_controller_step_response_graph(self, plot_model: bool = True, use_pade: bool = True,
                                            settling_time_threshold: float = 0.02,
                                            simulation_time: Union[float, None] = None):
        """
        Apresenta gráfico de resposta degrau do modelo.

        Utiliza :meth:`PlotUtils.plot_tf` para plotar o gráfico da resposta a sinal degrau do controlador, bem como
        as retas de tempo de acomodaçao, sobressinal, e valor de regime. Também faz o plot da função de transferencia
        do modelo por padrão.

        Parameters
        ----------
        plot_model : bool, optional
            Plotar ou não a função de transferência do modelo juntamente a do controlador.
        use_pade : bool, optional
            Plotar ou não consideranto o atraso de tempo (são introduzidos muitos termos a função de transferência com
            a aproximação de pade, isso pode causar problemas ao plotar os dados em alguns casos).
        settling_time_threshold : float, optional
            Percentual de desvio do valor de regime considerado do cálculo do tempo de acomodação.
        simulation_time : float, optional
            Unidade de tempo que a simulação deve durar. Calculado automaticamente se não for fornecido.
        """

        PlotUtils.plot_tf(
            tf={'Controller': self.controller.tf, 'Model': self.controller.model.tf
                } if plot_model else self.controller.tf,
            settling_time_threshold=settling_time_threshold,
            pade=self.controller.model.pade if use_pade else None,
            simulation_time=self.controller.model.get_simulation_time() if simulation_time is None else simulation_time,
        )

    def get_controller_step_response_data(self, settling_time_threshold: float = 0.02) -> Dict[str, Any]:
        """
        Retorna dados de resposta a sinal degrau do controlador.

        Utiliza `control.step_info()
        <https://python-control.readthedocs.io/en/latest/generated/control.step_info.html>`_
        para obtenção dos dados de resposta a sinal degrau do sistema no formato de dicionário (:class:`dict`).

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
            * - kp
              - Ganho Proporcional
              - Valor referente ao ganho Proporcional do controlador.
            * - ki
              - Ganho Integrador
              - Valor referente ao ganho Integrador do controlador.
            * - kd
              - Ganho Derivativo
              - Valor referente ao ganho Derivativo do controlador.
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
              - Valor de regime do sistema.

        """
        data = {
            'Ki': self.controller.ki,
            'Kp': self.controller.kp,
            'Kd': self.controller.kd
        }

        tf = copy.copy(self.controller.tf)
        if self.controller.model.pade is not None:
            tf = tf * self.controller.model.pade

        data.update(dict(control.step_info(
            tf,
            T=self.controller.model.get_simulation_time(),
            SettlingTimeThreshold=settling_time_threshold
        )))

        return data

    def print_controller_step_response_data(self, settling_time_threshold: float = 0.02):
        """
        *Pretty Print* dos dados de resposta a sinal de grau do sistema.

        Utiliza :meth:`PlotUtils.pprint_dict` para fazer o print dos dados.

        Parameters
        ----------
        settling_time_threshold : float, optional
            Percentual de desvio do valor de regime considerado do cálculo do tempo de acomodação.
        """

        PlotUtils.pprint_dict(self.get_controller_step_response_data())

    def print_tf(self):
        """
        Imprime a função de transferência na tela.

        Utiliza o método :meth:`PlotUtils.print_tf`
        Caso esteja em um ambiente `jupyter <https://jupyter.org/>`_, a função display() é chamada para que a função
        de transferência seja mostrada com formatação matemática.
        """
        PlotUtils.print_tf(self.controller.tf_symbolic)
