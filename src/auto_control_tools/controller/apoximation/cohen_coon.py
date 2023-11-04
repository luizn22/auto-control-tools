from .base import P, PI, PID
from .first_order_table import FirstOrderTableControllerAproximation, FirstOrderTableControllerAproximationItem
from ...model.first_order_model import FirstOrderModel
from ...controller.controller import Controller


class _PController(FirstOrderTableControllerAproximationItem):
    controller_type = P

    @classmethod
    def get_controller(cls, model: FirstOrderModel) -> Controller:
        k = model.K
        tau = model.tau
        theta = model.theta

        return Controller(
            model,
            kp=(1 / k) * (tau / theta) * (1 + theta / (3 * tau))
        )


class _PIController(FirstOrderTableControllerAproximationItem):
    controller_type = PI

    @classmethod
    def get_controller(cls, model: FirstOrderModel) -> Controller:
        k = model.K
        tau = model.tau
        theta = model.theta

        return Controller(
            model,
            kp=(1 / k) * (tau / theta) * (0.9 + theta / (12 * tau)),
            ki=(theta * (30 + 3 * (theta / tau))) / (9 + 20 * (theta / tau))
        )


class _PIDController(FirstOrderTableControllerAproximationItem):
    controller_type = PID

    @classmethod
    def get_controller(cls, model: FirstOrderModel) -> Controller:
        k = model.K
        tau = model.tau
        theta = model.theta

        return Controller(
            model,
            kp=(1 / k) * (tau / theta) * ((16 * tau + 3 * theta) / (12 * tau)),
            ki=(theta * (32 + 6 * (theta / tau))) / (13 + 8 * (theta / tau)),
            kd=(4 * theta) / (11 + 2 * (theta / tau))
        )


class CohenCoonControllerAproximation(FirstOrderTableControllerAproximation):
    """
    Método de Aproximação de Controlador Cohen Coon.

    Proposto por Cohen e Coon, como uma forma de obter os ganhos de controlador para os modelos itentificados
    pelo método clássico de identificação de controladores, descrito em :class:`FirstOrderModel`. O objetivo deste
    método de aproximação é obter parâmetros de ganho PID que façam sintonia do controlador.

    Em espessífico, este método se baseia na curva de reação do sistema a resposta de sinal degrau, que é exatamente o
    que :class:`FirstOrderModel` obtém, com o adicional de que os parâmetros :math:`K`, :math:`\\tau` e :math:`\\theta`
    já foram obtidos, e estão disponíveis para análise.

    Com isso basta aplicar as fórmulas descritas pelo Método de Cohen e Coon para Curva de Reação:


    .. list-table:: Método de Cohen e Coon para Curva de Reação
        :widths: 100 100 100 100
        :header-rows: 1

        * - Controlador
          - :math:`K_P`
          - :math:`T_I`
          - :math:`T_D`
        * - **P**
          - :math:`\\frac{1}{K}\\frac{\\tau}{\\theta}[1+\\frac{\\theta}{3\\tau}]`
          - :math:`\\infty`
          - :math:`0`
        * - **PI**
          - :math:`\\frac{1}{K}\\frac{\\tau}{\\theta}[0.9+\\frac{\\theta}{12\\tau}]`
          - :math:`\\frac{\\theta [30+3 \\frac{\\theta}{\\tau}]}{9+20 \\frac{\\theta}{\\tau}}`
          - :math:`0`
        * - **PID**
          - :math:`\\frac{1}{K}\\frac{\\tau}{\\theta}[\\frac{16\\tau+3\\theta}{12\\tau}]`
          - :math:`\\frac{\\theta [32+6 \\frac{\\theta}{\\tau}]}{13+8 \\frac{\\theta}{\\tau}}`
          - :math:`\\frac{4 \\theta}{11+2 \\frac{\\theta}{\\tau}}`

    Os valores de :math:`T_I` e :math:`T_D` são :math:`1/K_i` e :math:`1/K_d`, respectivamente.

    Notes
    -----
    Em casos de valores zerados de theta, o método não funciona.

    Examples
    --------

    >>> model = act.FirstOrderModel(K=1.95, tau=8.33, theta=1.48)
    >>> controller = act.CohenCoonControllerAproximation.get_controller(model, act.PID)
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

    .. image:: ../image_resources/cc_aprox_plot.png
    """
    _accepted_controllers = [P, PI, PID]
    _controller_table = {
        _PController.controller_type: _PController(),
        _PIController.controller_type: _PIController(),
        _PIDController.controller_type: _PIDController()
    }
