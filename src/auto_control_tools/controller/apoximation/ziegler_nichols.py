from .base import P, PI, PID
from .first_order_table import FirstOrderTableControllerAproximation, FirstOrderTableControllerAproximationItem
from ...model.first_order_model import FirstOrderModel
from ...controller.controller import Controller


class _PController(FirstOrderTableControllerAproximationItem):
    controller_type = P

    @classmethod
    def get_controller(cls, model: FirstOrderModel) -> Controller:
        return Controller(
            model,
            kp=(1/model.K)*(model.tau / model.theta)
        )


class _PIController(FirstOrderTableControllerAproximationItem):
    controller_type = PI

    @classmethod
    def get_controller(cls, model: FirstOrderModel) -> Controller:
        return Controller(
            model,
            kp=0.9*(1/model.K)*(model.tau / model.theta),
            ki=1/(model.theta / 0.3)
        )


class _PIDController(FirstOrderTableControllerAproximationItem):
    controller_type = PID

    @classmethod
    def get_controller(cls, model: FirstOrderModel) -> Controller:
        return Controller(
            model,
            kp=1.2*(1/model.K)*(model.tau / model.theta),
            ki=1/(2 * model.theta),
            kd=1/(0.5 * model.theta)
        )


class ZieglerNicholsControllerAproximation(FirstOrderTableControllerAproximation):
    """
    Método de Aproximação de Controlador Ziegler Nichols.

    Proposto por Ziegler e Nichols, como uma forma de obter os ganhos de controlador para os modelos itentificados
    pelo seu método de indentificação :class:`ZieglerNicholsModelIdentification`, o objetivo deste método é obter
    parâmetros de ganho PID que façam sintonia do controlador.

    Em espessífico, este método se baseia na curva de reação do sistema a resposta de sinal degrau, que é exatamente o
    que :class:`FirstOrderModel` obtém, com o adicional de que os parâmetros :math:`K`, :math:`\\tau` e :math:`\\theta`
    já foram obtidos, e estão disponíveis para análise.

    Com isso basta aplicar as fórmulas descritas pelo Método de Ziegler e Nichols para Curva de Reação:


    .. list-table:: Método de Ziegler e Nichols para Curva de Reação
        :widths: 100 100 100 100
        :header-rows: 1

        * - Controlador
          - :math:`K_P`
          - :math:`T_I`
          - :math:`T_D`
        * - **P**
          - :math:`\\frac{1}{K}\\frac{\\tau}{\\theta}`
          - :math:`\\infty`
          - :math:`0`
        * - **PI**
          - :math:`0.9\\frac{1}{K}\\frac{\\tau}{\\theta}`
          - :math:`\\frac{\\theta}{0.3}`
          - :math:`0`
        * - **PID**
          - :math:`1.2\\frac{1}{K}\\frac{\\tau}{\\theta}`
          - :math:`2\\theta`
          - :math:`0.5\\theta`

    Os valores de :math:`T_I` e :math:`T_D` são :math:`1/K_i` e :math:`1/K_d`, respectivamente.

    Notes
    -----
    Em casos de valores zerados de theta, o método não funciona.

    Examples
    --------

    >>> model = act.FirstOrderModel(K=1.95, tau=8.33, theta=1.48)
    >>> controller = act.ZieglerNicholsControllerAproximation.get_controller(model, PID)
    >>> controller.view.print_tf()

    .. math::

        \\frac{1.95 \\left(1.35s + 3.46 + \\frac{0.34}{s}\\right) \\exp(-1.48s)}{\\left(1 + \\frac{1.95
        \\left(1.35s + 3.46 + \\frac{0.34}{s}\\right) \\exp(-1.48s)}{8.33s + 1.0}\\right) \\left(8.33s + 1.0\\right)}


    >>> controller.view.print_controller_step_response_data()
    {'Kd': 1.3513513513513513,
     'Ki': 0.33783783783783783,
     'Kp': 3.4636174636174637,
     'Overshoot': 0,
     'Peak': 0.9862346245156599,
     'PeakTime': 11.353863720564135,
     'RiseTime': 3.7565870087792446,
     'SettlingMax': 1.0,
     'SettlingMin': 0.9002644728593896,
     'SettlingTime': 9.503604447583314,
     'SteadyStateValue': 1.0,
     'Undershoot': 24.031943999408444}

    >>> controller.view.plot_controller_step_response_graph()

    .. image:: ../image_resources/zn_aprox_plot.png
    """
    _accepted_controllers = [P, PI, PID]
    _controller_table = {
        _PController.controller_type: _PController(),
        _PIController.controller_type: _PIController(),
        _PIDController.controller_type: _PIDController()
    }
