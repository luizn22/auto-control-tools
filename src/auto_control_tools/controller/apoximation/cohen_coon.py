from .base import P, PI, PID
from .first_order_table import FirstOrderTableControllerAproximation, FirstOrderTableControllerAproximationItem
from ...model.first_order_model import FirstOrderModel
from ...controller.controller import Controller


class _PController(FirstOrderTableControllerAproximationItem):
    _controller_type = P

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
    _controller_type = PI

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
    _controller_type = PID

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
    _accepted_controllers = [P, PI, PID]
    _controller_table = {
        i._controller_type: i for i in [_PController(), _PIController(), _PIDController()]
    }
