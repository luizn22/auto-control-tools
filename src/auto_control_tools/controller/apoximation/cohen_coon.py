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
        teta = model.teta

        return Controller(
            model,
            kp=(1 / k) * (tau / teta) * (1 + teta / (3 * tau))
        )


class _PIController(FirstOrderTableControllerAproximationItem):
    _controller_type = PI

    @classmethod
    def get_controller(cls, model: FirstOrderModel) -> Controller:
        k = model.K
        tau = model.tau
        teta = model.teta

        return Controller(
            model,
            kp=(1 / k) * (tau / teta) * (0.9 + teta / (12 * tau)),
            ki=(teta * (30 + 3 * (teta / tau))) / (9 + 20 * (teta / tau))
        )


class _PIDController(FirstOrderTableControllerAproximationItem):
    _controller_type = PID

    @classmethod
    def get_controller(cls, model: FirstOrderModel) -> Controller:
        k = model.K
        tau = model.tau
        teta = model.teta

        return Controller(
            model,
            kp=(1 / k) * (tau / teta) * ((16 * tau + 3 * teta) / (12 * tau)),
            ki=(teta * (32 + 6 * (teta / tau))) / (13 + 8 * (teta / tau)),
            kd=(4*teta) / (11 + 2 * (teta / tau))
        )


class CohenCoonControllerAproximation(FirstOrderTableControllerAproximation):
    _accepted_controllers = [P, PI, PID]
    _controller_table = {
        i._controller_type: i for i in [_PController(), _PIController(), _PIDController()]
    }
