from .base import P, PI, PID
from .first_order_table import FirstOrderTableControllerAproximation, FirstOrderTableControllerAproximationItem
from ...model.first_order_model import FirstOrderModel
from ...controller.controller import Controller


class _PController(FirstOrderTableControllerAproximationItem):
    _controller_type = P

    @classmethod
    def get_controller(cls, model: FirstOrderModel) -> Controller:
        return Controller(
            model,
            kp=(1/model.K)*(model.tau/model.teta)
        )


class _PIController(FirstOrderTableControllerAproximationItem):
    _controller_type = PI

    @classmethod
    def get_controller(cls, model: FirstOrderModel) -> Controller:
        return Controller(
            model,
            kp=0.9*(1/model.K)*(model.tau/model.teta),
            ki=1/(model.teta/0.3)
        )


class _PIDController(FirstOrderTableControllerAproximationItem):
    _controller_type = PID

    @classmethod
    def get_controller(cls, model: FirstOrderModel) -> Controller:
        return Controller(
            model,
            kp=1.2*(1/model.K)*(model.tau/model.teta),
            ki=1/(2*model.teta),
            kd=1/(0.5*model.teta)
        )


class ZieglerNicholsControllerAproximation(FirstOrderTableControllerAproximation):
    _accepted_controllers = [P, PI, PID]
    _controller_table = {
        i._controller_type: i for i in [_PController(), _PIController(), _PIDController()]
    }
