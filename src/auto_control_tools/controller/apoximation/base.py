from abc import abstractmethod

from ...model.model import Model
from ...controller.controller import Controller

P = 'P'
I = 'I'  # noqa
D = 'D'
PI = P+I
PD = P+D
PID = P+I+D


class BaseControllerAproximation:
    accepted_controllers = [P, I, D, PI, PD, PID]

    @classmethod
    @abstractmethod
    def get_controller(cls, model: Model, controller_type: str) -> Controller:
        raise NotImplementedError('get_controller must be implemented in a subclass')

    @classmethod
    def _parse_controller_option(cls, controller_type: str):
        if controller_type not in cls.accepted_controllers:
            raise ValueError(f'Invalid Controller option, must be one of {cls.accepted_controllers}')
