from abc import abstractmethod

from ...model.model import Model
from ...controller.controller import Controller


class BaseControllerAproximation:
    @abstractmethod
    def get_controller(self, model: Model, *args, **kwargs) -> Controller:
        pass
