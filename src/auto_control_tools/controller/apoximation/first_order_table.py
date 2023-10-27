from abc import abstractmethod
from typing import Dict

from .base import BaseControllerAproximation
from ...model.first_order_model import FirstOrderModel
from ...controller.controller import Controller


class FirstOrderTableControllerAproximationItem:
    _controller_type: str = ''  # replace with controller type

    @classmethod
    @abstractmethod
    def get_controller(cls, model: FirstOrderModel) -> Controller:
        pass


class FirstOrderTableControllerAproximation(BaseControllerAproximation):
    _controller_table: Dict[str, FirstOrderTableControllerAproximationItem] = {}

    @classmethod
    def get_controller(
            cls,
            model: FirstOrderModel,  # type: ignore[override]
            controller_type: str
    ) -> Controller:
        cls._parse_controller_option(controller_type)
        return cls._controller_table[controller_type].get_controller(model)
