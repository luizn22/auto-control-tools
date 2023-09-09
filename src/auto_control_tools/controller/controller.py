from typing import Union, Dict, Any

from ..model.model import Model


class Controller:
    def __init__(
            self,
            model: Model,
            ki: float = 0,
            kp: float = 0,
            kd: float = 0,
            meta: Union[Dict[str, Any], None] = None
    ):
        self.model = model
        self.view = ControllerView(self)
        self.meta = meta if meta else {}
        self.ki = ki
        self.kp = kp
        self.kd = kd


class ControllerView:
    def __init__(self, controller):
        self.controller = controller

    def plot_controller_graph(self, *args, **kwargs):
        pass

    def get_controller_data(self) -> Dict[str, Any]:
        pass

    def print_controller_data(self, *args, **kwargs):
        pass
