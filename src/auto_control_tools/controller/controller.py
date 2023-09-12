from typing import Union, Dict, Any

import control
import sympy as sp
import matplotlib.pyplot as plt

from ..model.model import Model
from ..utils.data import DataUtils


class Controller:
    def __init__(
            self,
            model: Model,
            ki: float = 0,
            kp: float = 0,
            kd: float = 0
    ):
        self.model = model
        self.ki = ki
        self.kp = kp
        self.kd = kd

        s = sp.symbols('s')
        Kp, Ki, Kd = sp.symbols('Kp Ki Kd')
        pid_symbolic = Kp + (Ki / s) + (Kd * s)

        self.tf_symbolic = model.tf_symbolic * pid_symbolic / (1 + model.tf_symbolic * pid_symbolic)

        pid = control.TransferFunction([kd, kp, ki], [1])
        self.tf = model.tf.feedback(pid)

        self.view = ControllerView(self)


class ControllerView:
    def __init__(self, controller):
        self.controller = controller

    def plot_controller_graph(self, *args, **kwargs):
        time, response = control.step_response(self.controller.tf)
        plt.plot(time, response)
        plt.xlabel('Time')
        plt.ylabel('Response')
        plt.title('Step Response of PID Controlled System')
        plt.show()

    def get_controller_data(self) -> Dict[str, Any]:
        return control.step_info(self.controller.tf)

    def print_controller_data(self, *args, **kwargs):
        DataUtils.pprint_dict(self.get_controller_data())

