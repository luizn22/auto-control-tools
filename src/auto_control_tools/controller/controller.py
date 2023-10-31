from typing import Dict, Any

import control
import sympy as sp

from ..model.model import Model
from ..utils.plot import PlotUtils


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
        pid_symbolic = kp + (ki / s) + (kd * s)

        self.tf_symbolic = model.tf_symbolic * pid_symbolic / (1 + model.tf_symbolic * pid_symbolic)
        self.tf_symbolic = sp.trigsimp(self.tf_symbolic, tolerance=1e-4, rational=True)

        s = control.TransferFunction.s
        pid = kd * s + kp + ki / s
        # pid = control.TransferFunction([kd, kp, ki], [1, 0])
        self.tf = (model.tf * pid) / (1 + model.tf * pid)
        if (self.tf.num[0][0][-1] == 0) and (self.tf.den[0][0][-1] == 0):
            self.tf = control.TransferFunction(self.tf.num[0][0][:-1], self.tf.den[0][0][:-1])
        # self.tf = model.tf.feedback(pid)

        self.view = ControllerView(self)


class ControllerView:
    def __init__(self, controller: Controller):
        self.controller = controller

    def plot_controller_step_response_graph(self, plot_model=True):
        if plot_model:
            PlotUtils.plot_tf(
                {'Controller': self.controller.tf, 'Model': self.controller.model.tf}, pade=self.controller.model.pade)
        else:
            PlotUtils.plot_tf(self.controller.tf, pade=self.controller.model.pade)

    def get_controller_step_response_data(self) -> Dict[str, Any]:
        data = {
            'Ki': self.controller.ki,
            'Kp': self.controller.kp,
            'Kd': self.controller.kd
        }

        data.update(dict(control.step_info(self.controller.tf)))

        return data

    def print_controller_step_response_data(self, *args, **kwargs):
        PlotUtils.pprint_dict(self.get_controller_step_response_data())

    def print_tf(self):
        PlotUtils.print_tf(self.controller.tf_symbolic)
