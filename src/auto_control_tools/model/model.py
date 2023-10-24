from typing import Union, List, Dict, Any

import control
import pandas as pd
import sympy as sp

from ..utils.data import DataUtils
from ..utils.plot import PlotUtils


class Model:
    """
    Classe armazenando a função de transferência que representa um modelo, métodos de visualização
    bem como diversos outros dados úteis.
    """
    def __init__(
            self,
            tf: Union[control.TransferFunction, List[List[float]]],
            order: Union[int, None] = None,
            source_data: Union[pd.Series, None] = None,
            num: Union[List[float], None] = None,
            den: Union[List[float], None] = None,
    ):
        if isinstance(tf, control.TransferFunction):
            self.tf = tf
        else:
            self.tf = control.TransferFunction(*tf)

        self.num = self.tf.num[0][0].tolist()
        self.den = self.tf.den[0][0].tolist()

        if num is not None:
            self.num = num

        if den is not None:
            self.den = den

        s = sp.symbols('s')
        num_symbolic = sum([coefficient * s ** i for i, coefficient in enumerate(reversed(self.num))])
        den_symbolic = sum([coefficient * s ** i for i, coefficient in enumerate(reversed(self.den))])
        self.tf_symbolic = num_symbolic / den_symbolic

        self.order = order if order is not None else self.identify_model_order()
        self.source_data = source_data if source_data is not None else pd.Series().astype(float)

        self.pade = None

        self.view = ModelView(self)

    def identify_model_order(self) -> int:
        return int(self.tf.den[0][0].size) - 1


class ModelView:
    def __init__(self, model: Model):
        self.model = model

    def plot_model_graph(self):
        if self.model.source_data.empty:
            PlotUtils.plot_tf(self.model.tf, pade=self.model.pade)
        else:
            PlotUtils.plot_tf(self.model.tf, self.model.source_data, pade=self.model.pade)

    def get_model_data(self) -> Dict[str, Any]:
        return dict(control.step_info(self.model.tf))

    def print_model_data(self, *args, **kwargs):
        DataUtils.pprint_dict(self.get_model_data())

    def print_tf(self):
        PlotUtils.print_tf(self.model.tf_symbolic)
