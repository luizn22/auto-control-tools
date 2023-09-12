from typing import Union, List, Dict, Any

import control
import pandas as pd
import sympy as sp

from ..utils.data_input import DataInputUtils


class Model:
    def __init__(
            self,
            tf: Union[control.TransferFunction, List[List[float]]],
            order: Union[int, None] = None,
            source_data: Union[pd.DataFrame, None] = None,
            num: Union[List[float], None] = None,
            den: Union[List[float], None] = None,
    ):
        if isinstance(tf, control.TransferFunction):
            self.tf = tf
            self.num = self.tf.num[0][0].tolist()
            self.den = self.tf.den[0][0].tolist()
        else:
            self.tf = control.TransferFunction(*tf)
            self.num = tf[0]
            self.den = tf[1]

        if num is not None:
            self.num = num

        if den is not None:
            self.den = den

        s = sp.symbols('s')
        num_symbolic = sum([coefficient * s ** i for i, coefficient in enumerate(reversed(self.num))])
        den_symbolic = sum([coefficient * s ** i for i, coefficient in enumerate(reversed(self.den))])
        self.tf_symbolic = num_symbolic / den_symbolic

        self.order = order if order is not None else self.identify_model_order()
        self.source_data = source_data if source_data is not None else pd.DataFrame(
            columns=DataInputUtils.standard_fields)

        self.view = ModelView(self)

    def identify_model_order(self) -> int:
        return int(self.tf.den[0][0].size) - 1


class ModelView:
    def __init__(self, model: Model):
        self.model = model

    def plot_model_graph(self, *args, **kwargs):
        pass

    def get_model_data(self) -> Dict[str, Any]:
        pass

    def print_model_data(self, *args, **kwargs):
        pass
