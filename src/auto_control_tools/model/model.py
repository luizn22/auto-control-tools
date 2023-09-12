from typing import Union, List, Dict, Any

import control
import pandas as pd

from ..utils.data_input import DataInputUtils


class Model:
    def __init__(
            self,
            tf: Union[control.TransferFunction, List[List[float]]],
            order: Union[int, None] = None,
            source_data: Union[pd.DataFrame, None] = None
    ):
        self.system = self.get_control_system(tf)
        self.order = order if order is not None else self.identify_model_order()
        self.source_data = source_data if source_data is not None else pd.DataFrame(
            columns=DataInputUtils.standard_fields)

        self.view = ModelView(self)

    @staticmethod
    def get_control_system(tf: Union[control.TransferFunction, List[List[float]]]):
        if isinstance(tf, control.TransferFunction):
            return tf
        else:
            return control.TransferFunction(*tf)

    def identify_model_order(self) -> int:
        return int(self.system.den[0][0].size) - 1


class ModelView:
    def __init__(self, model: Model):
        self.model = model

    def plot_model_graph(self, *args, **kwargs):
        pass

    def get_model_data(self) -> Dict[str, Any]:
        pass

    def print_model_data(self, *args, **kwargs):
        pass
