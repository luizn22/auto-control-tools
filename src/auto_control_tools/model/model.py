from typing import Union, List, Dict, Any

import control
import pandas as pd

from ..utils.data_input import DataInputUtils


class Model:
    def __init__(
            self,
            tf: Union[control.TransferFunction, List[List[float]]],
            meta: Union[Dict[str, Any], None] = None
    ):
        self.view = ModelView(self)
        if isinstance(tf, control.TransferFunction):
            self.system = tf
        else:
            self.tf = control.TransferFunction(*tf)

        self.meta = meta if meta else {}

        self.order = self.meta.get('order', self.identify_model_order())
        self.source_data = self.meta.get('source_data', pd.DataFrame(columns=DataInputUtils.standard_fields))

    def identify_model_order(self) -> int:
        return int(self.tf.den[0][0].size)


class ModelView:
    def __init__(self, model: Model):
        self.model = model

    def plot_model_graph(self, *args, **kwargs):
        pass

    def get_model_data(self) -> Dict[str, Any]:
        pass

    def print_model_data(self, *args, **kwargs):
        pass
