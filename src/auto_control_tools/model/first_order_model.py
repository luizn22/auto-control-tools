from typing import Union

import control
import pandas as pd
import sympy as sp

from .model import Model


class FirstOrderModel(Model):
    """
    Classe de modelo para casos específicos de modelos de primeira ordem que respeitam o formato:
        (K/(tau*s + 1)) * exp(-teta*s)
    """
    def __init__(
            self,
            K: float,
            tau: float,
            teta: float = 0,
            pade_degree: int = 5,
            source_data: Union[pd.Series, None] = None,
    ):
        super().__init__([[K], [tau, 1]], source_data=source_data)
        self.K = K
        self.tau = tau
        self.teta = teta

        if teta != 0 and pade_degree > 0:
            self.pade = control.tf(*control.pade(teta, pade_degree))

            s = sp.symbols('s')
            self.tf_symbolic = self.tf_symbolic * sp.exp(-teta*s)
