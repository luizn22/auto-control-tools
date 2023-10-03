from typing import Dict, Any

import pandas as pd
import pprint
from scipy.signal import lfilter

from .envoirment import is_jupyter_environment
from IPython import display


class DataUtils:
    jupyter_env = is_jupyter_environment()

    @classmethod
    def pprint_dict(cls, di: Dict[str, Any]):
        if cls.jupyter_env:
            df = pd.DataFrame([di])
            display.display(df)
        else:
            pprint.pprint(di)

    @staticmethod
    def linfilter(series: pd.Series, smothness: int) -> pd.Series:
        # the larger smothness is, the smoother curve will be
        b = [1.0 / smothness] * smothness
        a = 1
        return pd.Series(lfilter(b, a, series), name=series.name)
