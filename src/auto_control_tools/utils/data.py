from typing import Tuple

import pandas as pd
from scipy.signal import lfilter

from .envoirment import is_jupyter_environment


class DataUtils:
    _jupyter_env = is_jupyter_environment()

    @staticmethod
    def linfilter(series: pd.Series, smothness: int) -> pd.Series:
        # the larger smothness is, the smoother curve will be
        b = [1.0 / smothness] * smothness
        a = 1
        return pd.Series(lfilter(b, a, series), name=series.name)

    @staticmethod
    def get_vreg(tf_data: pd.Series, delta: float = 0.02) -> Tuple[float, float]:
        for idx, value in tf_data.iloc[::1].items():
            local_s = tf_data[idx:]
            mean = local_s.mean()

            if all((local_s < (1 + delta) * mean) & (local_s > (1 - delta) * mean)):
                return idx, mean
        return 0, 0

    @staticmethod
    def get_max_tan(tf_data: pd.Series) -> Tuple[float, float]:
        diff = tf_data.diff()
        return float(diff.idxmax()), float(max(diff[1:]))

    @staticmethod
    def get_time_from_inclination(ref_time: float, ref_value: float, inclination: float, value: float) -> float:
        return (value - ref_value + inclination * ref_time) / inclination
