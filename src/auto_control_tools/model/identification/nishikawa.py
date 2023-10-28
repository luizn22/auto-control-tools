from typing import Union

import numpy as np

from ..first_order_model import FirstOrderModel
from .base import BaseModelIdentification


class NishikawaModelIdentification(BaseModelIdentification):
    @classmethod
    def get_model(
            cls,
            path: str,
            sample_time: Union[None, float] = None,
            step_signal: Union[None, float] = None,
            ignore_delay_threshold: float = 0.5,

    ) -> FirstOrderModel:
        df = cls._get_model_data_default(path, sample_time, step_signal)
        tf_data, step_signal = cls._setup_data_default(df, sample_time, step_signal)

        idx_vreg, vreg = cls.get_vreg(tf_data)

        A0 = vreg*idx_vreg - np.trapz(tf_data[:idx_vreg], tf_data[:idx_vreg].index)  # type: ignore

        t0 = A0/vreg

        A1 = np.trapz(tf_data.loc[tf_data.index <= t0], tf_data.loc[tf_data.index <= t0].index)

        K = vreg / step_signal

        tau = A1/(0.368*vreg)
        teta = t0 - tau

        if teta < ignore_delay_threshold:
            teta = 0

        return FirstOrderModel(K, tau, teta, source_data=tf_data)
