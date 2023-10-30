from typing import Union

import numpy as np

from ..first_order_model import FirstOrderModel
from .base import BaseModelIdentification
from ...utils.data import DataUtils
from ...utils.data_input import DataInputUtils


class NishikawaModelIdentification(BaseModelIdentification):
    @classmethod
    def get_model(
            cls,
            path: str,
            sample_time: Union[None, float] = None,
            step_signal: Union[None, float] = None,
            ignore_delay_threshold: Union[None, float] = 0.5,
            settling_time_threshold: float = 0.02

    ) -> FirstOrderModel:
        df = DataInputUtils.get_model_data_default(path, sample_time, step_signal)
        tf_data, step_signal = DataUtils.setup_data_default(df, sample_time, step_signal)

        idx_vreg, vreg = DataUtils.get_vreg(tf_data, settling_time_threshold=settling_time_threshold)

        A0 = vreg*idx_vreg - np.trapz(tf_data[:idx_vreg], tf_data[:idx_vreg].index)  # type: ignore

        t0 = A0/vreg

        A1 = np.trapz(tf_data.loc[tf_data.index <= t0], tf_data.loc[tf_data.index <= t0].index)

        K = vreg / step_signal

        tau = A1/(0.368*vreg)
        theta = t0 - tau

        if ignore_delay_threshold is not None and theta < ignore_delay_threshold:
            theta = 0

        return FirstOrderModel(K, tau, theta, source_data=tf_data)
