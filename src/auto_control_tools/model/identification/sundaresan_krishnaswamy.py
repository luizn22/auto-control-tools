from typing import Union

from ..first_order_model import FirstOrderModel
from .base import BaseModelIdentification
from ...utils.data import DataUtils
from ...utils.data_input import DataInputUtils


class SundaresanKrishnaswamyModelIdentification(BaseModelIdentification):
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

        t1 = tf_data[tf_data == min(tf_data[tf_data >= vreg*0.353])].index[0]
        t2 = tf_data[tf_data == min(tf_data[tf_data >= vreg*0.853])].index[0]

        K = vreg / step_signal
        tau = 0.67*(t2 - t1)
        theta = 1.3 * t1 - 0.29 * t2

        if ignore_delay_threshold is not None and theta < ignore_delay_threshold:
            theta = 0

        return FirstOrderModel(K, tau, theta, source_data=tf_data)
