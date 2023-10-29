from typing import Union

from ..first_order_model import FirstOrderModel
from .base import BaseModelIdentification
from ...utils.data import DataUtils


class SundaresanKrishnaswamyModelIdentification(BaseModelIdentification):
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

        idx_vreg, vreg = DataUtils.get_vreg(tf_data)

        t1 = tf_data[tf_data == min(tf_data[tf_data >= vreg*0.353])].index[0]
        t2 = tf_data[tf_data == min(tf_data[tf_data >= vreg*0.853])].index[0]

        K = vreg / step_signal
        tau = 0.67*(t2 - t1)
        theta = 1.3 * t1 - 0.29 * t2

        if theta < ignore_delay_threshold:
            theta = 0

        return FirstOrderModel(K, tau, theta, source_data=tf_data)
