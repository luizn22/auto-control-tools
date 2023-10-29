from typing import Union

from ..first_order_model import FirstOrderModel
from .base import BaseModelIdentification
from ...utils.data import DataUtils
from ...utils.data_input import DataInputUtils


class HagglundModelIdentification(BaseModelIdentification):
    @classmethod
    def get_model(
            cls,
            path: str,
            sample_time: Union[None, float] = None,
            step_signal: Union[None, float] = None,
            ignore_delay_threshold: Union[None, float] = 0.5,

    ) -> FirstOrderModel:
        df = DataInputUtils.get_model_data_default(path, sample_time, step_signal)
        tf_data, step_signal = DataUtils.setup_data_default(df, sample_time, step_signal)

        idx_vreg, vreg = DataUtils.get_vreg(tf_data)
        idx_tan, tan = DataUtils.get_max_tan(tf_data)
        tan_point_value = tf_data.loc[tf_data.index == idx_tan].iloc[0]

        t1 = DataUtils.get_time_from_inclination(idx_tan, tan_point_value, tan_point_value, 0)
        t2 = tf_data[tf_data == min(tf_data[tf_data >= vreg*0.632])].index[0]

        K = vreg / step_signal
        tau = t2 - t1
        theta = t1

        if ignore_delay_threshold is not None and theta < ignore_delay_threshold:
            theta = 0

        return FirstOrderModel(K, tau, theta, source_data=tf_data)
