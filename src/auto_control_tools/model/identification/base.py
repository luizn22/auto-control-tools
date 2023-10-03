from abc import abstractmethod
from copy import copy
from typing import List, Tuple, Union

import pandas as pd

from ..model import Model
from ...utils.data import DataUtils
from ...utils.data_input import DataInputUtils


class BaseModelIdentification:
    @classmethod
    @abstractmethod
    def get_model(cls, *args, **kwargs) -> Model:
        raise NotImplementedError('get_model must be implemented in a subclass')

    @classmethod
    def get_model_data_default(cls, path: str, sample_time: float = None, step_signal: float = None) -> pd.DataFrame:
        expected_fields = cls._expected_fields(sample_time, step_signal)
        df = DataInputUtils.read_table_with_fields(path, expected_fields)

        if any(f not in df.columns for f in expected_fields):
            missing_fields = [f for f in expected_fields if f not in df.columns]
            raise ValueError(f'The fields {expected_fields} are required and were informed in the input data')

        return df

    @classmethod
    def setup_data_default(cls, df: pd.DataFrame, sample_time: float = None, step_signal: float = None,
                           use_lin_filter: bool = False, linfilter_sothness: int = 5
                           ) -> Tuple[pd.Series, float]:
        if sample_time is not None:
            df['time'] = df.index * step_signal

        if step_signal is None:  # in case step signal is not informed, get it and then remove column
            step_signal = max(df['input'])
            df = cls.trunk_data_input(df)

        df = cls.offset_data_output(df)

        s = pd.Series(df['output'].values, index=df['time'])

        if use_lin_filter:
            s = DataUtils.linfilter(s, linfilter_sothness)

        return s, step_signal

    @classmethod
    def offset_data_output(cls, df: pd.DataFrame) -> pd.DataFrame:
        df['output'] = df['output'] - min(df['output'])
        return df

    @classmethod
    def trunk_data_input(cls, df: pd.DataFrame) -> pd.DataFrame:
        if 'input' not in df.columns:
            raise ValueError('input is not in df.columns')
        return df.loc[df['input'] != 0][[col for col in df if col != 'input']]

    @classmethod
    def get_data_input_layout(cls, path: str, sample_time: float = None, step_signal: float = None):
        DataInputUtils.create_table_with_fields(path, cls._expected_fields(sample_time, step_signal))

    @classmethod
    def _expected_fields(cls, sample_time: float = None, step_signal: float = None) -> List[str]:
        fields = copy(DataInputUtils.standard_fields)

        if sample_time is not None:
            fields.remove('time')

        if step_signal is not None:
            fields.remove('input')

        return fields

    @classmethod
    def get_vreg(cls, tf_data: pd.Series, delta: float = 0.02) -> Tuple[float, float]:
        for idx, value in tf_data.iloc[::1].items():
            local_s = tf_data[idx:]
            mean = local_s.mean()

            if all((local_s < (1 + delta) * mean) & (local_s > (1 - delta) * mean)):
                return idx, mean
        return 0, 0

    @classmethod
    def get_max_tan(cls, tf_data: pd.Series) -> Tuple[float, float]:
        diff = tf_data.diff()
        return float(diff.idxmax()), float(max(diff[1:]))

    @classmethod
    def get_time_from_inclination(cls, ref_time: float, ref_value: float, inclination: float, value: float) -> float:
        return (value - ref_value + inclination * ref_time) / inclination
