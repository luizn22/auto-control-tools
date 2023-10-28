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
    def _get_model_data_default(cls, path: str, sample_time: Union[None, float] = None,
                               step_signal: Union[float, None] = None) -> pd.DataFrame:
        expected_fields = cls._expected_fields(sample_time, step_signal)
        df = DataInputUtils.read_table_with_fields(path, expected_fields)

        if any(f not in df.columns for f in expected_fields):
            missing_fields = [f for f in expected_fields if f not in df.columns]
            raise ValueError(f'The fields {missing_fields} are required and were informed in the input data')

        return df

    @classmethod
    def _setup_data_default(cls, df: pd.DataFrame, sample_time: Union[float, None] = None,
                            step_signal: Union[float, None] = None,
                            use_lin_filter: bool = False, linfilter_sothness: int = 5
                            ) -> Tuple[pd.Series, float]:
        if sample_time is not None:
            df['time'] = df.index * step_signal

        if step_signal is None:  # in case step signal is not informed, get it and then remove column
            step_signal = max(df['input'])
            df = cls._trunk_data_input(df)

        df = cls._offset_data_output(df)

        s = pd.Series(df['output'].values, index=df['time'])

        if use_lin_filter:
            s = DataUtils.linfilter(s, linfilter_sothness)

        return s, step_signal

    @classmethod
    def _offset_data_output(cls, df: pd.DataFrame) -> pd.DataFrame:
        df['output'] = df['output'].apply(lambda x: x if x > 0 else 0)
        df['output'] = df['output'] - min(df['output'])
        return df

    @classmethod
    def _trunk_data_input(cls, df: pd.DataFrame) -> pd.DataFrame:
        if 'input' not in df.columns:
            raise ValueError('input is not in df.columns')
        return df.loc[df['input'] != 0][[col for col in df if col != 'input']]

    @classmethod
    def get_data_input_layout(cls, path: str, sample_time: Union[float, None] = None,
                              step_signal: Union[float, None] = None):
        DataInputUtils.create_table_with_fields(path, cls._expected_fields(sample_time, step_signal))

    @classmethod
    def _expected_fields(cls, sample_time: Union[float, None] = None,
                         step_signal: Union[float, None] = None) -> List[str]:
        fields = copy(DataInputUtils.standard_fields)

        if sample_time is not None:
            fields.remove('time')

        if step_signal is not None:
            fields.remove('input')

        return fields
