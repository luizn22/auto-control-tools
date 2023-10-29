import os
from typing import List, Union
from copy import copy

import pandas as pd


class DataInputUtils:
    default_table_name = 'data_input'  # update BaseModelIdentification.get_data_input_layout docstring if changed
    standard_fields = ['time', 'input', 'output']
    allowed_file_type = ['csv', 'xlsx']

    @classmethod
    def create_table_with_fields(
            cls,
            path: str,
            fields: List[str],
            table_name: str = default_table_name,
            save_as: str = 'csv'
    ) -> str:
        if save_as not in cls.allowed_file_type:
            raise ValueError(f'{save_as} is not an allowed file type')

        file_path = os.path.join(path, f'{table_name}.{save_as}')
        df = pd.DataFrame(columns=fields)

        if save_as == 'csv':
            df.to_csv(file_path, index=False)
        elif save_as == 'xlsx':
            df.to_excel(file_path, index=False)
        else:
            raise ValueError(f'{save_as} is not an allowed file type')

        return file_path

    @classmethod
    def read_table_with_fields(cls, path: str, fields: Union[List[str], None] = None) -> pd.DataFrame:
        if fields is None:
            fields = cls.standard_fields

        if path.lower().endswith(".xlsx"):
            df = pd.read_excel(path)
        elif path.lower().endswith(".csv"):
            df = pd.read_csv(path)
        else:
            raise ValueError(f'{os.path.splitext(path)[-1]} is not an allowed file type')

        df = df[[f for f in fields if f in df.columns]]

        return df

    @classmethod
    def expected_fields(cls, sample_time: Union[float, bool, None] = None,
                        step_signal: Union[float, bool, None] = None) -> List[str]:
        """
        Retorna a lista :attr:`standard_fields` com exeção dos campos sample_time ou step_signal caso sejam
        informados nos parâmetros.
        """
        fields = copy(cls.standard_fields)

        if sample_time is not None:
            fields.remove('time')

        if step_signal is not None:
            fields.remove('input')

        return fields

    @classmethod
    def get_model_data_default(cls, path: str, sample_time: Union[None, float] = None,
                               step_signal: Union[float, None] = None) -> pd.DataFrame:
        """
        Obtém os campos esperados, lê os dados do arquivo indicado através de
        :meth:`DataInputUtils.read_table_with_fields` verifica se não existem campos faltantes
        e retorna o pandas.DataFrame resultante.

        Parameters
        ----------
        path : str
            Caminho até o arquivo a ser lido. O leiaute pode ser obtido através de :meth:`get_data_input_layout`.

        sample_time : float, optional
            Valor do invervalo de amostragem. Caso informado, o intervalo de amostragem é considerado constante e
            igual ao valor fornecido.

        step_signal : float, optional
            Valor do sinal degrau de entrada. Se informado é considerado que o sinal está ativo em todos os momentos
            nos dados recebidos.

        Returns
        -------
        pandas.Dataframe os dados do arquivo recebido e as colunas esperadas
        """
        expected_fields = cls.expected_fields(sample_time, step_signal)
        df = cls.read_table_with_fields(path, expected_fields)

        if any(f not in df.columns for f in expected_fields):
            missing_fields = [f for f in expected_fields if f not in df.columns]
            raise ValueError(f'The fields {missing_fields} are required and were informed in the input data')

        return df
