import os
from typing import List

import pandas as pd


class DataInputUtils:
    standard_fields = ['time', 'input', 'output']
    allowed_file_type = ['csv', 'xlsx']

    @classmethod
    def create_table_with_fields(
            cls,
            path: str,
            fields: List[str],
            table_name: str = 'data_input',
            save_as: str = 'csv'
    ):
        if save_as not in cls.allowed_file_type:
            raise ValueError(f'{save_as} is not an allowed file type')

        file_path = os.path.join(path, f'{table_name}.{save_as}')
        df = pd.DataFrame(columns=fields)

        if save_as == 'csv':
            df.to_csv(file_path)
        elif save_as == 'xlsx':
            df.to_excel(file_path)
        else:
            raise ValueError(f'{save_as} is not an allowed file type')

    @classmethod
    def read_table_with_fields(cls, path: str, fields: List[str]):
        pass
