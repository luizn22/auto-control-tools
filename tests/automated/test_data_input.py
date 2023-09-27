import os.path
import random

import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from auto_control_tools.utils.data_input import DataInputUtils


@pytest.mark.parametrize('fields, save_as', [
    (['time', 'input', 'output'], 'csv'),
    (['input', 'output'], 'csv'),
    (['time', 'output'], 'csv'),
    (['output'], 'csv'),
    (['time', 'input', 'output'], 'xlsx'),
    (['input', 'output'], 'xlsx'),
    (['time', 'output'], 'xlsx'),
    (['output'], 'xlsx'),
    (['output'], 'not_allowed'),
])
def test_data_input(tmp_path, fields, save_as):
    d = tmp_path / "sub"
    d.mkdir()
    folder_path_str = str(d.resolve())
    file_path = os.path.join(folder_path_str, f'test_data_input.{save_as}')
    if save_as not in DataInputUtils.allowed_file_type:
        with pytest.raises(ValueError, match=f'{save_as} is not an allowed file type'):
            DataInputUtils.create_table_with_fields(folder_path_str, fields, 'test_data_input', save_as)
    else:
        DataInputUtils.create_table_with_fields(folder_path_str, fields, 'test_data_input', save_as)

        if file_path.lower().endswith(".xlsx"):
            df = pd.read_excel(file_path)
        elif file_path.lower().endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            raise ValueError("Unsupported file format. Please use .xlsx or .csv extension.")

        # Check if the DataFrame has headers
        if df.columns.empty:
            raise ValueError("Columns were not created")
        else:
            # Fill columns with 100 rows of random float values
            for column in df.columns:
                df[column] = [random.uniform(0, 1) for _ in range(100)]

            # Save the modified DataFrame back to the original file
            if file_path.lower().endswith(".xlsx"):
                df.to_excel(file_path, index=False)
            elif file_path.lower().endswith(".csv"):
                df.to_csv(file_path, index=False)

        assert_frame_equal(df, DataInputUtils.read_table_with_fields(file_path, fields))
