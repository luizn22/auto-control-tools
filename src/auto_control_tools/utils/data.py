from typing import Dict, Any

import pandas as pd
import pprint

from auto_control_tools.utils.envoirment import is_jupyter_environment
from IPython import display


class DataUtils:
    jupyter_env = is_jupyter_environment()

    @classmethod
    def pprint_dict(cls, di: Dict[str, Any]):
        if cls.jupyter_env:
            df = pd.DataFrame([di])
            display.display(df)
        else:
            pprint.pprint(di)

