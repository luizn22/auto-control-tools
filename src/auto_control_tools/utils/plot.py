from typing import Union, Dict, Any

import control
import pandas as pd
import matplotlib.pyplot as plt

from auto_control_tools.utils.envoirment import is_jupyter_environment


class PlotUtils:
    jupyter_env = is_jupyter_environment()

    @classmethod
    def plot_tf(cls, tf: control.TransferFunction, data_dict: Dict[str, Any],
                discrete_data: Union[pd.DataFrame, None] = None):
        if cls.jupyter_env:
            time, response = control.step_response(tf)
            plt.plot(time, response)
            plt.xlabel('Time')
            plt.ylabel('Response')
            plt.title('Step Response of Model')

            plt.show()
