from typing import Union, Tuple, Any

import control
import pandas as pd
import matplotlib.pyplot as plt

from .envoirment import is_jupyter_environment


class PlotUtils:
    jupyter_env = is_jupyter_environment()

    @classmethod
    def plot_tf(
            cls,
            tf: control.TransferFunction,
            discrete_data: Union[pd.Series, None] = None,
            settling_time: Union[float] = 0.02,
    ):

        if cls.jupyter_env:
            legend_kwargs = {
                'loc': 'upper center',
                'bbox_to_anchor': (1.25, 0.45)
            }

        else:
            legend_kwargs = {
                'loc': 'lower right'
            }
            plt.switch_backend('TkAgg')

        time, response = control.step_response(tf)
        data = pd.Series(response, time)
        info = control.step_info(tf, SettlingTimeThreshold=settling_time)

        plt.plot(time, response, label='Tf', color='red')
        if discrete_data is not None:
            discrete_data.plot(label='Discrete Data', color='blue')
        plt.xlabel('Time')
        plt.ylabel('Response')
        plt.title('Step Response of Model')

        plt.axhline(y=info['SteadyStateValue'], color='orange', label="vreg")
        plt.axhline(y=info['SteadyStateValue'] * (1 + settling_time), color='orange', linestyle='--',
                    label=f"vreg+/-{settling_time*100}%")
        plt.axhline(y=info['SteadyStateValue'] * (1 - settling_time), color='orange', linestyle='--')

        cls.plot_vertical(
            (info['SettlingTime'], data.loc[min(data.index, key=lambda x: abs(x - info['SettlingTime']))]),
            color='green',
            label='ta',
        )

        if info['Overshoot'] != 0:
            os = (1 + info['Overshoot']/100)*info['SteadyStateValue']
            cls.plot_vertical(
                ((data - os).abs().idxmin(), os),
                color='purple',
                label='So',
            )

        plt.legend(**legend_kwargs)
        plt.show()

    @classmethod
    def plot_vertical(cls, coordinate: Tuple[float, float], color: str, label: str,
                      linestyle: str = '--', marker: str = 'o'):
        pd.Series((coordinate[1],), (coordinate[0],)).plot(color=color, label=label, linestyle=linestyle, marker=marker)
        plt.axvline(x=coordinate[0], color=color, linestyle=linestyle)

    @classmethod
    def print_tf(cls, tf: Any):
        if cls.jupyter_env:
            from IPython import display
            display.display(tf)
        else:
            print(tf)
