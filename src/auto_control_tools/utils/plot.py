import copy
from typing import Union, Tuple, Any, Dict

import control
import pandas as pd
import matplotlib.pyplot as plt

from .envoirment import is_jupyter_environment


class PlotUtils:
    jupyter_env = is_jupyter_environment()
    _max_items_ploted = 5

    @classmethod
    def plot_tf(
            cls,
            tf: Union[control.TransferFunction, Dict[str, control.TransferFunction]],
            discrete_data: Union[pd.Series, None] = None,
            settling_time: Union[float] = 0.02,
            pade: control.TransferFunction = None
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

        if isinstance(tf, control.TransferFunction):
            tfs = {'': tf}
        else:
            tfs = tf

        colors = cls._generate_contrasting_colors(len(tfs) * cls._max_items_ploted)

        for sufix, tf in tfs.items():
            tf = copy.copy(tf)
            if pade is not None:
                tf = tf * pade

            time, response = control.step_response(tf)
            data = pd.Series(response, time)
            info = control.step_info(tf, SettlingTimeThreshold=settling_time)

            plt.plot(time, response, label=f'{sufix} tf', color=colors.pop())
            if discrete_data is not None:
                discrete_data.plot(label=f'{sufix} Discrete Data', color=colors.pop())

            steady_state_color = colors.pop()
            plt.axhline(y=info['SteadyStateValue'], color=steady_state_color, label=f"{sufix} vreg")
            plt.axhline(y=info['SteadyStateValue'] * (1 + settling_time), color=steady_state_color, linestyle='--',
                        label=f"{sufix} vreg+/-{settling_time*100}%")
            plt.axhline(y=info['SteadyStateValue'] * (1 - settling_time), color=steady_state_color, linestyle='--')

            cls.plot_vertical(
                (info['SettlingTime'], data.loc[min(data.index, key=lambda x: abs(x - info['SettlingTime']))]),
                color=colors.pop(),
                label=f'{sufix} ta',
            )

            if info['Overshoot'] != 0:
                os = (1 + info['Overshoot']/100)*info['SteadyStateValue']
                cls.plot_vertical(
                    ((data - os).abs().idxmin(), os),
                    color=colors.pop(),
                    label=f'{sufix} So',
                )

        plt.xlabel('Time')
        plt.ylabel('Response')
        plt.title('Step Response')
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

    @staticmethod
    def _generate_contrasting_colors(num_colors: int, saturation: float = 0.7, brightness: float = 0.7):
        hues = [i / num_colors for i in range(num_colors)]
        colors = [plt.colormaps.get_cmap('hsv')(h) for h in hues]
        colors = [[item[0], saturation * item[1], brightness * item[2], item[3]] for item in colors]
        return colors
