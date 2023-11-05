import copy
import math
import pprint
from typing import Union, Tuple, Any, Dict, List

import control
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .envoirment import is_jupyter_environment


class PlotUtils:
    """
    Classe utilitária para plot de funções de transferência e outras informações relacionadas.
    """

    _jupyter_env = is_jupyter_environment()
    _max_plot_tf = 4

    @classmethod
    def plot_tf(
            cls,
            tf: Union[control.TransferFunction, Dict[str, control.TransferFunction]],
            discrete_data: Union[pd.Series, None] = None,
            settling_time_threshold: float = 0.02,
            pade: control.TransferFunction = None,
            scale: Union[float, Dict[str, float]] = 1,
            simulation_time: Union[float, None] = None,
            qt_points: int = 1000
    ):
        """
        Plota a resposta ao degrau de uma ou mais funções de transferência.

        Dependendo do ambiente, faz plot em uma nova janela, ou abaixo da célula de execução (jupyter).

        Verificar documentação dos Parâmetros para entender a utilização do método.

        Parameters
        ----------
        tf : Union[control.TransferFunction, Dict[str, control.TransferFunction]]
            :term:`Função de Transferência` (:class:`control.TransferFunction`)
            ou dicionário de funções de transferência a serem plotadas.
            As chaves do dicionário serão usadas como sufixo dos itens de legenda.

        discrete_data : pandas.Series, optional
            Série temporal (:class:`pandas.Series`)
            referente aos dados discretos de resposta a sinal degrau a serem sobrepostos ao gráfico.

        settling_time_threshold : float, optional
            Limiar de tempo de acomodação, por padrão 0.02 (2%).

        pade : control.TransferFunction, optional
            :term:`Função de Transferência` de Pade para consideração do atraso na resposta ao sinal degrau.

        scale : Union[float, Dict[str, float]], optional
            Fator de escala para as funções de transferência, por padrão 1.
            Se for informado um :class:`float`, o fator será aplicado a todas as funções de transferência.
            Se um :class:`dict` for informado, o fator será aplicado para as funções de transferência cujo sufixo
            coincidir com alguma chave do dicionário.

        simulation_time : float, optional
            Tempo de simulação, caso não seja informado, o tempo de acomodação da :term:`Função de Transferência` será
            usado como base, se não for possível calcular, será utilizado ``simulation_time = 100`` no lugar.

        qt_points : int, optional
            Quantidade de pontos para a simulação, por padrão 1000.

        Examples
        --------
        >>> act.PlotUtils.plot_tf(tf, discrete_data, scale=step_signal_value)

        .. image:: ../image_resources/plot_utills_plot_tf_example.png
        """

        if isinstance(tf, control.TransferFunction):
            tfs = {'': tf}
        else:
            tfs = tf

        if isinstance(scale, (int, float)):
            scale = {'': scale}

        legend_kwargs = cls._setup_plot_env()

        colors = cls._generate_contrasting_colors(
            len(tfs) * cls._max_plot_tf + 1
            if discrete_data is not None
            else len(tfs) * cls._max_plot_tf
        )

        if discrete_data is not None:
            discrete_data.plot(label='Discrete Data', color=colors.pop())

        for sufix, tf in tfs.items():
            cls._plot_tf(
                tf=tf,
                colors=colors,
                sufix=sufix,
                settling_time_threshold=settling_time_threshold,
                pade=pade,
                scale=scale.get(sufix, 1),
                simulation_time=simulation_time,
                qt_points=qt_points,
            )

        plt.xlabel('Time')
        plt.ylabel('Response')
        plt.title('Step Response')
        plt.legend(**legend_kwargs)
        plt.show()

    @classmethod
    def _setup_plot_env(cls) -> Dict[str, Any]:
        """
        Prepara o ambiente para plot de dados e retorna as opções de legenda.

        Returns
        -------
        Dict[str, Any]
            Opções de legenda.
        """
        if cls._jupyter_env:
            return {
                'loc': 'upper center',
                'bbox_to_anchor': (1.25, 0.45)
            }

        else:
            plt.switch_backend('TkAgg')
            return {
                'loc': 'lower right'
            }

    @classmethod
    def _setup_plot_tf_data(
            cls,
            tf: control.TransferFunction,
            settling_time_threshold: float = 0.02,
            pade: control.TransferFunction = None,
            scale: float = 1,
            simulation_time: Union[float, None] = None,
            qt_points: int = 1000,
    ) -> Tuple[Dict[str, float], pd.Series]:
        """
        Calcula as informações e cria uma série temporal
        (:class:`pandas.Series`)
        referentes a resposta a sinal degrau da :term:`Função de Transferência`.

        Parameters
        ----------
        tf : control.TransferFunction
            :term:`Função de Transferência` (:class:`control.TransferFunction`).

        settling_time_threshold : float, optional
            Limiar de tempo de acomodação, por padrão 0.02 (2%).

        pade : control.TransferFunction, optional
            :term:`Função de Transferência` de Pade para consideração do atraso na resposta ao sinal degrau.

        scale : float, optional
            Fator de escala para a :term:`Função de Transferência`, por padrão 1.

        simulation_time : float, optional
            Tempo de simulação, caso não seja informado, o tempo de acomodação da :term:`Função de Transferência` será
            usado como base, se não for possível calcular, será utilizado ``simulation_time = 100`` no lugar.

        qt_points : int, optional
            Quantidade de pontos para a simulação, por padrão 1000.

        Returns
        -------
        Tuple[Dict[str, float], pd.Series]
            Informações sobre a :term:`Função de Transferência` e dados temporais.
        """

        # Setup tf
        tf = copy.copy(tf)
        if pade is not None:
            tf = tf * pade

        # Get tf response info
        info = control.step_info(tf, T=simulation_time, SettlingTimeThreshold=settling_time_threshold)

        # Define a max time for forced response
        if simulation_time is None:
            simulation_time = info['SettlingTime'] * 3
            if math.isnan(simulation_time):
                simulation_time = 100

        # get tf temporal series
        data = cls._get_tf_temporal_series(tf, simulation_time, scale, qt_points)

        return info, data

    @classmethod
    def _plot_tf(
            cls,
            tf: control.TransferFunction,
            colors: List[Any],
            sufix: str = '',
            settling_time_threshold: float = 0.02,
            pade: control.TransferFunction = None,
            scale: float = 1,
            simulation_time: Union[float, None] = None,
            qt_points: int = 1000,
    ):
        """
        Plota a resposta ao degrau de uma :term:`Função de Transferência`.

        Parameters
        ----------
        tf : control.TransferFunction
            :term:`Função de Transferência` (:class:`control.TransferFunction`).

        colors : List[Any]
            Lista de cores para o plot (cores utilizadas serão removidas da lista com o método :meth:`list.pop`).

        sufix : str, optional
            Sufixo para a legenda.

        settling_time_threshold : float, optional
            Limiar de tempo de acomodação, por padrão 0.02 (2%).

        pade : control.TransferFunction, optional
            :term:`Função de Transferência` de Pade para consideração do atraso na resposta ao sinal degrau.

        scale : float, optional
            Fator de escala para a :term:`Função de Transferência`, por padrão 1.

        simulation_time : float, optional
            Tempo de simulação, caso não seja informado, o tempo de acomodação da :term:`Função de Transferência` será
            usado como base, se não for possível calcular, será utilizado ``simulation_time = 100`` no lugar.

        qt_points : int, optional
            Quantidade de pontos para a simulação, por padrão 1000.
        """
        # setup legend sufix
        legend_sufix = f'{sufix} ' if sufix != '' else ''

        # setup data
        info, data = cls._setup_plot_tf_data(
            tf=tf, settling_time_threshold=settling_time_threshold, pade=pade,
            scale=scale, simulation_time=simulation_time, qt_points=qt_points,
        )

        # plot response data (plot1)
        plt.plot(data.index, data.values, label=f'{legend_sufix} tf', color=colors.pop())

        # plot steady state horizontal lines (plot2)
        steady_state_color = colors.pop()

        cls._plot_steady_state_lines(
            steady_state_value=info['SteadyStateValue'] * scale,
            steady_state_color=steady_state_color,
            legend_sufix=legend_sufix,
            settling_time_threshold=settling_time_threshold,
        )

        # plot settling time vertical line (plot3)
        cls.plot_vertical(
            (info['SettlingTime'], data.loc[min(data.index, key=lambda x: abs(x - info['SettlingTime']))]),
            color=colors.pop(),
            legend_sufix=f'{legend_sufix} ta',
        )

        # plot overshoot vertical line (plot4)
        if info['Overshoot'] != 0:
            overshoot = (1 + info['Overshoot'] / 100) * info['SteadyStateValue'] * scale
            cls.plot_vertical(
                ((data - overshoot).abs().idxmin(), overshoot),
                color=colors.pop(),
                legend_sufix=f'{legend_sufix} So',
            )

        # up to 4 plots!

    @staticmethod
    def _get_tf_temporal_series(
            tf: control.TransferFunction,
            simulation_time: float,
            scale: float = 1,
            qt_points: int = 1000,
    ) -> pd.Series:
        """
        Gera a série temporal da resposta ao degrau de uma :term:`Função de Transferência`.

        Parameters
        ----------
        tf : control.TransferFunction
            :term:`Função de Transferência`.

        simulation_time : float
            Tempo de simulação.

        scale : float, optional
            Fator de escala para a :term:`Função de Transferência`, por padrão 1.

        qt_points : int, optional
            Quantidade de pontos para a simulação, por padrão 1000.

        Returns
        -------
        pd.Series
            Série temporal da resposta ao degrau.
        """
        # Generate response data
        time = np.linspace(0, simulation_time, num=qt_points)
        time, response = control.forced_response(tf, time, np.ones_like(time))

        # Rescale response to scale
        response = response * scale

        # Turn response data into temporal series
        return pd.Series(response, time)

    @staticmethod
    def _plot_steady_state_lines(
            steady_state_value: float,
            steady_state_color,
            legend_sufix: str,
            settling_time_threshold: float = 0.02
    ):
        """
        Plota linhas horizontais referentes ao valor de regime e sua variação percentual.

        Parameters
        ----------
        steady_state_value : float
            Valor de regime de uma resposta de um :term:`Sistema` a sinal degrau.

        steady_state_color
            Cor para as linhas horizontais.

        legend_sufix : str
            Sufixo para a legenda.

        settling_time_threshold : float, optional
            Limiar de tempo de acomodação, por padrão 0.02 (2%).
        """
        plt.axhline(
            y=steady_state_value,
            color=steady_state_color,
            label=f"{legend_sufix} vreg"
        )
        plt.axhline(
            y=steady_state_value * (1 + settling_time_threshold),
            color=steady_state_color, linestyle='--',
            label=f"{legend_sufix} vreg+/-{settling_time_threshold * 100}%"
        )
        plt.axhline(
            y=steady_state_value * (1 - settling_time_threshold),
            color=steady_state_color,
            linestyle='--'
        )

    @classmethod
    def plot_vertical(cls, coordinate: Tuple[float, float], color: str, legend_sufix: str,
                      linestyle: str = '--', marker: str = 'o'):
        """
        Plota uma linha vertical em um gráfico.

        Parameters
        ----------
        coordinate : Tuple[float, float]
            Coordenadas (tempo, valor) para a linha vertical.

        color : str
            Cor para a linha vertical.

        legend_sufix : str
            Legenda para a linha vertical.

        linestyle : str, optional
            Estilo de linha, por padrão '--'.

        marker : str, optional
            Marcador para a linha vertical, por padrão 'o'.
        """
        pd.Series((coordinate[1],), (coordinate[0],)).plot(color=color, label=legend_sufix, linestyle=linestyle,
                                                           marker=marker)
        plt.axvline(x=coordinate[0], color=color, linestyle=linestyle)

    @staticmethod
    def _generate_contrasting_colors(num_colors: int, saturation: float = 0.7, brightness: float = 0.7):
        """
        Gera uma lista de cores contrastantes.

        As cores geradas estão em um formato aceito como valor de cor pelas ferramentas de plot da bilblioteca
        :mod:`matplotlib`.

        Parameters
        ----------
        num_colors : int
            Número de cores a serem geradas.

        saturation : float, optional
            Saturação das cores, por padrão 0.7.

        brightness : float, optional
            Brilho das cores, por padrão 0.7.

        Returns
        -------
        List
            Lista de cores contrastantes.
        """
        hues = [i / num_colors for i in range(num_colors)]
        colors = [plt.colormaps.get_cmap('hsv')(h) for h in hues]
        colors = [(item[0], saturation * item[1], brightness * item[2], item[3]) for item in colors]
        return colors

    @classmethod
    def print_tf(cls, tf: Any):
        """
        Imprime a :term:`Função de Transferência` na tela.

        Caso esteja em um ambiente `jupyter <https://jupyter.org/>`_, a função
        :func:`~IPython.display.display`
        é chamada para que a :term:`Função de Transferência` seja mostrada com formatação matemática.
        """
        if cls._jupyter_env:
            from IPython import display
            display.display(tf)
        else:
            print(tf)

    @classmethod
    def pprint_dict(cls, di: Dict[str, Any]):
        """
        Exibe um dicionário de forma formatada.

        Caso esteja em um ambiente `jupyter <https://jupyter.org/>`_, os dados do dicionario são transformados em um
        :class:`pandas.DataFrame` e a função
        :func:`~IPython.display.display`
        e utilizada para plota-lo de forma formatada.

        Parameters
        ----------
        di : Dict[str, Any]
            Dicionário a ser exibido.
        """
        if cls._jupyter_env:
            from IPython import display
            df = pd.DataFrame([di])
            display.display(df)
        else:
            pprint.pprint(di)
