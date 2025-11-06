from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import control
import matplotlib.pyplot as plt

from ..model.model import Model
from ..utils.plot import PlotUtils


class Impulse:
    """
    Calcula e organiza a resposta impulsiva de um :term:`Modelo`.

    Esta classe utiliza a :term:`Função de Transferência` (:class:`control.TransferFunction`)
    armazenada em :class:`Model` para computar a resposta ao impulso via
    :func:`control.impulse_response`.

    Parameters
    ----------
    model : Model
        Modelo matemático do processo (contém :attr:`Model.tf`).

    Attributes
    ----------
    model : Model
        Referência ao modelo utilizado.
    view : ImpulseView
        Ferramentas de visualização para a resposta impulsiva.

    Examples
    --------
    >>> import auto_control_tools as act
    >>> num, den = [1], [1, 1]
    >>> m = act.Model((num, den))
    >>> ir = act.Impulse(m)
    >>> data = ir.get_impulse_response_data()
    >>> ir.view.plot()  # plota a resposta ao impulso
    """
    def __init__(self, model: Model):
        self.model = model
        self.view = ImpulseView(self)

    # ---------------------------
    #         Core API
    # ---------------------------
    def compute(
        self,
        simulation_time: Optional[float] = None,
        qt_points: int = 1000,
        settling_time_threshold: float = 0.02,
    ) -> pd.Series:
        """
        Calcula a série temporal da resposta impulsiva.

        Parameters
        ----------
        simulation_time : float, optional
            Janela de simulação (s). Se None, tenta-se estimar a partir do
            :func:`control.step_info` (3x tempo de acomodação) com fallback para 100 s.
        qt_points : int, optional
            Número de pontos da simulação, por padrão 1000.
        settling_time_threshold : float, optional
            Limiar (em fração) para tempo de acomodação usado na estimação quando
            ``simulation_time`` é None. Padrão 0.02 (2%).

        Returns
        -------
        pandas.Series
            Série com index = tempo (s ou k para discreto) e valores = resposta impulsiva.
        """
        tf = self.model.tf

        # Estima janela de simulação caso não informada
        if simulation_time is None:
            try:
                info = control.step_info(
                    tf,
                    T=self.model.get_simulation_time(),
                    SettlingTimeThreshold=settling_time_threshold,
                )
                sim_t = float(info.get("SettlingTime", float("nan"))) * 3.0
                if np.isnan(sim_t) or sim_t <= 0.0:
                    sim_t = 100.0
            except Exception:
                sim_t = 100.0
        else:
            sim_t = float(simulation_time)

        # Gera grade temporal e calcula resposta ao impulso
        T = np.linspace(0.0, sim_t, int(qt_points)) if tf.dt is None else np.arange(int(qt_points))
        T, y = control.impulse_response(tf, T=T)  # T shape (N,), y shape (N,1) ou (N,)

        y = np.squeeze(y).astype(float)
        serie = pd.Series(data=y, index=T, name="impulse_response")
        return serie

    def get_impulse_response_data(
        self,
        simulation_time: Optional[float] = None,
        qt_points: int = 1000,
        settling_time_threshold: float = 0.02,
    ) -> Dict[str, Any]:
        """
        Retorna a resposta impulsiva em formato serializável + métricas básicas.

        Parameters
        ----------
        simulation_time : float, optional
            Ver :meth:`compute`.
        qt_points : int, optional
            Ver :meth:`compute`.
        settling_time_threshold : float, optional
            Ver :meth:`compute`.

        Returns
        -------
        dict
            Dicionário com chaves:
            - ``time``: lista de tempos (float);
            - ``response``: lista da resposta (float);
            - ``metrics``: dict com métricas úteis (``Peak``, ``PeakTime``, ``Energy``, ``Area``).
        """
        serie = self.compute(
            simulation_time=simulation_time,
            qt_points=qt_points,
            settling_time_threshold=settling_time_threshold,
        )

        # Métricas simples e úteis para análise:
        # - Peak, PeakTime: pico (em módulo) e seu instante
        # - Energy: ∫ y(t)^2 dt  (boa p/ comparar "energia" da resposta)
        # - Area: ∫ y(t) dt  (igual ao valor final da resposta ao degrau para sistemas estáveis)
        idx_peak = int(np.argmax(np.abs(serie.values)))
        peak = float(serie.values[idx_peak])
        peak_time = float(serie.index[idx_peak])
        energy = float(np.trapz(serie.values ** 2, serie.index))
        area = float(np.trapz(serie.values, serie.index))

        return {
            "time": serie.index.astype(float).tolist(),
            "response": serie.values.astype(float).tolist(),
            "metrics": {
                "Peak": peak,
                "PeakTime": peak_time,
                "Energy": energy,
                "Area": area,
            },
        }


class ImpulseView:
    """
    Visualização da resposta impulsiva vinculada a :class:`Impulse`.
    """
    def __init__(self, impulse: Impulse):
        self.impulse = impulse

    def plot(
        self,
        simulation_time: Optional[float] = None,
        qt_points: int = 1000,
        settling_time_threshold: float = 0.02,
        legend_suffix: str = "",
    ) -> None:
        """
        Plota a resposta ao impulso do :term:`Modelo`.

        Parameters
        ----------
        simulation_time : float, optional
            Ver :meth:`Impulse.compute`.
        qt_points : int, optional
            Ver :meth:`Impulse.compute`.
        settling_time_threshold : float, optional
            Ver :meth:`Impulse.compute`.
        legend_suffix : str, optional
            Sufixo para legenda (útil ao comparar múltiplos sistemas).
        """
        serie = self.impulse.compute(
            simulation_time=simulation_time,
            qt_points=qt_points,
            settling_time_threshold=settling_time_threshold,
        )

        legend_kwargs = PlotUtils._setup_plot_env()  # segue padrão visual da lib
        plt.plot(
            serie.index,
            serie.values,
            label=f"impulse{(' ' + legend_suffix) if legend_suffix else ''}",
        )
        plt.xlabel("Tempo [s]" if self.impulse.model.tf.dt is None else "Passos [k]")
        plt.ylabel("Amplitude")
        plt.title("Resposta ao Impulso")
        plt.grid(True)
        plt.legend(**legend_kwargs)
        plt.show()


# ------------------------------------------------------------------
#  Opcional: API funcional enxuta (útil para notebooks e exemplos)
# ------------------------------------------------------------------
def impulse_response(
    model: Model,
    simulation_time: Optional[float] = None,
    qt_points: int = 1000,
    settling_time_threshold: float = 0.02,
) -> pd.Series:
    """
    Atalho funcional para :meth:`Impulse.compute`.

    Returns
    -------
    pandas.Series
        Série com a resposta ao impulso.
    """
    return Impulse(model).compute(
        simulation_time=simulation_time,
        qt_points=qt_points,
        settling_time_threshold=settling_time_threshold,
    )
