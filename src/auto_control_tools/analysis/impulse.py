"""
Impulse Response Analysis Module
=================================

Este módulo fornece ferramentas para cálculo e análise completa da resposta ao impulso
de sistemas lineares invariantes no tempo (LTI).

.. module:: auto_control_tools.analysis.impulse
   :synopsis: Análise de resposta ao impulso com métricas temporais completas.

Classes
-------
.. autosummary::
   :toctree: generated/
   
   Impulse
   ImpulseView

Functions
---------
.. autosummary::
   :toctree: generated/
   
   impulse_response
   impulse_analysis

"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, Callable
import numpy as np
import pandas as pd
import control
import matplotlib.pyplot as plt

from ..model.model import Model
from ..utils.plot import PlotUtils


__all__ = ["Impulse", "ImpulseView", "impulse_response", "impulse_analysis"]


class Impulse:
    """
    Calcula e analisa a resposta impulsiva de um :term:`Modelo`.

    Esta classe utiliza a :term:`Função de Transferência` (:class:`control.TransferFunction`)
    armazenada em :class:`Model` para computar a resposta ao impulso via
    :func:`control.impulse_response` e fornece 20 métricas temporais completas.

    Parameters
    ----------
    model : Model
        Modelo matemático do processo (contém :attr:`Model.tf`).
    name : str, optional
        Nome do sistema para identificação nos gráficos e relatórios.
        Se não fornecido, usa "Sistema".

    Attributes
    ----------
    model : Model
        Referência ao modelo utilizado.
    name : str
        Nome identificador do sistema.
    view : ImpulseView
        Ferramentas de visualização para a resposta impulsiva.

    See Also
    --------
    Model : Classe de modelo do sistema.
    ImpulseView : Classe de visualização.
    control.impulse_response : Função base do python-control.

    Notes
    -----
    A resposta ao impulso :math:`h(t)` é a saída do sistema quando a entrada é
    um impulso unitário :math:`\\delta(t)`. Para um sistema com função de transferência
    :math:`G(s)`, temos:

    .. math::

        h(t) = \\mathcal{L}^{-1}\\{G(s)\\}

    A área sob a curva da resposta ao impulso é igual ao ganho DC do sistema:

    .. math::

        \\int_0^{\\infty} h(t)\\,dt = G(0) = \\lim_{s \\to 0} G(s)

    Examples
    --------
    Exemplo básico com sistema de primeira ordem:

    >>> import auto_control_tools as act
    >>> num, den = [1], [1, 1]
    >>> m = act.Model((num, den))
    >>> ir = act.Impulse(m)
    >>> data = ir.get_impulse_response_data()
    >>> ir.view.plot()  # plota a resposta ao impulso

    Análise completa com métricas:

    >>> import auto_control_tools as act
    >>> num, den = [25], [1, 5, 25]  # Sistema subamortecido
    >>> m = act.Model((num, den))
    >>> ir = act.Impulse(m, name="Sistema Subamortecido")
    >>> metrics = ir.get_metrics()  # 20 métricas completas
    >>> ir.summary()  # Relatório completo
    >>> ir.view.plot(show_metrics=True)  # Gráfico com anotações
    """

    def __init__(self, model: Model, name: Optional[str] = None):
        self.model = model
        self.name = name or "Sistema"
        self.view = ImpulseView(self)
        self._response_data: Optional[pd.Series] = None

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
            Janela de simulação (s). Se None, tenta-se estimar automaticamente
            usando o tempo de acomodação ou análise dos polos.
        qt_points : int, default 1000
            Número de pontos da simulação.
        settling_time_threshold : float, default 0.02
            Limiar (em fração) para tempo de acomodação usado na estimação quando
            ``simulation_time`` é None. Padrão 0.02 (2%).

        Returns
        -------
        pandas.Series
            Série com index = tempo (s ou k para discreto) e valores = resposta impulsiva.
            O nome da série é ``impulse_{nome_do_sistema}``.

        See Also
        --------
        get_impulse_response_data : Retorna dados em formato dicionário.
        get_metrics : Calcula métricas temporais completas.

        Notes
        -----
        Para sistemas instáveis, o tempo de simulação é automaticamente limitado
        para evitar divergência numérica.

        Examples
        --------
        >>> import auto_control_tools as act
        >>> m = act.Model(([1], [1, 2, 1]))
        >>> ir = act.Impulse(m)
        >>> serie = ir.compute(simulation_time=10.0, qt_points=500)
        >>> serie.head()
        0.00    0.000000
        0.02    0.019605
        0.04    0.038431
        0.06    0.056508
        0.08    0.073868
        Name: impulse_Sistema, dtype: float64
        """
        tf = self.model.tf

        # Estima janela de simulação caso não informada
        if simulation_time is None:
            simulation_time = self._estimate_simulation_time(settling_time_threshold)

        # Gera grade temporal e calcula resposta ao impulso
        if tf.dt is None or tf.dt == 0:  # Sistema contínuo
            T = np.linspace(0.0, simulation_time, int(qt_points))
        else:  # Sistema discreto
            T = np.arange(0, int(simulation_time / tf.dt)) * tf.dt

        try:
            T, y = control.impulse_response(tf, T=T)
            y = np.squeeze(y).astype(float)
        except Exception as e:
            # Fallback: resposta zero em caso de erro numérico
            y = np.zeros_like(T)

        serie = pd.Series(data=y, index=T, name=f"impulse_{self.name}")
        self._response_data = serie
        return serie

    def _estimate_simulation_time(self, settling_threshold: float = 0.02) -> float:
        """
        Estima tempo de simulação baseado no tempo de acomodação ou polos.

        Parameters
        ----------
        settling_threshold : float, default 0.02
            Limiar para tempo de acomodação.

        Returns
        -------
        float
            Tempo de simulação estimado em segundos.
        """
        tf = self.model.tf
        
        # Primeiro, tenta usar step_info
        try:
            info = control.step_info(
                tf,
                T=self.model.get_simulation_time(),
                SettlingTimeThreshold=settling_threshold,
            )
            settling_time = info.get("SettlingTime", None)

            if settling_time is not None and settling_time > 0:
                return float(settling_time) * 3.0
        except Exception:
            pass

        # Fallback: análise dos polos
        try:
            poles = tf.poles()
            real_parts = np.real(poles)

            # Verificar estabilidade
            if np.any(real_parts > 0):
                # Sistema instável - limitar simulação
                return 5.0

            # Usar polo dominante para estimar
            negative_poles = real_parts[real_parts < 0]
            if len(negative_poles) > 0:
                dominant_pole = np.max(negative_poles)
                time_constant = -1.0 / dominant_pole
                return float(time_constant * 15)  # ~5 constantes de tempo
        except Exception:
            pass

        return 10.0  # Fallback final

    def get_impulse_response_data(
        self,
        simulation_time: Optional[float] = None,
        qt_points: int = 1000,
        settling_time_threshold: float = 0.02,
    ) -> Dict[str, Any]:
        """
        Retorna a resposta impulsiva em formato serializável + métricas básicas.

        Este método é útil para exportação de dados ou integração com APIs REST.

        Parameters
        ----------
        simulation_time : float, optional
            Ver :meth:`compute`.
        qt_points : int, default 1000
            Ver :meth:`compute`.
        settling_time_threshold : float, default 0.02
            Ver :meth:`compute`.

        Returns
        -------
        dict
            Dicionário com chaves:

            - ``time`` : list of float
                Lista de tempos.
            - ``response`` : list of float
                Lista da resposta impulsiva.
            - ``metrics`` : dict
                Métricas básicas: ``Peak``, ``PeakTime``, ``Energy``, ``Area``.

        See Also
        --------
        get_metrics : Retorna todas as 20 métricas temporais.

        Examples
        --------
        >>> import auto_control_tools as act
        >>> m = act.Model(([1], [1, 1]))
        >>> ir = act.Impulse(m)
        >>> data = ir.get_impulse_response_data()
        >>> print(data['metrics'])
        {'Peak': 1.0, 'PeakTime': 0.0, 'Energy': 0.5, 'Area': 1.0}
        """
        serie = self.compute(
            simulation_time=simulation_time,
            qt_points=qt_points,
            settling_time_threshold=settling_time_threshold,
        )

        # Métricas básicas (compatibilidade)
        idx_peak = int(np.argmax(np.abs(serie.values)))
        peak = float(serie.values[idx_peak])
        peak_time = float(serie.index[idx_peak])
        energy = float(np.trapz(serie.values**2, serie.index))
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

    def get_metrics(self, settling_band: float = 0.02) -> Dict[str, Any]:
        """
        Calcula TODAS as características temporais da resposta ao impulso.

        Fornece 20 métricas completas para análise detalhada do comportamento
        dinâmico do sistema.

        Parameters
        ----------
        settling_band : float, default 0.02
            Banda de acomodação (padrão 2% = 0.02).

        Returns
        -------
        dict
            Dicionário completo com 20 métricas temporais:

            **BÁSICAS (5):**
                - ``Peak`` : float
                    Valor do pico (máximo absoluto).
                - ``PeakTime`` : float
                    Tempo do pico [s].
                - ``Energy`` : float
                    Energia total :math:`\\int h^2(t)\\,dt`.
                - ``Area`` : float
                    Área sob a curva (ganho DC).
                - ``Duration`` : float
                    Duração efetiva (até 2% do pico) [s].

            **SOBRESSINAIS E AMPLITUDES (6):**
                - ``Overshoot`` : float
                    Sobressinal em % (relativo ao ganho DC).
                - ``Undershoot`` : float
                    Pico negativo (se existir).
                - ``UndershootTime`` : float
                    Tempo do undershoot [s].
                - ``MaxValue`` : float
                    Valor máximo.
                - ``MinValue`` : float
                    Valor mínimo.
                - ``PeakToPeak`` : float
                    Amplitude pico a pico.

            **TEMPOS CARACTERÍSTICOS (2):**
                - ``RiseTime`` : float
                    Tempo de subida (10% a 90%) [s].
                - ``SettlingTime`` : float
                    Tempo de acomodação (banda ±2%) [s].

            **OSCILAÇÕES (4):**
                - ``NumOscillations`` : int
                    Número de oscilações.
                - ``OscillationPeriod`` : float
                    Período de oscilação [s].
                - ``OscillationFrequency`` : float
                    Frequência de oscilação [Hz].
                - ``DampedFrequency`` : float
                    Frequência angular amortecida [rad/s].

            **TAXAS DE VARIAÇÃO (3):**
                - ``MaxRiseRate`` : float
                    Taxa máxima de subida :math:`dh/dt`.
                - ``MaxFallRate`` : float
                    Taxa máxima de descida :math:`dh/dt`.
                - ``SlewRate`` : float
                    Taxa de variação no pico.

        See Also
        --------
        summary : Imprime relatório formatado com todas as métricas.
        get_impulse_response_data : Retorna métricas básicas apenas.

        Notes
        -----
        Para sistemas de segunda ordem subamortecidos, as métricas de oscilação
        são especialmente úteis. A frequência amortecida :math:`\\omega_d` está
        relacionada à frequência natural :math:`\\omega_n` por:

        .. math::

            \\omega_d = \\omega_n \\sqrt{1 - \\zeta^2}

        onde :math:`\\zeta` é o fator de amortecimento.

        Examples
        --------
        >>> import auto_control_tools as act
        >>> num, den = [25], [1, 5, 25]  # Sistema subamortecido
        >>> m = act.Model((num, den))
        >>> ir = act.Impulse(m)
        >>> metrics = ir.get_metrics()
        >>> print(f"Pico: {metrics['Peak']:.4f}")
        >>> print(f"Tempo de subida: {metrics['RiseTime']:.4f} s")
        >>> print(f"Oscilações: {metrics['NumOscillations']}")
        """
        if self._response_data is None:
            self.compute()

        h = self._response_data.values
        t = self._response_data.index

        metrics = {}

        # ==========================================
        # BÁSICAS
        # ==========================================
        idx_peak = int(np.argmax(np.abs(h)))
        metrics["Peak"] = float(h[idx_peak])
        metrics["PeakTime"] = float(t[idx_peak])
        metrics["Energy"] = float(np.trapz(h**2, t))
        metrics["Area"] = float(np.trapz(h, t))

        # Duração efetiva (até 2% do pico)
        threshold = abs(metrics["Peak"]) * 0.02
        significant_indices = np.where(np.abs(h) > threshold)[0]
        metrics["Duration"] = (
            float(t[significant_indices[-1]]) if len(significant_indices) > 0 else 0.0
        )

        # ==========================================
        # CARACTERÍSTICAS TEMPORAIS AVANÇADAS
        # ==========================================

        # Valor final estimado (ganho DC = área)
        valor_final = metrics["Area"]

        # Sobressinal (Overshoot) em %
        if abs(valor_final) > 1e-10:
            metrics["Overshoot"] = float(
                ((metrics["Peak"] - valor_final) / abs(valor_final)) * 100
            )
        else:
            metrics["Overshoot"] = 0.0

        # Máximo e Mínimo absolutos
        metrics["MaxValue"] = float(np.max(h))
        metrics["MinValue"] = float(np.min(h))
        metrics["PeakToPeak"] = metrics["MaxValue"] - metrics["MinValue"]

        # Undershoot (pico negativo mais significativo)
        picos_negativos = []
        for i in range(1, len(h) - 1):
            if h[i] < h[i - 1] and h[i] < h[i + 1] and h[i] < -threshold:
                picos_negativos.append((t[i], h[i]))

        if picos_negativos:
            undershoot = min(picos_negativos, key=lambda x: x[1])
            metrics["Undershoot"] = float(undershoot[1])
            metrics["UndershootTime"] = float(undershoot[0])
        else:
            metrics["Undershoot"] = 0.0
            metrics["UndershootTime"] = 0.0

        # Tempo de subida (10% a 90% do pico)
        try:
            nivel_10 = 0.1 * abs(metrics["Peak"])
            nivel_90 = 0.9 * abs(metrics["Peak"])
            idx_10 = np.where(np.abs(h) >= nivel_10)[0][0]
            idx_90 = np.where(np.abs(h) >= nivel_90)[0][0]
            metrics["RiseTime"] = float(t[idx_90] - t[idx_10])
        except (IndexError, ValueError):
            metrics["RiseTime"] = 0.0

        # Tempo de acomodação (settling time)
        banda = (
            settling_band * abs(valor_final)
            if abs(valor_final) > 1e-10
            else settling_band
        )
        try:
            fora_banda = np.where(np.abs(h - valor_final) > banda)[0]
            if len(fora_banda) > 0:
                metrics["SettlingTime"] = float(t[fora_banda[-1]])
            else:
                metrics["SettlingTime"] = float(t[0])
        except (IndexError, ValueError):
            metrics["SettlingTime"] = metrics["Duration"]

        # Número de oscilações
        cruzamentos_zero = np.where(np.diff(np.sign(h)))[0]
        metrics["NumOscillations"] = int(len(cruzamentos_zero) // 2)

        # Período e frequência de oscilação
        picos_positivos = []
        for i in range(1, len(h) - 1):
            if h[i] > h[i - 1] and h[i] > h[i + 1] and h[i] > threshold:
                picos_positivos.append((t[i], h[i]))

        if len(picos_positivos) >= 2:
            periodo = picos_positivos[1][0] - picos_positivos[0][0]
            metrics["OscillationPeriod"] = float(periodo)
            metrics["OscillationFrequency"] = float(1.0 / periodo)  # Hz
            metrics["DampedFrequency"] = float(2 * np.pi / periodo)  # rad/s
        else:
            metrics["OscillationPeriod"] = 0.0
            metrics["OscillationFrequency"] = 0.0
            metrics["DampedFrequency"] = 0.0

        # Taxas de variação
        if len(h) > 1:
            dh_dt = np.gradient(h, t)
            metrics["MaxRiseRate"] = float(np.max(dh_dt))
            metrics["MaxFallRate"] = float(np.min(dh_dt))

            if 0 < idx_peak < len(dh_dt):
                metrics["SlewRate"] = float(dh_dt[idx_peak])
            else:
                metrics["SlewRate"] = 0.0
        else:
            metrics["MaxRiseRate"] = 0.0
            metrics["MaxFallRate"] = 0.0
            metrics["SlewRate"] = 0.0

        return metrics

    def summary(self, settling_band: float = 0.02) -> None:
        """
        Imprime relatório completo com TODAS as características temporais.

        Gera um relatório formatado no terminal com informações do sistema,
        polos, análise de estabilidade e todas as 20 métricas temporais.

        Parameters
        ----------
        settling_band : float, default 0.02
            Banda de acomodação (padrão 2%).

        See Also
        --------
        get_metrics : Retorna métricas como dicionário.

        Examples
        --------
        >>> import auto_control_tools as act
        >>> num, den = [25], [1, 5, 25]
        >>> m = act.Model((num, den))
        >>> ir = act.Impulse(m, name="Subamortecido")
        >>> ir.summary()
        ================================================================================
        RELATÓRIO COMPLETO: Subamortecido
        ================================================================================
        ...
        """
        if self._response_data is None:
            self.compute()

        metrics = self.get_metrics(settling_band=settling_band)
        poles = self.model.tf.poles()

        print("=" * 80)
        print(f"RELATÓRIO COMPLETO: {self.name}")
        print("=" * 80)
        print(f"\nSistema: {self.model.tf}")
        print(f"\nPolos:")
        for i, pole in enumerate(poles, 1):
            if abs(pole.imag) < 1e-10:
                print(f"  p{i} = {pole.real:.4f}")
            else:
                print(f"  p{i} = {pole.real:.4f} {pole.imag:+.4f}j")

        print(f"\nEstabilidade:")
        if np.all(np.real(poles) < 0):
            print("  ✓ Sistema ESTÁVEL")
        elif np.any(np.real(poles) > 0):
            print("  ✗ Sistema INSTÁVEL")
        else:
            print("  ⚠ Sistema MARGINALMENTE ESTÁVEL")

        print("\n" + "=" * 80)
        print("CARACTERÍSTICAS TEMPORAIS COMPLETAS")
        print("=" * 80)

        print("\n▸ MÉTRICAS BÁSICAS:")
        print(f"  • Pico (máx. absoluto):        {metrics['Peak']:>12.6f}")
        print(f"  • Tempo do pico:               {metrics['PeakTime']:>12.6f} s")
        print(f"  • Energia total:               {metrics['Energy']:>12.6f}")
        print(f"  • Área (Ganho DC):             {metrics['Area']:>12.6f}")
        print(f"  • Duração efetiva:             {metrics['Duration']:>12.6f} s")

        print("\n▸ SOBRESSINAIS E AMPLITUDES:")
        print(f"  • Sobressinal (Overshoot):     {metrics['Overshoot']:>12.2f} %")
        print(f"  • Undershoot:                  {metrics['Undershoot']:>12.6f}")
        print(f"  • Tempo do undershoot:         {metrics['UndershootTime']:>12.6f} s")
        print(f"  • Valor máximo:                {metrics['MaxValue']:>12.6f}")
        print(f"  • Valor mínimo:                {metrics['MinValue']:>12.6f}")
        print(f"  • Amplitude pico-a-pico:       {metrics['PeakToPeak']:>12.6f}")

        print("\n▸ TEMPOS CARACTERÍSTICOS:")
        print(f"  • Tempo de subida (10%-90%):   {metrics['RiseTime']:>12.6f} s")
        print(
            f"  • Tempo de acomodação ({int(settling_band*100)}%):    {metrics['SettlingTime']:>12.6f} s"
        )

        print("\n▸ OSCILAÇÕES:")
        print(f"  • Número de oscilações:        {metrics['NumOscillations']:>12d}")
        if metrics["OscillationPeriod"] > 0:
            print(
                f"  • Período de oscilação:        {metrics['OscillationPeriod']:>12.6f} s"
            )
            print(
                f"  • Frequência de oscilação:     {metrics['OscillationFrequency']:>12.6f} Hz"
            )
            print(
                f"  • Freq. angular amortecida:    {metrics['DampedFrequency']:>12.6f} rad/s"
            )
        else:
            print(f"  • Sistema não oscilatório")

        print("\n▸ TAXAS DE VARIAÇÃO:")
        print(f"  • Taxa máxima de subida:       {metrics['MaxRiseRate']:>12.6f}")
        print(f"  • Taxa máxima de descida:      {metrics['MaxFallRate']:>12.6f}")
        print(f"  • Slew rate (no pico):         {metrics['SlewRate']:>12.6f}")

        print("=" * 80)

    def compare_analytical(
        self,
        analytical_func: Callable[[np.ndarray], np.ndarray],
        simulation_time: Optional[float] = None,
        qt_points: int = 1000,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
        """
        Compara resposta numérica com solução analítica.

        Útil para validação de modelos ou para fins didáticos, comparando
        a resposta calculada numericamente com a solução analítica conhecida.

        Parameters
        ----------
        analytical_func : callable
            Função que recebe vetor de tempo (np.ndarray) e retorna a resposta
            analítica (np.ndarray).
        simulation_time : float, optional
            Tempo de simulação.
        qt_points : int, default 1000
            Número de pontos.
        save_path : str, optional
            Caminho para salvar figura (formato .png, .pdf, etc.).

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figura criada.
        ax1 : matplotlib.axes.Axes
            Eixo da comparação.
        ax2 : matplotlib.axes.Axes
            Eixo do erro.

        Examples
        --------
        >>> import auto_control_tools as act
        >>> import numpy as np
        >>> # Sistema de primeira ordem: h(t) = e^(-t)
        >>> m = act.Model(([1], [1, 1]))
        >>> ir = act.Impulse(m)
        >>> analytical = lambda t: np.exp(-t)
        >>> fig, ax1, ax2 = ir.compare_analytical(analytical)
        """
        # Calcular resposta numérica
        series = self.compute(simulation_time, qt_points)
        time = series.index

        # Calcular resposta analítica
        analytical = analytical_func(time)

        # Calcular erro
        error = np.abs(series.values - analytical)
        max_error = np.max(error)
        mean_error = np.mean(error)

        # Criar figura com 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot 1: Comparação
        ax1.plot(time, series.values, "b-", linewidth=2, label="Numérico")
        ax1.plot(time, analytical, "r--", linewidth=2, label="Analítico")
        ax1.set_xlabel("Tempo [s]", fontsize=11)
        ax1.set_ylabel("Amplitude h(t)", fontsize=11)
        ax1.set_title(
            f"Comparação: Numérico vs Analítico - {self.name}",
            fontsize=13,
            fontweight="bold",
        )
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="best")
        ax1.axhline(0, color="black", linewidth=0.8)

        # Plot 2: Erro
        ax2.plot(time, error, "g-", linewidth=2)
        ax2.set_xlabel("Tempo [s]", fontsize=11)
        ax2.set_ylabel("Erro Absoluto", fontsize=11)
        ax2.set_title(
            f"Erro: Máx={max_error:.2e}, Médio={mean_error:.2e}", fontsize=12
        )
        ax2.grid(True, alpha=0.3)
        ax2.axhline(0, color="black", linewidth=0.8)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"✓ Comparação salva em: {save_path}")

        return fig, ax1, ax2


class ImpulseView:
    """
    Visualização da resposta impulsiva vinculada a :class:`Impulse`.

    Esta classe fornece métodos para plotagem da resposta ao impulso com
    diferentes níveis de detalhamento e personalização.

    Parameters
    ----------
    impulse : Impulse
        Instância de Impulse à qual esta view está vinculada.

    Attributes
    ----------
    impulse : Impulse
        Referência ao objeto Impulse.

    See Also
    --------
    Impulse : Classe principal de análise.
    PlotUtils : Utilitários de plotagem da biblioteca.

    Examples
    --------
    >>> import auto_control_tools as act
    >>> m = act.Model(([1], [1, 1]))
    >>> ir = act.Impulse(m)
    >>> ir.view.plot()  # Plot simples
    >>> ir.view.plot(show_metrics=True)  # Plot com métricas
    """

    def __init__(self, impulse: Impulse):
        self.impulse = impulse

    def plot(
        self,
        simulation_time: Optional[float] = None,
        qt_points: int = 1000,
        settling_time_threshold: float = 0.02,
        legend_suffix: str = "",
        show_metrics: bool = False,
        save_path: Optional[str] = None,
        figsize: Tuple[float, float] = (12, 7),
    ) -> Optional[Tuple[plt.Figure, plt.Axes]]:
        """
        Plota a resposta ao impulso do :term:`Modelo`.

        Parameters
        ----------
        simulation_time : float, optional
            Ver :meth:`Impulse.compute`.
        qt_points : int, default 1000
            Ver :meth:`Impulse.compute`.
        settling_time_threshold : float, default 0.02
            Ver :meth:`Impulse.compute`.
        legend_suffix : str, optional
            Sufixo para legenda (útil ao comparar múltiplos sistemas).
        show_metrics : bool, default False
            Se True, exibe métricas principais no gráfico e adiciona
            marcadores visuais (pico, settling time, etc.).
        save_path : str, optional
            Caminho para salvar a figura.
        figsize : tuple of float, default (12, 7)
            Tamanho da figura (largura, altura).

        Returns
        -------
        tuple of (Figure, Axes) or None
            Se ``show_metrics=True``, retorna figura e eixos.
            Caso contrário, retorna None (modo compatibilidade).

        See Also
        --------
        Impulse.get_metrics : Métricas exibidas no gráfico.
        matplotlib.pyplot.savefig : Salvar figura.

        Examples
        --------
        Plot simples (compatibilidade com versão anterior):

        >>> import auto_control_tools as act
        >>> m = act.Model(([1], [1, 1]))
        >>> ir = act.Impulse(m)
        >>> ir.view.plot()

        Plot avançado com métricas:

        >>> ir.view.plot(show_metrics=True, save_path="impulse.png")
        """
        serie = self.impulse.compute(
            simulation_time=simulation_time,
            qt_points=qt_points,
            settling_time_threshold=settling_time_threshold,
        )

        if show_metrics:
            # Modo avançado com métricas
            metrics = self.impulse.get_metrics()

            # Criar figura
            fig, ax = plt.subplots(figsize=figsize)

            # Plotar resposta
            ax.plot(serie.index, serie.values, "b-", linewidth=2, label="h(t)")

            # Marcar pico
            ax.plot(
                metrics["PeakTime"],
                metrics["Peak"],
                "ro",
                markersize=10,
                label=f"Pico: {metrics['Peak']:.4f}",
            )

            # Marcar undershoot se existir
            if metrics["Undershoot"] != 0:
                ax.plot(
                    metrics["UndershootTime"],
                    metrics["Undershoot"],
                    "mo",
                    markersize=10,
                    label=f"Undershoot: {metrics['Undershoot']:.4f}",
                )

            # Linha de tempo de acomodação
            if metrics["SettlingTime"] > 0:
                ax.axvline(
                    metrics["SettlingTime"],
                    color="green",
                    linestyle="--",
                    alpha=0.7,
                    label=f"Settling: {metrics['SettlingTime']:.3f}s",
                )

            # Banda de acomodação (±2%)
            valor_final = metrics["Area"]
            banda = 0.02 * abs(valor_final)
            if abs(valor_final) > 1e-10:
                ax.axhline(valor_final + banda, color="orange", linestyle=":", alpha=0.5)
                ax.axhline(valor_final - banda, color="orange", linestyle=":", alpha=0.5)
                ax.fill_between(
                    serie.index,
                    valor_final - banda,
                    valor_final + banda,
                    color="orange",
                    alpha=0.1,
                    label="Banda ±2%",
                )

            # Configurar gráfico
            time_label = (
                "Tempo [s]"
                if self.impulse.model.tf.dt is None
                else "Passos [k]"
            )
            ax.set_xlabel(time_label, fontsize=12)
            ax.set_ylabel("Amplitude h(t)", fontsize=12)
            ax.set_title(
                f"Resposta ao Impulso - {self.impulse.name}",
                fontsize=14,
                fontweight="bold",
            )
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color="black", linewidth=0.8)

            # Adicionar métricas como texto
            metrics_text = (
                f"Peak: {metrics['Peak']:.4f} @ t={metrics['PeakTime']:.3f}s\n"
                f"Rise Time: {metrics['RiseTime']:.4f}s\n"
                f"Settling Time: {metrics['SettlingTime']:.4f}s\n"
                f"Overshoot: {metrics['Overshoot']:.2f}%\n"
                f"Oscillations: {metrics['NumOscillations']}\n"
                f"Energy: {metrics['Energy']:.4f}"
            )
            ax.text(
                0.02,
                0.98,
                metrics_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                family="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

            ax.legend(loc="best", fontsize=10)
            plt.tight_layout()

            if save_path:
                fig.savefig(save_path, dpi=200, bbox_inches="tight")
                print(f"✓ Gráfico salvo em: {save_path}")

            plt.show()
            return fig, ax

        else:
            # Modo simples (compatibilidade com versão anterior)
            legend_kwargs = PlotUtils._setup_plot_env()
            plt.plot(
                serie.index,
                serie.values,
                label=f"impulse{(' ' + legend_suffix) if legend_suffix else ''}",
            )
            plt.xlabel(
                "Tempo [s]" if self.impulse.model.tf.dt is None else "Passos [k]"
            )
            plt.ylabel("Amplitude")
            plt.title("Resposta ao Impulso")
            plt.grid(True)
            plt.legend(**legend_kwargs)
            plt.show()
            return None


# ------------------------------------------------------------------
#  API funcional (atalhos para uso em notebooks e scripts)
# ------------------------------------------------------------------
def impulse_response(
    model: Model,
    simulation_time: Optional[float] = None,
    qt_points: int = 1000,
    settling_time_threshold: float = 0.02,
) -> pd.Series:
    """
    Atalho funcional para :meth:`Impulse.compute`.

    Parameters
    ----------
    model : Model
        Modelo do sistema.
    simulation_time : float, optional
        Tempo de simulação.
    qt_points : int, default 1000
        Número de pontos.
    settling_time_threshold : float, default 0.02
        Limiar de settling.

    Returns
    -------
    pandas.Series
        Série com a resposta ao impulso.

    See Also
    --------
    Impulse.compute : Método equivalente na classe.

    Examples
    --------
    >>> import auto_control_tools as act
    >>> m = act.Model(([1], [1, 1]))
    >>> serie = act.impulse_response(m)
    >>> serie.plot()
    """
    return Impulse(model).compute(
        simulation_time=simulation_time,
        qt_points=qt_points,
        settling_time_threshold=settling_time_threshold,
    )


def impulse_analysis(
    model: Model,
    name: str = "Sistema",
    analytical_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    simulation_time: Optional[float] = None,
    show: bool = True,
) -> Impulse:
    """
    Função de conveniência para análise rápida de resposta ao impulso.

    Realiza análise completa: imprime relatório, calcula métricas e
    opcionalmente plota o gráfico.

    Parameters
    ----------
    model : Model
        Modelo do sistema a analisar.
    name : str, default "Sistema"
        Nome do sistema.
    analytical_func : callable, optional
        Função analítica para comparação. Se fornecida, plota comparação
        numérico vs analítico.
    simulation_time : float, optional
        Tempo de simulação.
    show : bool, default True
        Se True, mostra gráficos.

    Returns
    -------
    Impulse
        Objeto analisador configurado com 20 métricas disponíveis.

    See Also
    --------
    Impulse : Classe principal.
    Impulse.summary : Relatório detalhado.
    Impulse.compare_analytical : Comparação com solução analítica.

    Examples
    --------
    >>> import auto_control_tools as act
    >>> m = act.Model(([25], [1, 5, 25]))
    >>> analyzer = act.impulse_analysis(m, name="Subamortecido")
    >>> metrics = analyzer.get_metrics()
    """
    analyzer = Impulse(model, name=name)
    analyzer.summary()

    if analytical_func is not None:
        analyzer.compare_analytical(analytical_func, simulation_time)
    else:
        analyzer.view.plot(simulation_time, show_metrics=True)

    if show:
        plt.show()

    return analyzer