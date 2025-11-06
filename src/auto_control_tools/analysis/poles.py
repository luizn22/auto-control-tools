# src/auto_control_tools/analysis/poles.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from control import TransferFunction, tf, pole, zero  # python-control

ArrayLike = Union[Iterable[float], np.ndarray]
NumDen = Tuple[ArrayLike, ArrayLike]


@dataclass(frozen=True)
class PoleZeroPlotConfig:
    """Configurações visuais para o mapa de polos e zeros."""
    title: str = "Mapa de Polos e Zeros"
    pole_marker: str = "x"
    zero_marker: str = "o"
    pole_markersize: int = 10
    zero_markersize: int = 8
    grid: bool = True
    axis_equal: bool = True
    show_axes_cross: bool = True
    unit_circle: bool = True  # útil para sistemas discretos


class PoleZeroAnalyzer:
    """
    Analisa polos e zeros de sistemas SISO e produz o gráfico no plano complexo.

    Parameters
    ----------
    system : control.TransferFunction | tuple[num, den]
        Pode ser um objeto `TransferFunction` do python-control ou um par (num, den)
        com os coeficientes dos polinômios.
    name : str, optional
        Rótulo do sistema para usar em títulos/legendas.
    """

    def __init__(
        self,
        system: Union[TransferFunction, NumDen],
        name: Optional[str] = None,
    ) -> None:
        self._sys: TransferFunction = self._ensure_tf(system)
        self.name: str = name or "G(s)" if self._sys.dt is None else "G(z)"
        self._zeros: Optional[np.ndarray] = None
        self._poles: Optional[np.ndarray] = None
        self._validate_siso()

    # ------------------------- propriedades computadas -------------------------

    @property
    def zeros(self) -> np.ndarray:
        """Retorna os zeros (array de complexos)."""
        if self._zeros is None:
            self._zeros = np.asarray(zero(self._sys))
        return self._zeros

    @property
    def poles(self) -> np.ndarray:
        """Retorna os polos (array de complexos)."""
        if self._poles is None:
            self._poles = np.asarray(pole(self._sys))
        return self._poles

    @property
    def is_discrete(self) -> bool:
        """True se o sistema for discreto (dt não nulo)."""
        return self._sys.dt not in (None, 0, False)

    # ------------------------------ API principal ------------------------------

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        config: Optional[PoleZeroPlotConfig] = None,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        show: bool = False,
        savepath: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plota o mapa de polos (X) e zeros (O).

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Eixo existente para desenhar. Se None, cria figura nova.
        config : PoleZeroPlotConfig, optional
            Configurações visuais do gráfico.
        xlim, ylim : tuple[float, float], optional
            Limites dos eixos real e imaginário.
        show : bool, default False
            Se True, chama plt.show() ao final.
        savepath : str, optional
            Se informado, salva a figura no caminho especificado.

        Returns
        -------
        fig, ax : tuple[Figure, Axes]
            Figura e eixo do gráfico gerado.
        """
        cfg = config or PoleZeroPlotConfig()

        if ax is None:
            fig, ax = plt.subplots(figsize=(6.5, 5.0))
        else:
            fig = ax.figure  # type: ignore[assignment]

        # Polos e zeros
        z = self.zeros
        p = self.poles

        # Dispersão: zeros com "O", polos com "X"
        if z.size > 0:
            ax.plot(np.real(z), np.imag(z), cfg.zero_marker, markersize=cfg.zero_markersize, label="Zeros")
        if p.size > 0:
            ax.plot(np.real(p), np.imag(p), cfg.pole_marker, markersize=cfg.pole_markersize, label="Polos")

        # Eixos e grade
        ax.set_xlabel("Parte Real")
        ax.set_ylabel("Parte Imaginária")
        title = f"{cfg.title} — {self.name}"
        ax.set_title(title)
        if cfg.grid:
            ax.grid(True, linestyle="--", alpha=0.5)

        if cfg.axis_equal:
            ax.set_aspect("equal", adjustable="datalim")

        # Cruz dos eixos
        if cfg.show_axes_cross:
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.axvline(0.0, color="black", linewidth=0.8)

        # Círculo unitário para discretos (referência de estabilidade)
        if cfg.unit_circle and self.is_discrete:
            theta = np.linspace(0, 2 * np.pi, 512)
            ax.plot(np.cos(theta), np.sin(theta), linestyle=":", linewidth=1.0)

        # Limites automáticos, com margem
        self._auto_limits(ax, xlim, ylim)

        # Legenda se houver elementos
        if (z.size + p.size) > 0:
            ax.legend(loc="best")

        if savepath:
            fig.savefig(savepath, bbox_inches="tight", dpi=200)

        if show:
            plt.show()

        return fig, ax

    def summary(self) -> dict:
        """
        Retorna um resumo com polos/zeros e metadados do sistema.

        Returns
        -------
        dict
            Dicionário com arrays de polos e zeros (complexos), ordem e dt.
        """
        return {
            "name": self.name,
            "zeros": self.zeros.copy(),
            "poles": self.poles.copy(),
            "order_num": self._poly_order(self._sys.num[0][0]),
            "order_den": self._poly_order(self._sys.den[0][0]),
            "is_discrete": self.is_discrete,
            "dt": self._sys.dt,
        }

    # ------------------------------ utilitários --------------------------------

    @staticmethod
    def _ensure_tf(system: Union[TransferFunction, NumDen]) -> TransferFunction:
        """Garante que o sistema seja um `TransferFunction` do python-control."""
        if isinstance(system, TransferFunction):
            return system

        if (
            isinstance(system, tuple)
            and len(system) == 2
        ):
            num, den = system
            return tf(num, den)

        raise TypeError(
            "Parâmetro `system` deve ser control.TransferFunction "
            "ou um par (num, den) de coeficientes."
        )

    def _validate_siso(self) -> None:
        """Garante que o sistema seja SISO; lança ValueError caso contrário."""
        if getattr(self._sys, "inputs", 1) != 1 or getattr(self._sys, "outputs", 1) != 1:
            raise ValueError("Apenas sistemas SISO são suportados neste analisador.")

    @staticmethod
    def _poly_order(poly: ArrayLike) -> int:
        """Ordem do polinômio (removendo zeros à esquerda)."""
        c = np.trim_zeros(np.asarray(poly, dtype=float), trim="f")
        return (len(c) - 1) if len(c) > 0 else 0

    @staticmethod
    def _auto_limits(
        ax: plt.Axes,
        xlim: Optional[Tuple[float, float]],
        ylim: Optional[Tuple[float, float]],
        pad: float = 0.15,
    ) -> None:
        """Define limites com pequena margem, a menos que o usuário forneça."""
        if xlim is None or ylim is None:
            # pega limites atuais
            ax.relim()
            ax.autoscale()
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()

            # margem
            xr = xmax - xmin
            yr = ymax - ymin
            if xr == 0:
                xr = 1.0
            if yr == 0:
                yr = 1.0

            if xlim is None:
                ax.set_xlim(xmin - pad * xr, xmax + pad * xr)
            else:
                ax.set_xlim(*xlim)

            if ylim is None:
                ax.set_ylim(ymin - pad * yr, ymax + pad * yr)
            else:
                ax.set_ylim(*ylim)
        else:
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)


# --------- interface funcional “amigável”, mantendo padrão da biblioteca ---------

def pzmap(
    system: Union[TransferFunction, NumDen],
    name: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    config: Optional[PoleZeroPlotConfig] = None,
    **plot_kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Atalho funcional para gerar o mapa de polos e zeros.

    Examples
    --------
    >>> from control import tf
    >>> G = tf([1], [1, 2, 1])  # 1/(s+1)^2
    >>> fig, ax = pzmap(G, name="Exemplo")
    """
    analyzer = PoleZeroAnalyzer(system, name=name)
    return analyzer.plot(ax=ax, config=config, **plot_kwargs)
