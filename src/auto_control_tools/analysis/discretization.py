# src/auto_control_tools/analysis/discretization.py
"""
Discretização e análise de resposta (ZOH, Tustin, FOH, Forward Euler).

- Entrada: sistema LTI contínuo (TransferFunction ou StateSpace, control ou scipy)
- Saída: sistema discreto (A, B, C, D, dt) + plot da resposta (passo/impulso)

FOH e ZOH são implementados com exponencial de matrizes (Van Loan),
sem depender de inversa de A (robusto inclusive para A singular).
Tustin e Forward Euler usam cont2discrete (SciPy) por clareza e reprodutibilidade.

Autor: Matheus Cheim
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Tuple, Union, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm
from scipy.signal import (
    cont2discrete, StateSpace as SciPySS, TransferFunction as SciPyTF,
    lti as SciPyLTI, dlti as SciPyDLTI, dstep, dimpulse, step, impulse
)

try:
    # Opcional: suporte a `python-control` se estiver instalado
    import control  # type: ignore
    _HAS_CONTROL = True
except Exception:
    control = None
    _HAS_CONTROL = False

LTIAny = Union[SciPySS, SciPyTF, SciPyLTI, "control.StateSpace", "control.TransferFunction"]


@dataclass
class DiscretizationResult:
    Ad: NDArray[np.float64]
    Bd: NDArray[np.float64]
    Cd: NDArray[np.float64]
    Dd: NDArray[np.float64]
    dt: float
    # FOH: quando relevante, soma dos ganhos para degrau (u_k = u_{k+1})
    Bd_foh_u_k: Optional[NDArray[np.float64]] = None
    Bd_foh_u_k1: Optional[NDArray[np.float64]] = None


class _Conversions:
    @staticmethod
    def _to_ss(sys: LTIAny) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """Normaliza para espaço de estados contínuo (A,B,C,D)."""
        # python-control
        if _HAS_CONTROL:
            if isinstance(sys, control.TransferFunction):
                ss = control.ss(sys)
                A, B, C, D = control.ssdata(ss)
                return np.asarray(A, float), np.asarray(B, float), np.asarray(C, float), np.asarray(D, float)
            if isinstance(sys, control.StateSpace):
                A, B, C, D = control.ssdata(sys)
                return np.asarray(A, float), np.asarray(B, float), np.asarray(C, float), np.asarray(D, float)

        # SciPy
        if isinstance(sys, SciPyTF):
            A, B, C, D = SciPyTF.to_ss(sys).A, SciPyTF.to_ss(sys).B, SciPyTF.to_ss(sys).C, SciPyTF.to_ss(sys).D
            return np.asarray(A, float), np.asarray(B, float), np.asarray(C, float), np.asarray(D, float)
        if isinstance(sys, SciPySS):
            return np.asarray(sys.A, float), np.asarray(sys.B, float), np.asarray(sys.C, float), np.asarray(sys.D, float)
        if isinstance(sys, SciPyLTI):
            # lti contínuo (SciPy), traz como SS
            ss: SciPySS = sys.to_ss()
            return np.asarray(ss.A, float), np.asarray(ss.B, float), np.asarray(ss.C, float), np.asarray(ss.D, float)

        # Tuplas (num, den) — tenta interpretar como TF (SISO)
        if isinstance(sys, tuple) and len(sys) == 2:
            num, den = sys
            tf = SciPyTF(num, den)
            ss = tf.to_ss()
            return np.asarray(ss.A, float), np.asarray(ss.B, float), np.asarray(ss.C, float), np.asarray(ss.D, float)

        raise TypeError("Tipo de sistema não suportado. Use TransferFunction/StateSpace (control ou scipy).")

    @staticmethod
    def _to_dlti(res: DiscretizationResult) -> SciPyDLTI:
        return SciPyDLTI(res.Ad, res.Bd, res.Cd, res.Dd, dt=res.dt)


class _VanLoan:
    @staticmethod
    def phi1(A: NDArray, T: float) -> Tuple[NDArray, NDArray]:
        """
        Retorna (Ad, Φ1) onde:
          Ad = e^{A T}
          Φ1 = ∫_0^T e^{A τ} dτ
        via exponencial de bloco (sem inversa).
        """
        n = A.shape[0]
        Z = np.zeros((n, n))
        I = np.eye(n)
        # exp( [A I; 0 0] T ) = [Ad Φ1; 0 I]
        M = np.block([[A, I],
                      [Z, Z]])
        expM = expm(M * T)
        Ad = expM[:n, :n]
        Phi1 = expM[:n, n:]
        return Ad, Phi1

    @staticmethod
    def phi1_phi2(A: NDArray, T: float) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Retorna (Ad, Φ1, Φ2) onde:
          Ad = e^{A T}
          Φ1 = ∫_0^T e^{A τ} dτ
          Φ2 = ∫_0^T τ e^{A τ} dτ
        via exponencial de bloco 3x3 (Van Loan).
        """
        n = A.shape[0]
        Z = np.zeros((n, n))
        I = np.eye(n)
        # exp( [[A I 0],[0 A I],[0 0 0]] T ) = [[Ad Φ1 Φ2],[0 Ad Φ1],[0 0 I]]
        M = np.block([
            [A, I, Z],
            [Z, A, I],
            [Z, Z, Z]
        ])
        expM = expm(M * T)
        Ad = expM[:n, :n]
        Phi1 = expM[:n, n:2*n]
        Phi2 = expM[:n, 2*n:3*n]
        return Ad, Phi1, Phi2


class Discretizer:
    """
    Discretizador de sistemas contínuos com múltiplos métodos e plot de resposta.

    Exemplo rápido:
        from auto_control_tools.analysis.discretization import Discretizer
        Gs = control.tf([1],[1, 1, 1])  # exemplo; pode usar scipy também
        disc = Discretizer(Gs, Ts=0.05)
        result = disc.discretize(method="zoh")
        disc.plot_response(result, kind="step", Tfinal=3.0, compare_continuous=True)
    """

    def __init__(self, sys_c: LTIAny, Ts: float):
        if Ts <= 0:
            raise ValueError("Ts deve ser > 0.")
        self.A, self.B, self.C, self.D = _Conversions._to_ss(sys_c)
        self.Ts = float(Ts)

    # ---------------------------
    # Métodos de discretização
    # ---------------------------
    def zoh(self) -> DiscretizationResult:
        """Discretização exata por ZOH (Van Loan)."""
        Ad, Phi1 = _VanLoan.phi1(self.A, self.Ts)
        Bd = Phi1 @ self.B
        return DiscretizationResult(Ad, Bd, self.C.copy(), self.D.copy(), dt=self.Ts)

    def foh(self) -> DiscretizationResult:
        """
        Discretização por FOH (first-order hold) para entrada linear por amostra.
        Para degrau (u_k = u_{k+1}), o ganho efetivo é Bd = (Bd0 + Bd1).

        x_{k+1} = Ad x_k + Bd0 u_k + Bd1 u_{k+1}
        """
        Ad, Phi1, Phi2 = _VanLoan.phi1_phi2(self.A, self.Ts)
        Bd0 = (Phi1 - (1.0 / self.Ts) * Phi2) @ self.B
        Bd1 = ((1.0 / self.Ts) * Phi2) @ self.B
        # Para simular degrau sem dois termos, usamos Bd = Bd0 + Bd1
        Bd = Bd0 + Bd1
        return DiscretizationResult(Ad, Bd, self.C.copy(), self.D.copy(), dt=self.Ts,
                                    Bd_foh_u_k=Bd0, Bd_foh_u_k1=Bd1)

    def tustin(self) -> DiscretizationResult:
        """
        Discretização por Tustin (bilinear) via SciPy.
        s ≈ (2/Ts) * (z - 1)/(z + 1)
        """
        Ad, Bd, Cd, Dd, _ = cont2discrete((self.A, self.B, self.C, self.D), self.Ts, method="bilinear")
        return DiscretizationResult(Ad, Bd, Cd, Dd, dt=self.Ts)

    def forward_euler(self) -> DiscretizationResult:
        """
        Discretização por Forward Euler (diferença para frente) via SciPy.
        s ≈ (z - 1)/Ts
        """
        Ad, Bd, Cd, Dd, _ = cont2discrete((self.A, self.B, self.C, self.D), self.Ts, method="forward_diff")
        return DiscretizationResult(Ad, Bd, Cd, Dd, dt=self.Ts)

    def discretize(self, method: Literal["zoh", "tustin", "foh", "forward_euler"]) -> DiscretizationResult:
        if method == "zoh":
            return self.zoh()
        if method == "foh":
            return self.foh()
        if method == "tustin":
            return self.tustin()
        if method == "forward_euler":
            return self.forward_euler()
        raise ValueError(f"Método '{method}' não suportado.")

    # ---------------------------
    # Plot de resposta
    # ---------------------------
    def plot_response(
        self,
        disc_sys: DiscretizationResult,
        kind: Literal["step", "impulse"] = "step",
        Tfinal: float = 5.0,
        compare_continuous: bool = True,
        fig=None, ax=None,
        title_suffix: str = ""
    ):
        """
        Plota resposta do sistema discretizado (e opcionalmente do contínuo, para comparação).

        kind: "step" ou "impulse"
        Tfinal: janela de simulação (s)
        """
        import matplotlib.pyplot as plt

        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(8, 4.2))

        # Discreto
        dlti = _Conversions._to_dlti(disc_sys)
        N = max(5, int(np.ceil(Tfinal / disc_sys.dt)))
        if kind == "step":
            tout_d, y_d = dstep(dlti, n=N)
        else:
            tout_d, y_d = dimpulse(dlti, n=N)
        t_d = np.asarray(tout_d).squeeze() * disc_sys.dt  # índice * Ts -> tempo
        y_d = np.squeeze(y_d)

        # Contínuo (comparação)
        if compare_continuous:
            # Usa SciPy para simular contínuo a 4x a frequência de amostragem
            t_c = np.linspace(0, Tfinal, N * 4)
            if kind == "step":
                tout_c, y_c = step((self.A, self.B, self.C, self.D), T=t_c)
            else:
                tout_c, y_c = impulse((self.A, self.B, self.C, self.D), T=t_c)
            ax.plot(tout_c, np.squeeze(y_c), linewidth=2, alpha=0.8, label=f"Contínuo ({kind})")

        # Plot do discreto (staircase para destacar amostras)
        ax.step(t_d, y_d, where="post", linewidth=2, label=f"Discreto [{kind}] — {disc_sys.dt:.4g}s")

        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel("Saída")
        ttl = f"Resposta {kind.upper()} — método discreto{(' ' + title_suffix) if title_suffix else ''}"
        ax.set_title(ttl)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
        plt.tight_layout()
        plt.show()
