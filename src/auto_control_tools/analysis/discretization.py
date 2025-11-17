"""
Módulo de Discretização e Análise de Sistemas de Controle
==========================================================

Este módulo fornece ferramentas avançadas para discretização de sistemas de controle
contínuos, incluindo suporte para sistemas em malha fechada e múltiplos métodos de
discretização.

.. module:: auto_control_tools.analysis.discretization
   :platform: Unix, Windows
   :synopsis: Discretização avançada de sistemas de controle

.. moduleauthor:: Matheus Cheim

Exemplos
--------

Discretização básica de um sistema::

    >>> from auto_control_tools.analysis.discretization import Discretizer
    >>> import control
    >>> 
    >>> # Sistema contínuo de segunda ordem
    >>> Gs = control.tf([1], [1, 1, 1])
    >>> 
    >>> # Discretização com período de amostragem de 0.05s
    >>> disc = Discretizer(Gs, Ts=0.05)
    >>> result = disc.discretize(method="zoh")
    >>> 
    >>> # Plotar comparação
    >>> disc.plot_response(result, kind="step", Tfinal=5.0)

Discretização com controlador em malha fechada::

    >>> # Sistema e controlador
    >>> plant = control.tf([1], [1, 2, 1])
    >>> controller = control.tf([2, 1], [1, 0])  # PI
    >>> 
    >>> # Discretização em malha fechada
    >>> disc = Discretizer(plant, Ts=0.1)
    >>> result = disc.discretize_closed_loop(
    ...     controller=controller,
    ...     method="tustin",
    ...     feedback_sign=-1
    ... )

Notas
-----
O módulo implementa os seguintes métodos de discretização:

* **ZOH** (Zero-Order Hold): Método exato usando exponencial de matriz
* **FOH** (First-Order Hold): Para entrada linear por amostra
* **Tustin** (Bilinear): Transformação bilinear s → z
* **Forward Euler**: Diferenças finitas para frente
* **Matched**: Casamento de pólos e zeros (em desenvolvimento)

Para sistemas em malha fechada, o módulo oferece discretização que preserva
as características da resposta contínua mesmo com períodos de amostragem maiores.

"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Tuple, Union, Optional, Dict, Any, List
from enum import Enum
import warnings
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm, inv
from scipy.signal import (
    cont2discrete, StateSpace as SciPySS, TransferFunction as SciPyTF,
    lti as SciPyLTI, dlti as SciPyDLTI, dstep, dimpulse, step, impulse,
    freqz, bode
)

try:
    import control  # type: ignore
    _HAS_CONTROL = True
except ImportError:
    control = None
    _HAS_CONTROL = False

# Type aliases para melhor legibilidade
LTISystem = Union[SciPySS, SciPyTF, SciPyLTI]
if _HAS_CONTROL:
    LTISystem = Union[LTISystem, "control.StateSpace", "control.TransferFunction"]

FloatArray = NDArray[np.float64]


class DiscretizationMethod(Enum):
    """
    Enumeração dos métodos de discretização disponíveis.
    
    Attributes
    ----------
    ZOH : str
        Zero-Order Hold - Mantém entrada constante durante período de amostragem
    FOH : str
        First-Order Hold - Interpolação linear da entrada
    TUSTIN : str
        Transformação bilinear (Tustin)
    FORWARD_EULER : str
        Diferenças finitas para frente
    MATCHED : str
        Casamento de pólos e zeros (experimental)
    """
    ZOH = "zoh"
    FOH = "foh"
    TUSTIN = "tustin"
    FORWARD_EULER = "forward_euler"
    MATCHED = "matched"


@dataclass
class DiscretizationResult:
    """
    Resultado da discretização de um sistema.
    
    Contém as matrizes do sistema discretizado e metadados sobre o processo
    de discretização.
    
    Attributes
    ----------
    Ad : NDArray[np.float64]
        Matriz de estado discretizada (n×n)
    Bd : NDArray[np.float64]
        Matriz de entrada discretizada (n×m)
    Cd : NDArray[np.float64]
        Matriz de saída discretizada (p×n)
    Dd : NDArray[np.float64]
        Matriz de transmissão direta discretizada (p×m)
    dt : float
        Período de amostragem em segundos
    method : str
        Método de discretização utilizado
    metadata : Dict[str, Any]
        Metadados adicionais (e.g., erro de aproximação, pólos)
    Bd_foh_u_k : Optional[NDArray[np.float64]]
        Matriz B para u_k (apenas FOH)
    Bd_foh_u_k1 : Optional[NDArray[np.float64]]
        Matriz B para u_{k+1} (apenas FOH)
    
    Examples
    --------
    >>> result = discretizer.discretize(method="zoh")
    >>> print(f"Período: {result.dt}s")
    >>> print(f"Método: {result.method}")
    >>> print(f"Pólos discretos: {result.metadata.get('poles', [])}")
    """
    Ad: FloatArray
    Bd: FloatArray
    Cd: FloatArray
    Dd: FloatArray
    dt: float
    method: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    Bd_foh_u_k: Optional[FloatArray] = None
    Bd_foh_u_k1: Optional[FloatArray] = None
    
    def to_dlti(self) -> SciPyDLTI:
        """
        Converte o resultado para sistema discreto SciPy.
        
        Returns
        -------
        SciPyDLTI
            Sistema discreto no formato SciPy
        """
        return SciPyDLTI(self.Ad, self.Bd, self.Cd, self.Dd, dt=self.dt)
    
    def get_poles(self) -> FloatArray:
        """
        Calcula os pólos do sistema discretizado.
        
        Returns
        -------
        NDArray[np.float64]
            Array com os pólos (autovalores de Ad)
        """
        return np.linalg.eigvals(self.Ad)
    
    def is_stable(self) -> bool:
        """
        Verifica estabilidade do sistema discretizado.
        
        Returns
        -------
        bool
            True se todos os pólos estão dentro do círculo unitário
        """
        poles = self.get_poles()
        return bool(np.all(np.abs(poles) < 1.0))


class ValidationError(Exception):
    """Exceção para erros de validação nos parâmetros."""
    pass


class DiscretizationValidator:
    """
    Validador para parâmetros de discretização.
    
    Esta classe fornece métodos estáticos para validação de parâmetros
    e sistemas antes da discretização.
    """
    
    @staticmethod
    def validate_sampling_time(Ts: float, system_poles: Optional[FloatArray] = None) -> None:
        """
        Valida o período de amostragem.
        
        Parameters
        ----------
        Ts : float
            Período de amostragem em segundos
        system_poles : Optional[NDArray[np.float64]]
            Pólos do sistema contínuo para validação de Nyquist
            
        Raises
        ------
        ValidationError
            Se Ts <= 0 ou não satisfaz critério de Nyquist
        """
        if Ts <= 0:
            raise ValidationError("Período de amostragem deve ser positivo")
        
        if system_poles is not None and len(system_poles) > 0:
            # Verifica critério de Nyquist-Shannon
            max_freq = np.max(np.abs(np.imag(system_poles))) / (2 * np.pi)
            if max_freq > 0:
                nyquist_Ts = 1 / (2 * max_freq)
                if Ts > nyquist_Ts:
                    warnings.warn(
                        f"Período de amostragem ({Ts:.4f}s) pode ser muito grande. "
                        f"Recomendado: Ts < {nyquist_Ts:.4f}s para satisfazer Nyquist",
                        UserWarning
                    )
    
    @staticmethod
    def validate_system_dimensions(A: FloatArray, B: FloatArray, 
                                  C: FloatArray, D: FloatArray) -> None:
        """
        Valida dimensões das matrizes do sistema.
        
        Parameters
        ----------
        A, B, C, D : NDArray[np.float64]
            Matrizes do sistema em espaço de estados
            
        Raises
        ------
        ValidationError
            Se as dimensões são incompatíveis
        """
        n = A.shape[0]
        if A.shape != (n, n):
            raise ValidationError(f"Matriz A deve ser quadrada, recebido shape {A.shape}")
        
        if B.shape[0] != n:
            raise ValidationError(f"B deve ter {n} linhas, recebido {B.shape[0]}")
        
        if C.shape[1] != n:
            raise ValidationError(f"C deve ter {n} colunas, recebido {C.shape[1]}")
        
        m = B.shape[1]
        p = C.shape[0]
        
        if D.shape != (p, m):
            raise ValidationError(f"D deve ter shape ({p}, {m}), recebido {D.shape}")


class SystemConverter:
    """
    Conversor entre diferentes representações de sistemas LTI.
    
    Suporta conversão entre formatos do SciPy e python-control.
    """
    
    @staticmethod
    def to_state_space(sys: LTISystem) -> Tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """
        Converte sistema para representação em espaço de estados.
        
        Parameters
        ----------
        sys : LTISystem
            Sistema em qualquer formato suportado
            
        Returns
        -------
        Tuple[NDArray, NDArray, NDArray, NDArray]
            Matrizes (A, B, C, D) do sistema
            
        Raises
        ------
        TypeError
            Se o tipo de sistema não é suportado
        """
        # python-control
        if _HAS_CONTROL:
            if isinstance(sys, control.TransferFunction):
                ss = control.ss(sys)
                A, B, C, D = control.ssdata(ss)
                return (np.asarray(A, dtype=float), np.asarray(B, dtype=float),
                       np.asarray(C, dtype=float), np.asarray(D, dtype=float))
            
            if isinstance(sys, control.StateSpace):
                A, B, C, D = control.ssdata(sys)
                return (np.asarray(A, dtype=float), np.asarray(B, dtype=float),
                       np.asarray(C, dtype=float), np.asarray(D, dtype=float))
        
        # SciPy
        if isinstance(sys, SciPyTF):
            ss = sys.to_ss()
            return (np.asarray(ss.A, dtype=float), np.asarray(ss.B, dtype=float),
                   np.asarray(ss.C, dtype=float), np.asarray(ss.D, dtype=float))
        
        if isinstance(sys, (SciPySS, SciPyLTI)):
            if hasattr(sys, 'to_ss'):
                ss = sys.to_ss()
            else:
                ss = sys
            return (np.asarray(ss.A, dtype=float), np.asarray(ss.B, dtype=float),
                   np.asarray(ss.C, dtype=float), np.asarray(ss.D, dtype=float))
        
        # Tupla (num, den)
        if isinstance(sys, tuple) and len(sys) == 2:
            num, den = sys
            tf = SciPyTF(num, den)
            ss = tf.to_ss()
            return (np.asarray(ss.A, dtype=float), np.asarray(ss.B, dtype=float),
                   np.asarray(ss.C, dtype=float), np.asarray(ss.D, dtype=float))
        
        raise TypeError(
            f"Tipo de sistema não suportado: {type(sys)}. "
            "Use TransferFunction ou StateSpace (control ou scipy)."
        )


class MatrixExponentialCalculator:
    """
    Calculador de exponenciais de matriz usando método de Van Loan.
    
    Este método evita inversão de matrizes, sendo robusto para sistemas
    com matriz A singular.
    """
    
    @staticmethod
    def compute_phi1(A: FloatArray, T: float) -> Tuple[FloatArray, FloatArray]:
        """
        Calcula Ad = e^(AT) e Φ₁ = ∫₀ᵀ e^(Aτ) dτ.
        
        Parameters
        ----------
        A : NDArray[np.float64]
            Matriz de estado contínua
        T : float
            Período de integração
            
        Returns
        -------
        Tuple[NDArray, NDArray]
            (Ad, Φ₁) onde Ad é a matriz de transição e Φ₁ a integral
            
        Notes
        -----
        Usa o método de Van Loan que constrói uma matriz aumentada::
        
            M = [A  I]
                [0  0]
                
        tal que exp(MT) = [Ad  Φ₁]
                          [0   I ]
        """
        n = A.shape[0]
        Z = np.zeros((n, n))
        I = np.eye(n)
        
        # Matriz aumentada
        M = np.block([[A, I],
                     [Z, Z]])
        
        # Exponencial da matriz aumentada
        expM = expm(M * T)
        
        Ad = expM[:n, :n]
        Phi1 = expM[:n, n:]
        
        return Ad, Phi1
    
    @staticmethod
    def compute_phi1_phi2(A: FloatArray, T: float) -> Tuple[FloatArray, FloatArray, FloatArray]:
        """
        Calcula Ad, Φ₁ e Φ₂ para discretização FOH.
        
        Parameters
        ----------
        A : NDArray[np.float64]
            Matriz de estado contínua
        T : float
            Período de integração
            
        Returns
        -------
        Tuple[NDArray, NDArray, NDArray]
            (Ad, Φ₁, Φ₂) onde:
            - Ad = e^(AT)
            - Φ₁ = ∫₀ᵀ e^(Aτ) dτ  
            - Φ₂ = ∫₀ᵀ τe^(Aτ) dτ
        """
        n = A.shape[0]
        Z = np.zeros((n, n))
        I = np.eye(n)
        
        # Matriz aumentada 3x3
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
    Classe principal para discretização de sistemas de controle.
    
    Fornece métodos para discretização de sistemas contínuos usando
    diferentes técnicas, incluindo suporte para sistemas em malha fechada.
    
    Parameters
    ----------
    sys_c : LTISystem
        Sistema contínuo (TransferFunction ou StateSpace)
    Ts : float
        Período de amostragem em segundos
    validate : bool, optional
        Se True, valida parâmetros automaticamente (default: True)
        
    Attributes
    ----------
    A, B, C, D : NDArray[np.float64]
        Matrizes do sistema em espaço de estados
    Ts : float
        Período de amostragem
    continuous_poles : NDArray[np.float64]
        Pólos do sistema contínuo
        
    Raises
    ------
    ValidationError
        Se os parâmetros são inválidos
        
    Examples
    --------
    Discretização simples:
    
    >>> disc = Discretizer(sys, Ts=0.1)
    >>> result = disc.discretize("zoh")
    
    Discretização em malha fechada:
    
    >>> disc = Discretizer(plant, Ts=0.1)
    >>> result = disc.discretize_closed_loop(controller, method="tustin")
    
    Análise de múltiplos métodos:
    
    >>> results = disc.compare_methods(["zoh", "tustin", "foh"])
    >>> disc.plot_comparison(results)
    """
    
    def __init__(self, sys_c: LTISystem, Ts: float, validate: bool = True):
        """Inicializa o discretizador."""
        self.A, self.B, self.C, self.D = SystemConverter.to_state_space(sys_c)
        self.Ts = float(Ts)
        self.continuous_poles = np.linalg.eigvals(self.A)
        
        if validate:
            DiscretizationValidator.validate_sampling_time(self.Ts, self.continuous_poles)
            DiscretizationValidator.validate_system_dimensions(
                self.A, self.B, self.C, self.D
            )
    
    def zoh(self) -> DiscretizationResult:
        """
        Discretização por Zero-Order Hold (ZOH).
        
        Método exato que assume entrada constante durante o período
        de amostragem.
        
        Returns
        -------
        DiscretizationResult
            Sistema discretizado
            
        Notes
        -----
        A discretização ZOH é dada por:
        
        .. math::
            
            x_{k+1} = e^{AT_s} x_k + \\int_0^{T_s} e^{A\\tau} d\\tau \\cdot B u_k
            
            y_k = C x_k + D u_k
        """
        Ad, Phi1 = MatrixExponentialCalculator.compute_phi1(self.A, self.Ts)
        Bd = Phi1 @ self.B
        
        result = DiscretizationResult(
            Ad=Ad, Bd=Bd, Cd=self.C.copy(), Dd=self.D.copy(),
            dt=self.Ts, method="zoh"
        )
        
        # Adiciona metadados
        result.metadata['continuous_poles'] = self.continuous_poles
        result.metadata['discrete_poles'] = result.get_poles()
        result.metadata['is_stable'] = result.is_stable()
        
        return result
    
    def foh(self) -> DiscretizationResult:
        """
        Discretização por First-Order Hold (FOH).
        
        Assume interpolação linear da entrada entre amostras.
        
        Returns
        -------
        DiscretizationResult
            Sistema discretizado com matrizes adicionais para FOH
            
        Notes
        -----
        A discretização FOH resulta em:
        
        .. math::
            
            x_{k+1} = A_d x_k + B_{d0} u_k + B_{d1} u_{k+1}
            
        Para entrada tipo degrau, usa-se :math:`B_d = B_{d0} + B_{d1}`
        """
        Ad, Phi1, Phi2 = MatrixExponentialCalculator.compute_phi1_phi2(self.A, self.Ts)
        
        Bd0 = (Phi1 - (1.0 / self.Ts) * Phi2) @ self.B
        Bd1 = ((1.0 / self.Ts) * Phi2) @ self.B
        
        # Para simulação com entrada degrau
        Bd = Bd0 + Bd1
        
        result = DiscretizationResult(
            Ad=Ad, Bd=Bd, Cd=self.C.copy(), Dd=self.D.copy(),
            dt=self.Ts, method="foh",
            Bd_foh_u_k=Bd0, Bd_foh_u_k1=Bd1
        )
        
        result.metadata['continuous_poles'] = self.continuous_poles
        result.metadata['discrete_poles'] = result.get_poles()
        result.metadata['is_stable'] = result.is_stable()
        
        return result
    
    def tustin(self, prewarp_frequency: Optional[float] = None) -> DiscretizationResult:
        """
        Discretização por transformação bilinear (Tustin).
        
        Parameters
        ----------
        prewarp_frequency : float, optional
            Frequência para pré-warping em rad/s
            
        Returns
        -------
        DiscretizationResult
            Sistema discretizado
            
        Notes
        -----
        A transformação de Tustin usa:
        
        .. math::
            
            s = \\frac{2}{T_s} \\frac{z - 1}{z + 1}
            
        Com pré-warping:
        
        .. math::
            
            s = \\frac{\\omega}{\\tan(\\omega T_s / 2)} \\frac{z - 1}{z + 1}
        """
        if prewarp_frequency is not None:
            # Implementação com pré-warping
            w = prewarp_frequency
            alpha = w / np.tan(w * self.Ts / 2)
            method_opts = {'alpha': alpha}
        else:
            method_opts = None
        
        Ad, Bd, Cd, Dd, _ = cont2discrete(
            (self.A, self.B, self.C, self.D), 
            self.Ts, 
            method="bilinear",
            method_options=method_opts
        )
        
        result = DiscretizationResult(
            Ad=Ad, Bd=Bd, Cd=Cd, Dd=Dd,
            dt=self.Ts, method="tustin"
        )
        
        result.metadata['continuous_poles'] = self.continuous_poles
        result.metadata['discrete_poles'] = result.get_poles()
        result.metadata['is_stable'] = result.is_stable()
        if prewarp_frequency:
            result.metadata['prewarp_frequency'] = prewarp_frequency
        
        return result
    
    def forward_euler(self) -> DiscretizationResult:
        """
        Discretização por Forward Euler (diferenças finitas).
        
        Returns
        -------
        DiscretizationResult
            Sistema discretizado
            
        Notes
        -----
        Aproximação de primeira ordem:
        
        .. math::
            
            s \\approx \\frac{z - 1}{T_s}
            
        .. warning::
            
            Este método pode gerar sistemas instáveis mesmo quando o
            sistema contínuo é estável. Use com cuidado.
        """
        Ad, Bd, Cd, Dd, _ = cont2discrete(
            (self.A, self.B, self.C, self.D),
            self.Ts,
            method="forward_diff"
        )
        
        result = DiscretizationResult(
            Ad=Ad, Bd=Bd, Cd=Cd, Dd=Dd,
            dt=self.Ts, method="forward_euler"
        )
        
        result.metadata['continuous_poles'] = self.continuous_poles
        result.metadata['discrete_poles'] = result.get_poles()
        result.metadata['is_stable'] = result.is_stable()
        
        if not result.is_stable() and np.all(np.real(self.continuous_poles) < 0):
            warnings.warn(
                "Forward Euler gerou sistema instável a partir de sistema estável. "
                "Considere reduzir Ts ou usar outro método.",
                UserWarning
            )
        
        return result
    
    def matched(self) -> DiscretizationResult:
        """
        Discretização por casamento de pólos e zeros (experimental).
        
        Returns
        -------
        DiscretizationResult
            Sistema discretizado
            
        Notes
        -----
        Este método mapeia pólos e zeros do plano-s para o plano-z usando:
        
        .. math::
            
            z = e^{sT_s}
            
        .. warning::
            
            Implementação experimental. Use com cautela.
        """
        # Implementação simplificada - usa ZOH como base
        warnings.warn(
            "Método 'matched' ainda experimental. Usando ZOH como aproximação.",
            UserWarning
        )
        result = self.zoh()
        result.method = "matched (experimental)"
        return result
    
    def discretize(self, method: Union[str, DiscretizationMethod],
                  **kwargs) -> DiscretizationResult:
        """
        Discretiza o sistema usando o método especificado.
        
        Parameters
        ----------
        method : str or DiscretizationMethod
            Método de discretização ('zoh', 'foh', 'tustin', 'forward_euler', 'matched')
        **kwargs
            Argumentos adicionais para o método (e.g., prewarp_frequency para Tustin)
            
        Returns
        -------
        DiscretizationResult
            Sistema discretizado
            
        Raises
        ------
        ValueError
            Se o método não é suportado
        """
        if isinstance(method, DiscretizationMethod):
            method = method.value
        
        method_map = {
            "zoh": self.zoh,
            "foh": self.foh,
            "tustin": lambda: self.tustin(**kwargs),
            "forward_euler": self.forward_euler,
            "matched": self.matched
        }
        
        if method not in method_map:
            raise ValueError(
                f"Método '{method}' não suportado. "
                f"Use um de: {list(method_map.keys())}"
            )
        
        return method_map[method]()
    
    def discretize_closed_loop(self, controller: LTISystem,
                             method: Union[str, DiscretizationMethod] = "tustin",
                             feedback_sign: float = -1.0,
                             reference_filter: Optional[LTISystem] = None,
                             **kwargs) -> DiscretizationResult:
        """
        Discretiza sistema em malha fechada com controlador.
        
        Este método é crucial para obter uma discretização que preserva
        as características da resposta em malha fechada do sistema contínuo.
        
        Parameters
        ----------
        controller : LTISystem
            Controlador do sistema
        method : str or DiscretizationMethod, optional
            Método de discretização (default: 'tustin')
        feedback_sign : float, optional
            Sinal da realimentação (-1 para negativa, default)
        reference_filter : LTISystem, optional
            Filtro de referência (pré-filtro)
        **kwargs
            Argumentos adicionais para o método
            
        Returns
        -------
        DiscretizationResult
            Sistema em malha fechada discretizado
            
        Notes
        -----
        O sistema em malha fechada é formado por:
        
        .. math::
            
            T(s) = \\frac{C(s)G(s)}{1 + feedback\\_sign \\cdot C(s)G(s)}
            
        onde C(s) é o controlador e G(s) é a planta.
        
        Examples
        --------
        >>> # Planta e controlador PI
        >>> plant = control.tf([1], [1, 2, 1])
        >>> controller = control.tf([2, 1], [1, 0])
        >>> 
        >>> # Discretização em malha fechada
        >>> disc = Discretizer(plant, Ts=0.1)
        >>> result = disc.discretize_closed_loop(
        ...     controller=controller,
        ...     method="tustin",
        ...     feedback_sign=-1
        ... )
        """
        # Converte controlador para espaço de estados
        Ac, Bc, Cc, Dc = SystemConverter.to_state_space(controller)
        
        # Forma o sistema em malha fechada aumentado
        # Estado aumentado: [x_planta; x_controlador]
        n_plant = self.A.shape[0]
        n_ctrl = Ac.shape[0]
        
        # Verifica compatibilidade dimensional
        if self.C.shape[0] != Bc.shape[0]:
            raise ValidationError(
                f"Dimensões incompatíveis: saída da planta ({self.C.shape[0]}) "
                f"!= entrada do controlador ({Bc.shape[0]})"
            )
        
        if self.B.shape[1] != Cc.shape[0]:
            raise ValidationError(
                f"Dimensões incompatíveis: entrada da planta ({self.B.shape[1]}) "
                f"!= saída do controlador ({Cc.shape[0]})"
            )
        
        # Constrói sistema aumentado em malha fechada
        # Assumindo configuração padrão: realimentação unitária negativa
        
        # Matriz A aumentada
        A_aug = np.block([
            [self.A - feedback_sign * self.B @ Dc @ self.C, self.B @ Cc],
            [-feedback_sign * Bc @ self.C, Ac]
        ])
        
        # Matriz B aumentada (entrada de referência)
        B_aug = np.block([
            [self.B @ Dc],
            [Bc]
        ])
        
        # Matriz C aumentada (saída = saída da planta)
        C_aug = np.block([self.C, np.zeros((self.C.shape[0], n_ctrl))])
        
        # Matriz D aumentada
        D_aug = self.D @ Dc
        
        # Aplica filtro de referência se fornecido
        if reference_filter is not None:
            Af, Bf, Cf, Df = SystemConverter.to_state_space(reference_filter)
            # TODO: Implementar cascata com filtro de referência
            warnings.warn(
                "Filtro de referência ainda não implementado completamente",
                UserWarning
            )
        
        # Cria discretizador para sistema em malha fechada
        closed_loop_sys = SciPySS(A_aug, B_aug, C_aug, D_aug)
        disc_cl = Discretizer(closed_loop_sys, self.Ts, validate=False)
        
        # Discretiza usando método especificado
        result = disc_cl.discretize(method, **kwargs)
        
        # Adiciona metadados sobre malha fechada
        result.metadata['closed_loop'] = True
        result.metadata['controller_type'] = type(controller).__name__
        result.metadata['feedback_sign'] = feedback_sign
        result.metadata['original_plant_poles'] = self.continuous_poles
        result.metadata['closed_loop_poles_continuous'] = np.linalg.eigvals(A_aug)
        
        return result
    
    def compare_methods(self, methods: Optional[List[str]] = None) -> Dict[str, DiscretizationResult]:
        """
        Compara múltiplos métodos de discretização.
        
        Parameters
        ----------
        methods : List[str], optional
            Lista de métodos a comparar. Se None, usa todos disponíveis.
            
        Returns
        -------
        Dict[str, DiscretizationResult]
            Dicionário com resultados de cada método
            
        Examples
        --------
        >>> results = disc.compare_methods(["zoh", "tustin", "foh"])
        >>> for method, result in results.items():
        ...     print(f"{method}: estável = {result.is_stable()}")
        """
        if methods is None:
            methods = ["zoh", "tustin", "foh", "forward_euler"]
        
        results = {}
        for method in methods:
            try:
                results[method] = self.discretize(method)
            except Exception as e:
                warnings.warn(f"Erro ao discretizar com {method}: {e}")
                
        return results
    
    def estimate_discretization_error(self, result: DiscretizationResult,
                                     freq_range: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """
        Estima o erro de discretização comparando respostas em frequência.
        
        Parameters
        ----------
        result : DiscretizationResult
            Resultado da discretização
        freq_range : Tuple[float, float], optional
            Faixa de frequência para análise (Hz)
            
        Returns
        -------
        Dict[str, Any]
            Métricas de erro incluindo erro RMS e máximo
        """
        # Define faixa de frequência
        if freq_range is None:
            # Usa até frequência de Nyquist
            freq_max = np.pi / self.Ts  # rad/s
            freq_range = (0.01, freq_max)
        
        # Frequências para análise
        w = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), 100)
        
        # Resposta em frequência contínua
        sys_c = SciPyLTI(self.A, self.B, self.C, self.D)
        _, mag_c, phase_c = bode(sys_c, w)
        
        # Resposta em frequência discreta
        sys_d = result.to_dlti()
        w_d = w * self.Ts  # Normaliza frequência
        _, h = freqz(sys_d.num, sys_d.den, worN=w_d)
        mag_d = 20 * np.log10(np.abs(h))
        phase_d = np.angle(h, deg=True)
        
        # Calcula erros
        mag_error = mag_d - mag_c
        phase_error = phase_d - phase_c
        
        # Normaliza erro de fase para [-180, 180]
        phase_error = np.mod(phase_error + 180, 360) - 180
        
        error_metrics = {
            'mag_error_rms': np.sqrt(np.mean(mag_error**2)),
            'mag_error_max': np.max(np.abs(mag_error)),
            'phase_error_rms': np.sqrt(np.mean(phase_error**2)),
            'phase_error_max': np.max(np.abs(phase_error)),
            'frequencies': w,
            'mag_error': mag_error,
            'phase_error': phase_error
        }
        
        return error_metrics
    
    def plot_response(self, disc_sys: DiscretizationResult,
                     kind: Literal["step", "impulse"] = "step",
                     Tfinal: float = 5.0,
                     compare_continuous: bool = True,
                     fig=None, ax=None,
                     title_suffix: str = "") -> None:
        """
        Plota resposta temporal do sistema discretizado.
        
        Parameters
        ----------
        disc_sys : DiscretizationResult
            Sistema discretizado
        kind : {'step', 'impulse'}, optional
            Tipo de resposta (default: 'step')
        Tfinal : float, optional
            Tempo final de simulação em segundos (default: 5.0)
        compare_continuous : bool, optional
            Se True, plota também resposta contínua (default: True)
        fig, ax : matplotlib objects, optional
            Figura e eixo para plotagem
        title_suffix : str, optional
            Sufixo para o título do gráfico
            
        Examples
        --------
        >>> result = disc.discretize("zoh")
        >>> disc.plot_response(result, kind="step", Tfinal=10.0)
        """
        import matplotlib.pyplot as plt
        
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sistema discreto
        dlti = disc_sys.to_dlti()
        N = max(10, int(np.ceil(Tfinal / disc_sys.dt)))
        
        if kind == "step":
            tout_d, y_d = dstep(dlti, n=N)
        else:
            tout_d, y_d = dimpulse(dlti, n=N)
        
        t_d = np.asarray(tout_d).squeeze() * disc_sys.dt
        y_d = np.squeeze(y_d)
        
        # Sistema contínuo para comparação
        if compare_continuous:
            t_c = np.linspace(0, Tfinal, max(1000, N * 10))
            if kind == "step":
                tout_c, y_c = step((self.A, self.B, self.C, self.D), T=t_c)
            else:
                tout_c, y_c = impulse((self.A, self.B, self.C, self.D), T=t_c)
            
            ax.plot(tout_c, np.squeeze(y_c), 'b-', linewidth=2, alpha=0.7,
                   label=f'Contínuo ({kind})')
        
        # Plota discreto
        ax.step(t_d, y_d, 'r-', where='post', linewidth=2,
               label=f'Discreto [{disc_sys.method}] — Ts={disc_sys.dt:.3f}s')
        ax.plot(t_d, y_d, 'ro', markersize=4, alpha=0.6)
        
        # Formatação
        ax.set_xlabel('Tempo (s)', fontsize=11)
        ax.set_ylabel('Amplitude', fontsize=11)
        
        title = f'Resposta ao {kind.capitalize()}'
        if disc_sys.metadata.get('closed_loop'):
            title += ' (Malha Fechada)'
        if title_suffix:
            title += f' — {title_suffix}'
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        ax.grid(True, which='both', alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=10)
        
        # Adiciona informações sobre estabilidade
        if disc_sys.is_stable():
            stability_text = "Sistema Estável"
            color = 'green'
        else:
            stability_text = "Sistema Instável!"
            color = 'red'
        
        ax.text(0.02, 0.98, stability_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', color=color,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if fig is not None:
            plt.show()
    
    def plot_comparison(self, results: Dict[str, DiscretizationResult],
                       kind: Literal["step", "impulse"] = "step",
                       Tfinal: float = 5.0) -> None:
        """
        Plota comparação entre múltiplos métodos de discretização.
        
        Parameters
        ----------
        results : Dict[str, DiscretizationResult]
            Dicionário com resultados de diferentes métodos
        kind : {'step', 'impulse'}, optional
            Tipo de resposta (default: 'step')
        Tfinal : float, optional
            Tempo final de simulação (default: 5.0)
            
        Examples
        --------
        >>> results = disc.compare_methods(["zoh", "tustin"])
        >>> disc.plot_comparison(results, kind="step")
        """
        import matplotlib.pyplot as plt
        
        n_methods = len(results)
        fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 5))
        
        if n_methods == 1:
            axes = [axes]
        
        for ax, (method, result) in zip(axes, results.items()):
            self.plot_response(
                result, kind=kind, Tfinal=Tfinal,
                compare_continuous=True,
                fig=fig, ax=ax,
                title_suffix=f"Método: {method.upper()}"
            )
        
        plt.suptitle(f'Comparação de Métodos de Discretização — Ts = {self.Ts:.3f}s',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()
    
    def plot_pole_zero_map(self, results: Optional[Dict[str, DiscretizationResult]] = None) -> None:
        """
        Plota mapa de pólos e zeros no plano-z.
        
        Parameters
        ----------
        results : Dict[str, DiscretizationResult], optional
            Resultados a plotar. Se None, plota apenas ZOH.
            
        Examples
        --------
        >>> results = disc.compare_methods()
        >>> disc.plot_pole_zero_map(results)
        """
        import matplotlib.pyplot as plt
        
        if results is None:
            results = {"zoh": self.zoh()}
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Círculo unitário
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, linewidth=1)
        
        # Cores para diferentes métodos
        colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
        
        for (method, result), color in zip(results.items(), colors):
            poles = result.get_poles()
            ax.scatter(poles.real, poles.imag, s=100, c=[color],
                      marker='x', linewidth=2, label=f'{method} pólos')
        
        # Pólos contínuos mapeados
        cont_poles_mapped = np.exp(self.continuous_poles * self.Ts)
        ax.scatter(cont_poles_mapped.real, cont_poles_mapped.imag,
                  s=50, c='gray', marker='o', alpha=0.5,
                  label='Pólos contínuos mapeados')
        
        ax.set_xlabel('Real', fontsize=11)
        ax.set_ylabel('Imaginário', fontsize=11)
        ax.set_title(f'Mapa de Pólos no Plano-Z — Ts = {self.Ts:.3f}s',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        ax.axis('equal')
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        
        # Região de estabilidade
        ax.fill_between(np.cos(theta), np.sin(theta), -2, alpha=0.1, color='green')
        ax.text(0, 0, 'Região\nEstável', ha='center', va='center',
               fontsize=10, alpha=0.5)
        
        plt.tight_layout()
        plt.show()


def optimize_sampling_time(sys: LTISystem, 
                          method: str = "zoh",
                          min_samples_per_period: int = 10,
                          max_samples_per_rise_time: int = 5) -> float:
    """
    Estima período de amostragem ótimo para o sistema.
    
    Parameters
    ----------
    sys : LTISystem
        Sistema contínuo
    method : str, optional
        Método de discretização a considerar (default: 'zoh')
    min_samples_per_period : int, optional
        Mínimo de amostras por período de oscilação (default: 10)
    max_samples_per_rise_time : int, optional
        Máximo de amostras por tempo de subida (default: 5)
        
    Returns
    -------
    float
        Período de amostragem recomendado
        
    Notes
    -----
    O período é escolhido baseado em:
    
    1. Frequência natural do sistema
    2. Tempo de acomodação
    3. Requisitos de Nyquist
    
    Examples
    --------
    >>> Ts_opt = optimize_sampling_time(sys)
    >>> print(f"Período recomendado: {Ts_opt:.4f}s")
    """
    A, B, C, D = SystemConverter.to_state_space(sys)
    poles = np.linalg.eigvals(A)
    
    if len(poles) == 0:
        return 0.1  # Default
    
    # Frequência natural máxima
    wn_max = np.max(np.abs(poles))
    if wn_max > 0:
        # Período baseado na frequência natural
        Ts_freq = 2 * np.pi / (min_samples_per_period * wn_max)
    else:
        Ts_freq = 1.0
    
    # Tempo de acomodação (2% criterion)
    real_parts = np.real(poles[poles.real < 0])
    if len(real_parts) > 0:
        settling_time = 4 / np.min(np.abs(real_parts))
        Ts_settling = settling_time / (max_samples_per_rise_time * 10)
    else:
        Ts_settling = Ts_freq
    
    # Escolhe o menor período
    Ts = min(Ts_freq, Ts_settling)
    
    # Limita entre valores práticos
    Ts = np.clip(Ts, 0.001, 1.0)
    
    return float(Ts)


# Funções de conveniência para uso rápido

def quick_discretize(sys: LTISystem, Ts: float = None, 
                    method: str = "zoh") -> DiscretizationResult:
    """
    Discretização rápida com período automático.
    
    Parameters
    ----------
    sys : LTISystem
        Sistema a discretizar
    Ts : float, optional
        Período de amostragem. Se None, calcula automaticamente.
    method : str, optional
        Método de discretização (default: 'zoh')
        
    Returns
    -------
    DiscretizationResult
        Sistema discretizado
        
    Examples
    --------
    >>> result = quick_discretize(sys)
    >>> print(f"Ts automático: {result.dt}s")
    """
    if Ts is None:
        Ts = optimize_sampling_time(sys, method)
        print(f"Período de amostragem automático: {Ts:.4f}s")
    
    disc = Discretizer(sys, Ts)
    return disc.discretize(method)


def compare_all_methods(sys: LTISystem, Ts: float = None) -> None:
    """
    Compara todos os métodos de discretização visualmente.
    
    Parameters
    ----------
    sys : LTISystem
        Sistema a analisar
    Ts : float, optional
        Período de amostragem. Se None, calcula automaticamente.
        
    Examples
    --------
    >>> compare_all_methods(sys, Ts=0.1)
    """
    if Ts is None:
        Ts = optimize_sampling_time(sys)
    
    disc = Discretizer(sys, Ts)
    results = disc.compare_methods()
    
    # Plota comparação
    disc.plot_comparison(results)
    
    # Plota mapa de pólos
    disc.plot_pole_zero_map(results)
    
    # Imprime resumo
    print("\n" + "="*60)
    print(f"RESUMO DA DISCRETIZAÇÃO (Ts = {Ts:.4f}s)")
    print("="*60)
    
    for method, result in results.items():
        print(f"\n{method.upper()}:")
        print(f"  - Estável: {'✓' if result.is_stable() else '✗'}")
        print(f"  - Maior pólo: {np.max(np.abs(result.get_poles())):.4f}")
        
        # Estima erro se possível
        try:
            error = disc.estimate_discretization_error(result)
            print(f"  - Erro RMS magnitude: {error['mag_error_rms']:.2f} dB")
            print(f"  - Erro RMS fase: {error['phase_error_rms']:.2f}°")
        except:
            pass


if __name__ == "__main__":
    # Exemplo de uso
    print("Exemplo de discretização avançada")
    print("-" * 40)
    
    # Sistema de exemplo (pode usar scipy ou control)
    from scipy.signal import TransferFunction
    
    # Sistema de segunda ordem subamortecido
    wn = 2.0  # Frequência natural
    zeta = 0.5  # Fator de amortecimento
    num = [wn**2]
    den = [1, 2*zeta*wn, wn**2]
    
    sys = TransferFunction(num, den)
    
    # Discretização automática
    print("\n1. Discretização com Ts automático:")
    result = quick_discretize(sys)
    
    # Comparação de métodos
    print("\n2. Comparação de todos os métodos:")
    compare_all_methods(sys, Ts=0.1)