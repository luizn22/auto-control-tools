import numpy as np
import control
from auto_control_tools.analysis import c2d

def test_c2d_zoh_pole_mapping():
    Gc = control.tf([1], [1, 1])  # 1/(s+1)
    Ts = 0.1
    Gz = c2d(Gc, Ts, method="zoh")
    # polo discreto esperado: e^{-Ts}
    expected = np.exp(-Ts)
    pz = control.pole(Gz)  # array de polos (complexos)
    # como é 1 polo real
    assert abs(pz[0].real - expected) < 1e-3
