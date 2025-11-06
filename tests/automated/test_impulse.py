import numpy as np
import control
from auto_control_tools.analysis import impulse_response

def test_impulse_first_order_shape():
    G = control.tf([1], [1, 1])
    T = np.linspace(0, 5, 200)
    t, y = impulse_response(G, T=T)
    assert t.shape == y.shape
    # y(0) ~ 1 e decai
    assert y[0] == y.max()
