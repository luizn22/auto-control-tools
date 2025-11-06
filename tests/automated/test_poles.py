import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend não interativo para CI

from control import tf
from auto_control_tools.analysis import PoleZeroAnalyzer

def test_poles_zeros_simple():
    G = tf([1], [1, 2, 1])  # (s+1)^-2
    a = PoleZeroAnalyzer(G, name="G")
    # Polos esperados: -1 (duplicado)
    assert np.isclose(np.sort(a.poles.real), np.array([-1.0, -1.0])).all()
    assert a.zeros.size == 0

def test_plot_runs():
    G = tf([1, 1], [1, 2, 1])  # zero em -1, polos em -1,-1
    a = PoleZeroAnalyzer(G)
    fig, ax = a.plot(show=False)
    assert fig is not None and ax is not None
