import unittest
import numpy as np
import matplotlib
matplotlib.use("Agg")  # evita abrir janela

from control import tf
from auto_control_tools.analysis import PoleZeroAnalyzer

class TestPoleZero(unittest.TestCase):
    def test_polos_zeros_basico(self):
        G = tf([1], [1, 2, 1])  # 1/(s+1)^2
        a = PoleZeroAnalyzer(G)
        self.assertTrue(np.allclose(np.sort(a.poles.real), np.array([-1.0, -1.0])))
        self.assertEqual(a.zeros.size, 0)

    def test_plot_sem_erro(self):
        G = tf([1, 1], [1, 2, 1])  # zero em -1
        a = PoleZeroAnalyzer(G)
        fig, ax = a.plot(show=False)
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

    def test_invariantes_ganho(self):
        # Escalar numerador/denominador não muda zeros/polos
        G1 = tf([1, 3], [1, 2, 1])
        G2 = tf([2, 6], [1, 2, 1])   # num * 2 -> zeros iguais
        G3 = tf([1, 3], [2, 4, 2])   # den * 2 -> polos iguais
        a1, a2, a3 = PoleZeroAnalyzer(G1), PoleZeroAnalyzer(G2), PoleZeroAnalyzer(G3)
        self.assertTrue(np.allclose(np.sort(a1.zeros), np.sort(a2.zeros)))
        self.assertTrue(np.allclose(np.sort(a1.poles), np.sort(a3.poles)))

if __name__ == "__main__":
    unittest.main()
