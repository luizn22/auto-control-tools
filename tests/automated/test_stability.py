import math
import pytest
from auto_control_tools.analysis import RouthHurwitz

@pytest.mark.parametrize("coeffs, expected_rhp, expected_stable", [
    ([1, 2, 3, 4], 0, True),        # estável
    ([1, -2, 2], 2, False),         # s^2 - 2s + 2 => 2 polos em RHP
    ([1, 0, 2, 0, 1], 0, True),     # linha nula tratada (aux + deriv)
])
def test_routh_basic(coeffs, expected_rhp, expected_stable):
    res = RouthHurwitz(coeffs).compute()
    assert res.rhp_poles == expected_rhp
    assert res.is_stable == expected_stable
    assert all(not (math.isnan(x) or math.isinf(x)) for x in res.first_column)
