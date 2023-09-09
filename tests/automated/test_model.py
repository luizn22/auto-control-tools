import pytest
from control import TransferFunction
from numpy.testing import assert_array_equal

from auto_control_tools import Model


@pytest.mark.parametrize('num,den', [
    [[1], [1, 2]],
    [[1], [1, 2, 3]],
    [[1, 2], [1, 2, 3]],
])
def test_model(num, den):
    order = len(den) - 1

    m = Model([num, den])
    tf = TransferFunction(num, den)
    assert_array_equal(m.tf.num, tf.num)
    assert_array_equal(m.tf.den, tf.den)
    assert m.order == order
