import pytest
import control
from numpy.testing import assert_array_equal

from auto_control_tools import Model
from tests.conftest import MODEL_TEST_CASES


@pytest.mark.parametrize('tf, tf_type', MODEL_TEST_CASES)
def test_model(tf, tf_type):
    num = tf[0]
    den = tf[1]
    order = len(den) - 1

    if tf_type == 'list':
        inp = [num, den]
    elif tf_type == 'control':
        inp = control.TransferFunction(*tf)

    else:
        raise ValueError(f'Unsupported tf_type {tf_type}')

    m = Model(inp)
    tf = control.TransferFunction(num, den)
    assert_array_equal(m.tf.num, tf.num)
    assert_array_equal(m.tf.den, tf.den)
    assert m.order == order
