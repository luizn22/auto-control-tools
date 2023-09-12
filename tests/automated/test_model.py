import pytest
import control
from numpy.testing import assert_array_equal

from auto_control_tools import Model


@pytest.mark.parametrize('tf, tf_type', [
    item for sublist in [
        [
            ([[1], [1, 2]], tf_type),
            ([[1], [1, 2, 3]], tf_type),
            ([[1, 2], [1, 2, 3]], tf_type),
        ] for tf_type in ['list', 'control']
    ] for item in sublist
])
def test_model_tf_list(tf, tf_type):
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
    assert_array_equal(m.system.num, tf.num)
    assert_array_equal(m.system.den, tf.den)
    assert m.order == order
