import pytest

from auto_control_tools import Model, Controller
from tests.conftest import CONTROLLER_TEST_CASES


@pytest.mark.parametrize('tf, pid', CONTROLLER_TEST_CASES)
def test_controller(tf, pid):
    model = Model(tf)
    ki, kp, kd = pid
    ctrl = Controller(model, ki=ki, kp=kp, kd=kd)

