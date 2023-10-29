import pytest

from auto_control_tools import (
    ZieglerNicholsModelIdentification,
    HagglundModelIdentification,
    SmithModelIdentification,
    SundaresanKrishnaswamyModelIdentification,
    NishikawaModelIdentification,
)
from tests.automated.expected_result_generation.json_handler import compare_to_json
from tests.conftest import FIRST_ORDER_IDENTIFICATION_TEST_CASES, NO_DELAY_CONST_SAMPLE_TIME_IDENTIFICATION_RESULTS


@pytest.mark.parametrize("method", [
    ZieglerNicholsModelIdentification.get_model,
    HagglundModelIdentification.get_model,
    SmithModelIdentification.get_model,
    SundaresanKrishnaswamyModelIdentification.get_model,
    NishikawaModelIdentification.get_model,
])
@pytest.mark.parametrize("method_kwargs", FIRST_ORDER_IDENTIFICATION_TEST_CASES)
@pytest.mark.parametrize("results", [NO_DELAY_CONST_SAMPLE_TIME_IDENTIFICATION_RESULTS])
def test_first_order_get_model(method, method_kwargs, results):
    model = method(**method_kwargs)
    ins = {
            'method': str(method.__self__),
            'method_kwargs': method_kwargs
        }
    out = model.__dict__

    assert compare_to_json(
        ins,
        out,
        NO_DELAY_CONST_SAMPLE_TIME_IDENTIFICATION_RESULTS
    )
