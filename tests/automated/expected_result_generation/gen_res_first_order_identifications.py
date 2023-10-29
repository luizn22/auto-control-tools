import os

from auto_control_tools import (
    ZieglerNicholsModelIdentification,
    HagglundModelIdentification,
    SmithModelIdentification,
    SundaresanKrishnaswamyModelIdentification,
    NishikawaModelIdentification,
)
from tests.automated.expected_result_generation.json_handler import save_result_set_to_json
from tests.conftest import FIRST_ORDER_IDENTIFICATION_TEST_CASES


def gen_res_first_order_get_model(method, method_kwargs):

    result = method(**method_kwargs)

    return {
               'method': str(method.__self__),
               'method_kwargs': method_kwargs
           }, result.__dict__


def yield_results(meths, inputs):
    for meth in meths:
        for inp in inputs:
            yield gen_res_first_order_get_model(meth, inp)


if __name__ == "__main__":
    # Get the directory of the current script (conftest.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Set the current working directory to the directory of conftest.py
    os.chdir(current_dir)

    save_result_set_to_json(os.path.abspath('generated/'), 'first_order_get_model_results', yield_results([
        ZieglerNicholsModelIdentification.get_model,
        HagglundModelIdentification.get_model,
        SmithModelIdentification.get_model,
        SundaresanKrishnaswamyModelIdentification.get_model,
        NishikawaModelIdentification.get_model,
    ], FIRST_ORDER_IDENTIFICATION_TEST_CASES))
