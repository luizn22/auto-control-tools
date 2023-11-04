from tests.conftest import FIRST_ORDER_IDENTIFICATION_TEST_CASES


def run():
    import auto_control_tools as act
    x = FIRST_ORDER_IDENTIFICATION_TEST_CASES[2].copy()

    m = act.NishikawaModelIdentification.get_model(**x)  # type: ignore
    m.view.print_tf()
    m.view.print_model_step_response_data()
    m.view.plot_model_step_response_graph()


if __name__ == '__main__':
    run()
