from tests.conftest import FIRST_ORDER_IDENTIFICATION_TEST_CASES


def run():
    import auto_control_tools as act
    x = FIRST_ORDER_IDENTIFICATION_TEST_CASES[2].copy()

    m = act.NishikawaModelIdentification.get_model(**x)  # type: ignore

    # m = act.FirstOrderModel(K=1.95, tau=8.33, theta=1.48)

    c = act.CohenCoonControllerAproximation.get_controller(m, act.PID)

    c.view.print_tf()
    c.view.print_controller_step_response_data()
    c.view.plot_controller_step_response_graph()


if __name__ == '__main__':
    run()
