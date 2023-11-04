

def run():
    import auto_control_tools as act

    num = [1]
    den = [1, 1]
    model = act.Model((num, den))
    # model = act.FirstOrderModel(1, 2, 1)
    model.view.print_tf()
    model.view.print_model_step_response_data()
    model.view.plot_model_step_response_graph()


if __name__ == '__main__':
    run()
