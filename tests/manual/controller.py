def run():
    import auto_control_tools as act

    model = act.NishikawaModelIdentification.get_model('data_input.csv')

    controller = act.CohenCoonControllerAproximation.get_controller(model, act.PID)
    controller.view.print_tf()
    print(controller.tf)
    model.view.plot_model_step_response_graph()
    controller.view.plot_controller_graph()
    controller.view.print_controller_data()


if __name__ == '__main__':
    run()
