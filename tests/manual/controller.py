def run():
    import auto_control_tools as act
    model = act.FirstOrderModel(
        K=1.95,
        tau=8.33,
        teta=1.48
    )

    controller = act.CohenCoonControllerAproximation.get_controller(model, act.P)
    controller.view.print_tf()
    print(controller.tf)
    model.view.plot_model_graph()
    controller.view.plot_controller_graph()
    controller.view.print_controller_data()


if __name__ == '__main__':
    run()
