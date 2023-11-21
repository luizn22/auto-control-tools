import auto_control_tools as act

model = act.FirstOrderModel(K=1.95, tau=8.33, theta=1.48)

model.view.print_tf()
model.view.print_model_step_response_data()
model.view.plot_model_step_response_graph()

controller = act.ZieglerNicholsControllerAproximation.get_controller(model, act.PID)

controller.view.print_tf()
controller.view.print_controller_step_response_data()
controller.view.plot_controller_step_response_graph(plot_model=False)

controller = act.CohenCoonControllerAproximation.get_controller(model, act.PID)

controller.view.print_tf()
controller.view.print_controller_step_response_data()
controller.view.plot_controller_step_response_graph(plot_model=False)
