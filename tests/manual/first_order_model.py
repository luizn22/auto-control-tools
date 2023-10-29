

def run():
    import auto_control_tools as act

    model = act.FirstOrderModel(K=1, tau=2, theta=2, pade_degree=2)
    model.view.plot_model_step_response_graph()

    model = act.FirstOrderModel(K=1, tau=2, theta=2, pade_degree=10)
    model.view.plot_model_step_response_graph()


if __name__ == '__main__':
    run()
