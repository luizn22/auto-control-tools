

def run():
    import auto_control_tools as act

    # tf = [[1, 1], [1, 2, 3]]
    # model = Model(tf)
    model = act.FirstOrderModel(1, 2, 1)
    model.view.print_tf()
    model.view.plot_model_graph()


if __name__ == '__main__':
    run()
