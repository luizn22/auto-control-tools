

def run():
    from auto_control_tools import Model, FirstOrderModel

    tf = [[1, 1], [1, 2, 3]]
    # model = Model(tf)
    model = FirstOrderModel(1, 2, 1)
    model.view.print_tf()
    model.view.plot_model_graph()


if __name__ == '__main__':
    run()
