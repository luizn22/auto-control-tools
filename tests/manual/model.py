

def run():
    from auto_control_tools import Model, Controller

    tf = [[1, 1], [1, 2, 3]]
    model = Model(tf)
    model.view.plot_model_graph()


if __name__ == '__main__':
    run()
