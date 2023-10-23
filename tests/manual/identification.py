import os


def run():
    import auto_control_tools as act
    pth = r'C:\Users\luiz\PycharmProjects\auto-control-tools\tests\manual'
    m = act.NishikawaModelIdentification.get_model(os.path.join(pth, 'data_input.csv'))
    m.view.plot_model_graph()
    m.view.print_model_data()


if __name__ == '__main__':
    run()
