import os

import auto_control_tools as act

folders = [
    r'../resources/forno/',
    r'../resources/mesa_giratoria/',
    r'../resources/nivel_tanque/',
]

di = {}

for folder in folders:
    di[folder] = {}
    for file in os.listdir(folder):
        di[folder][file] = {}
        file_path = os.path.join(folder, file)
        print(file_path)
        last = 'none'
        try:
            di[folder][file]['model'] = {}
            last = 'model'

            model = act.SmithModelIdentification.get_model(
                file_path,
                sample_time=1,
                ignore_delay_threshold=0,
                use_lin_filter=True,
                # linfilter_smoothness=5
            )
            di[folder][file]['model']['tf'] = str(model.tf_symbolic)
            di[folder][file]['model']['resp'] = model.view.get_model_step_response_data()

            model.view.print_tf()
            model.view.print_model_step_response_data()
            # model.view.plot_model_step_response_graph()

            di[folder][file]['controller'] = {}

            last = 'controller'
            controller = act.CohenCoonControllerAproximation.get_controller(model, act.PID)
            di[folder][file]['controller']['tf'] = str(controller.tf_symbolic)
            di[folder][file]['controller']['resp'] = controller.view.get_controller_step_response_data()

            controller.view.print_tf()
            controller.view.print_controller_step_response_data()
            # controller.view.plot_controller_step_response_graph(plot_model=False)

        except Exception as e:
            di[folder][file][last]['error'] = str(e)
            print(e)

print(di)
