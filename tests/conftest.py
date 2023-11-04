import random
import os

prev_path = os.getcwd()

# Get the directory of the current script (conftest.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Set the current working directory to the directory of conftest.py
os.chdir(current_dir)

MODEL_EXAMPLES = [
    [[1], [1, 2]],
    [[1], [1, 2, 3]],
    [[1, 2], [1, 2, 3]],
]

MODEL_TEST_CASES = [
    item for sublist in [
        [
            (item, tf_type) for item in MODEL_EXAMPLES
        ] for tf_type in ['list', 'control']
    ] for item in sublist
]

print(os.getcwd())

NO_DELAY_CONST_SAMPLE_TIME_IDENTIFICATION_CASES = [
    os.path.abspath(os.path.join(directory_path, file))
    for directory_path in [
        r"resources/no_delay_constant_sample_time_identification_cases/"
    ]
    for file in os.listdir(directory_path) if file.endswith('.csv')
]

NO_DELAY_CONST_SAMPLE_TIME_IDENTIFICATION_RESULTS = os.path.abspath(
    r"resources/no_delay_constant_sample_time_identification_cases/first_order_get_model_results.json"
)

FIRST_ORDER_IDENTIFICATION_TEST_CASES = [
    {  # Test kwargs
        "path": file,
        "sample_time": spec.get('sample_time'),
        "step_signal": spec.get('step_signal'),
        "ignore_delay_threshold": spec.get('sample_time'),
    }
    for resource_files, spec in [
        (NO_DELAY_CONST_SAMPLE_TIME_IDENTIFICATION_CASES, {
           "sample_time": 1
        }),
    ]
    for file in resource_files
]

CONTROLLER_GAIN_EXAMPLES = [  # KI, KP, KD
    [random.uniform(0.1, 20) * sub_item for sub_item in item] for item in [
        (0, 1, 0),
        (1, 1, 0),
        (1, 1, 1),
    ]
]

CONTROLLER_TEST_CASES = [
    (m, c) for m in MODEL_EXAMPLES for c in CONTROLLER_GAIN_EXAMPLES
]


os.chdir(prev_path)
