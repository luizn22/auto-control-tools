import random

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
