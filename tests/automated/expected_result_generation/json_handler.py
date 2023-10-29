import json
import os
from typing import Tuple, Iterable


def save_result_set_to_json(path: str, name: str, in_out: Iterable[Tuple[dict, dict]]):
    """

    :param path: path to folder where json will be saved
    :param name: file name
    :param in_out: espects an iterable of tuples (input and output)
    :return:
    """

    if name.endswith('.json') is False:
        name = f"{name}.json"

    path = os.path.join(path, name)

    j_li = [
        {
            'in': inp,
            'out': {
                key: value
                for key, value in out.items()
                if is_serializable(value)
            }
        } for inp, out in in_out
    ]

    with open(path, 'w') as json_file:
        json.dump(j_li, json_file)


def compare_to_json(inputs: dict, outputs: dict, json_path: str):
    with open(json_path, 'r') as json_file:
        j = json.load(json_file)

    def search():
        for item in j:
            if item.get('in') == inputs:
                return item.get('out')

    return search() == {k: v for k,v in outputs.items() if is_serializable(v)}


def is_serializable(obj):
    try:
        json.dumps(obj)
        return True
    except TypeError:
        return False
