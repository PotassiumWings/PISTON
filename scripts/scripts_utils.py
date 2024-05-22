import re
from collections import defaultdict


def ensure_4d(dic, m, n, s, t):
    if m not in dic:
        dic[m] = {}
    if n not in dic[m]:
        dic[m][n] = {}
    if s not in dic[m][n]:
        dic[m][n][s] = {}


def ensure_3d(dic, m, n, s):
    if m not in dic:
        dic[m] = {}
    if n not in dic[m]:
        dic[m][n] = {}


def ensure_2d(dic, m, n):
    if m not in dic:
        dic[m] = {}


def parse_f(f, pattern: str, features: list, default: list, result_pattern: str):
    def nested_dict():
        return defaultdict(nested_dict)

    def insert_result(nested_map, keys, value):
        current_dict = nested_map
        for key in keys[:-1]:
            current_dict = current_dict.setdefault(key, {})
        current_dict[keys[-1]] = value

    result = nested_dict()

    current_features = default.copy()
    for line in f:
        # case 1: python app.py --arguments
        if pattern in line:
            if not "GraphWavenet" in line:
                continue
            current_features = default.copy()
            args = line.strip("\n").split(" ")
            for i in range(len(args)):
                for j in range(len(features)):
                    if args[i] == features[j]:
                        current_features[j] = args[i + 1]
        elif result_pattern in line:
            rmse, mae, mape = re.findall("rmse ([\\d|.]+), mae ([\\d|.]+), mape ([\\d|.]+)", line)[0]
            temp_result = {"rmse": float(rmse), "mae": float(mae), "mape": float(mape)}
            insert_result(result, current_features, temp_result)
    return result
