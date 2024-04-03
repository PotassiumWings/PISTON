import numpy as np
from scripts_utils import *


f = open("logs_pq", "r", encoding="utf-8")
result = parse_f(f, pattern="python app.py", features=["--p", "--q"], default=["1", "1"], result_pattern="Test:")

res = [[0 for j in range(7)] for i in range(7)]
for i in range(7):
    if result.get(str(i)) is None:
        continue
    for j in range(7):
        # import pdb
        # pdb.set_trace()
        if result[str(i)].get(str(j)) is None:
            continue
        res[i][j] = result[str(i)][str(j)]["mae"]
for i in range(1, 7):
    for j in range(1, 7):
        print(res[i][j], end=',')
    print()
print(result)
