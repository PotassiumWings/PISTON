from datetime import datetime

import numpy as np


def get_timestamp(s):
    return int(datetime.strptime(s, time_format).timestamp())


time_format = "%Y/%m/%d %H:%M"
timestep = 60 * 60

data = "../data/NYCBike/201404.csv"
f = open(data, "r")
start_time = get_timestamp("2014/4/1 0:00")
id_to_idx = {}
idx = 0
lons, lats = [], []

datas = []
max_tim = -1

for line in f:
    d = line.split(",")
    duration = d[0]
    if 'd' in duration:
        continue

    s_id = int(d[3])
    t_id = int(d[7])
    sx, sy = float(d[5]), float(d[6])
    tx, ty = float(d[9]), float(d[10])
    tim = get_timestamp(d[1]) - start_time

    if s_id not in id_to_idx:
        id_to_idx[s_id] = idx
        lons.append(sx)
        lats.append(sy)
        idx += 1

    if t_id not in id_to_idx:
        id_to_idx[t_id] = idx
        lons.append(tx)
        lats.append(ty)
        idx += 1

    datas.append([s_id, t_id, tim])
    max_tim = max(max_tim, tim)

T = (max_tim + timestep - 1) // timestep
V = idx
print(idx)
print(len(datas))
print(T)
print(T * V * V)

res = np.zeros((T, V, V))
# T * V * V -> N T V V

for s, t, tim in datas:
    s = id_to_idx[s]
    t = id_to_idx[t]
    res[tim // timestep][s][t] += 1

# res: T V V
train_index = int(0.7 * T)
val_index = int(0.8 * T)
train = res[:train_index]
val = res[train_index:val_index]
test = res[val_index:]

input_len = 24

for name in ["train", "val", "test"]:
    all_data = locals()[name]  # T V V -> N T' V V
    x = np.array([all_data[i:i + input_len] for i in range(len(all_data) - input_len - 1)])
    y = np.array([all_data[i + input_len:i + input_len + 1] for i in range(len(all_data) - input_len - 1)])

    np.savez_compressed(
        name + ".npz",
        x=x, y=y
    )
