import numpy as np

origin_data = np.load("../data/NYC-TOD/oddata.npy", allow_pickle=True, encoding='latin1')[()][0]

t, n, r, c = origin_data.shape
# t = 365 * 24 * 2

# 30min 步长求和聚合到 2h 步长
data = origin_data.reshape(t // 4, 4, n, n)
data = data.sum(1)
t //= 4

# train_index = int(t * 0.35)
# val_index = int(t * 0.4)
# test_index = int(t * 0.5)

train_index = int(t * 0.7)
val_index = int(t * 0.8)
test_index = int(t * 1)

train_data = data[:train_index, :, :]
val_data = data[train_index:val_index, :, :]
test_data = data[val_index:test_index, :, :]

timestep = 48  # 4day
horizon = 12


def get_data(origin_data):
    T, v, _ = origin_data.shape
    res_x = np.asarray([origin_data[i:i + timestep, :, :] for i in range(T - timestep - horizon - 1)])
    res_y = np.asarray([origin_data[i + timestep:i + timestep + 12, :, :] for i in range(T - timestep - horizon - 1)])
    print(res_x.shape, res_y.shape)
    return res_x, res_y


x, y = get_data(train_data)
np.savez_compressed("../data/NYC-TOD5/train.npz", x=x, y=y)
x, y = get_data(val_data)
np.savez_compressed("../data/NYC-TOD5/val.npz", x=x, y=y)
x, y = get_data(test_data)
np.savez_compressed("../data/NYC-TOD5/test.npz", x=x, y=y)
