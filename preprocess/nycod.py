import numpy as np


data = np.load("../data/NYC-TOD/oddata.npy", allow_pickle=True, encoding='latin1')[()][0]

t, n, r, c = data.shape

data = data.reshape(t, n, n)
train_index = int(t * 0.7)
val_index = int(t * 0.8)

train_data = data[:train_index, :, :]
val_data = data[train_index:val_index, :, :]
test_data = data[val_index:, :, :]

timestep = 12  # 1day


def get_data(origin_data):
    T, v, _ = origin_data.shape
    res_x = np.asarray([[origin_data[i:i+timestep, :, :]] for i in range(T - timestep - 1)])
    res_y = np.asarray([[origin_data[i+timestep:i+timestep+1, :, :]] for i in range(T - timestep - 1)])
    print(res_x.shape, res_y.shape)
    return res_x, res_y


x, y = get_data(train_data)
np.savez_compressed("../data/NYC-TOD2/train.npz", x=x, y=y)
x, y = get_data(val_data)
np.savez_compressed("../data/NYC-TOD2/val.npz", x=x, y=y)
x, y = get_data(test_data)
np.savez_compressed("../data/NYC-TOD2/test.npz", x=x, y=y)

