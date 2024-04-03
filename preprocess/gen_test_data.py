import numpy as np
import os


if not os.path.exists("../data/test"):
    os.mkdir("../data/test")

input_len = 48
output_len = 1
num_nodes = 20

train_x = np.random.rand(32 * 7, input_len, num_nodes, num_nodes) + 5
train_y = np.random.rand(32 * 7, output_len, num_nodes, num_nodes) + 5
val_x = np.random.rand(32, input_len, num_nodes, num_nodes) + 5
val_y = np.random.rand(32, output_len, num_nodes, num_nodes) + 5
test_x = np.random.rand(64, input_len, num_nodes, num_nodes) + 5
test_y = np.random.rand(64, output_len, num_nodes, num_nodes) + 5
adj = np.eye(num_nodes)

np.savez_compressed("../data/test/train.npz", x=train_x, y=train_y)
np.savez_compressed("../data/test/val.npz", x=val_x, y=val_y)
np.savez_compressed("../data/test/test.npz", x=test_x, y=test_y)
np.savez_compressed("../data/test/adj_mx.npz", adj_mx=adj)
