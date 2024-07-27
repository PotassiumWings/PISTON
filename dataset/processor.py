import os
import random

import numpy as np
import torch

from configs.arguments import TrainingArguments
from utils.normalize import get_scaler
from utils.device import DEVICE


class AbstractDataset(object):
    def __init__(self, config: TrainingArguments):
        self.config = config
        self.device = DEVICE
        self.train_iter, self.val_iter, self.test_iter = None, None, None
        self.supports = None
        self.scaler = None


class MyDataset(AbstractDataset):
    def __init__(self, config: TrainingArguments):
        super().__init__(config)
        self.data_dir = os.path.join("data", self.config.dataset_name)
        self.batch_size = config.batch_size

        self._load_adj()
        # TODO: 根据邻接矩阵先把data变成稀疏的，方便scaler处理等
        self._load_data()

    def _load_adj(self):
        adj_mx_filename = os.path.join(self.data_dir, "adj_mx.npz")
        self.supports = [np.load(adj_mx_filename)["adj_mx"]]
        # self.supports = [np.ones([self.config.num_nodes, self.config.num_nodes])]

    def _load_data(self):
        train_filename = os.path.join(self.data_dir, "train.npz")
        val_filename = os.path.join(self.data_dir, "val.npz")
        test_filename = os.path.join(self.data_dir, "test.npz")
        self.train_iter, self.train_data = self._get_iter(train_filename, define_scaler=True)
        self.val_iter, _ = self._get_iter(val_filename)
        self.test_iter, _ = self._get_iter(test_filename, shuffle=False)

    def _get_iter(self, filename, define_scaler=False, shuffle=True):
        data = np.load(filename)
        x, y = data['x'], data['y']
        # x, y: N L V V
        x = torch.Tensor(x)
        y = torch.Tensor(y)

        if define_scaler:
            self.scaler = get_scaler(self.config.scaler, x)

        if self.config.tradition_problem:
            x = self.scaler.transform(x)
        return MyDatasetIterator((x, y), self.config.batch_size, self.device, shuffle), x


class MyDatasetIterator(object):
    def __init__(self, data, batch_size, device, shuffle):
        self.batch_size = batch_size
        # x: (N, L, V, V)
        # y: (N, L', V, V)
        self.x: torch.Tensor = data[0]
        self.y: torch.Tensor = data[1]
        self.N = self.x.size(0)

        self.n_batches = int(round(self.N // batch_size))
        self.device = device

        self.batches = list(np.arange(0, self.N))
        if shuffle:
            self.shuffle()

        self.index = 0

    def shuffle(self):
        random.shuffle(self.batches)
        self.index = 0

    def _to_tensor(self, indexes):
        x = torch.FloatTensor([self.x[i].numpy() for i in indexes]).to(self.device)  # NLVV
        y = torch.FloatTensor([self.y[i].numpy() for i in indexes]).to(self.device)  # NLVV
        return x, y

    def __next__(self):
        if self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[
                      self.index * self.batch_size: min((self.index + 1) * self.batch_size, len(self.batches))]
            self.index += 1
            x, y = self._to_tensor(batches)
            return x, y.unsqueeze(1)

    def __iter__(self):
        return self

    def __len__(self):
        return self.n_batches
