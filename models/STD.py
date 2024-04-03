import copy

import torch
import logging
import torch.nn as nn
import numpy as np

from configs.arguments import TrainingArguments
from models import loss
from datetime import datetime
from models.decomposition import STDecomposition
from utils.normalize import Scaler


class STGDL(nn.Module):
    def __init__(self, config: TrainingArguments, supports, scaler: Scaler, device):
        super(STGDL, self).__init__()
        self.config = config
        self.device = device
        self.scaler = scaler
        self.num_nodes = config.num_nodes

        self.use_model_pool = config.use_model_pool
        self.stds = []
        self.used_time = None
        if self.use_model_pool:
            model_pool = config.model_pool
            self.models = model_pool.split(",")
            self.lm = len(self.models)
            for i in range(self.lm):
                temp_config = copy.deepcopy(config)
                temp_config.st_encoder = self.models[i]
                temp_config.is_od_model = config.model_pool_od[i] == '1'
                std = STDecomposition(temp_config, supports, scaler)
                self.stds.append(std)
                self.add_module(f"std_{i}", std)
            self.used_time = np.zeros(self.lm)
        else:
            std = STDecomposition(config, supports, scaler)
            self.stds.append(std)
            self.add_module(f"std", std)

        self.end_conv = nn.Conv2d(in_channels=self.config.c_hid, out_channels=self.config.c_out,
                                  kernel_size=(1, 1), bias=True)

        self.loss_func = loss.masked_mae_loss(config.mae_mask)

    def forward(self, x, trues):
        # x: NCVL
        # preds: tk*sk N L_o V V
        if self.use_model_pool:
            res = []
            for i in range(self.lm):
                # logging.info(f"Proceeding {self.models[i]}...")
                start_time = datetime.now()
                preds = self.stds[i](x, trues)
                self.used_time[i] += (datetime.now() - start_time).total_seconds()

                for j in range(len(preds)):
                    res.append(preds[j])
            # res: tkskm N L_o V V
            res = torch.stack(res) / (self.config.p * self.config.q * self.lm)
        else:
            res = 0
            preds = self.stds[0](x, trues)
            for i in range(len(preds)):
                res += preds[i]

        return self.scaler.inverse_transform(res)

    def print_time_consumed(self):
        total_time = sum(self.used_time)
        partial_time = self.used_time / total_time * 100
        s = "Time consumption ratio: "
        for i in range(self.lm):
            s += "({}: {:.2f}%)".format(self.models[i], partial_time[i])
        logging.info(s)

    def calculate_loss(self, ys, preds, get_forward_loss=True):
        if self.use_model_pool:
            ys = ys.unsqueeze(0)
            ys = ys.broadcast_to(preds.shape)
        res = self.loss_func(preds, ys)
        if get_forward_loss:
            for i in range(len(self.stds)):
                res += self.stds[i].get_forward_loss()
        return res
