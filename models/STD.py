import torch
import torch.nn as nn

from configs.arguments import TrainingArguments
from models import loss
from models.decomposition import STDecomposition
from utils.normalize import Scaler


class STGDL(nn.Module):
    def __init__(self, config: TrainingArguments, supports, scaler: Scaler, device):
        super(STGDL, self).__init__()
        self.config = config
        self.device = device
        self.scaler = scaler
        self.num_nodes = config.num_nodes

        self.std = STDecomposition(config, supports, scaler)
        self.add_module(f"std", self.std)

        # self.weights = nn.Parameter(torch.zeros(size=(self.std.total, 1, self.config.num_nodes,
        #                                               self.num_nodes, self.config.output_len)))
        # self.register_parameter(f"gather_weights", self.weights)

        self.end_conv = nn.Conv2d(in_channels=self.config.c_hid, out_channels=self.config.c_out,
                                  kernel_size=(1, 1), bias=True)

        self.loss_func = loss.masked_mae_loss(config.mae_mask)

    def forward(self, x, trues):
        # x: NCVL
        # preds: tk*sk N L_o V V
        preds = self.std(x, trues, self.scaler)

        res = 0
        for i in range(self.std.total):
            res += preds[i]
        return self.scaler.inverse_transform(res)

    def calculate_loss(self, ys, preds, get_forward_loss=True):
        res = self.loss_func(preds, ys)
        if get_forward_loss:
            res += self.std.get_forward_loss()
        return res
