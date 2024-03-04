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

        self.std = STDecomposition(config, supports)
        self.weights = nn.Parameter(torch.zeros(size=(self.std.total, 1, self.config.c_out,
                                                      self.num_nodes, self.config.output_len)),
                                    requires_grad=True)

        self.loss_func = loss.masked_mae_loss(config.mae_mask)

    def forward(self, x):
        # x: NCVL
        # preds: tk*sk N C_h V L
        preds = self.std(x)

        res = 0
        for i in range(self.std.total):
            res += self.weights[i] * preds[i]
        return res
