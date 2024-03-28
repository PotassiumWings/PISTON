import torch
import torch.nn as nn

from configs.arguments import TrainingArguments


class AbstractSTEncoder(nn.Module):
    def __init__(self, config: TrainingArguments, gp_supports, scaler=None):
        super(AbstractSTEncoder, self).__init__()
        self.config = config
        self.scaler = scaler
        self.cache_embedding = None
        self.gp_supports = gp_supports  # supports using graph partitioning
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.forward_loss = 0

    def forward(self, x, adj, trues):
        raise NotImplementedError("ST Encoder forward(x, adj) is not implemented.")

    def get_forward_loss(self):
        # loss generated in forward()
        return self.forward_loss

    def get_embedding(self):
        raise NotImplementedError("get_embedding is not implemented.")
