import torch
import torch.nn as nn


class PredictionHead(nn.Module):
    def __init__(self, d_model, num_nodes, c_out, output_len, traditional):
        super(PredictionHead, self).__init__()
        self.num_nodes = num_nodes
        self.c_out = c_out
        self.output_len = output_len
        self.d_model = d_model

        self.traditional = traditional
        if traditional:
            self.linear = nn.Conv2d(in_channels=d_model * output_len,
                                    out_channels=c_out * output_len,
                                    kernel_size=(1, 1), bias=True)
        else:
            self.linear = nn.Conv2d(in_channels=d_model * output_len,
                                    out_channels=num_nodes * c_out * output_len,
                                    kernel_size=(1, 1), bias=True)

    def forward(self, x):
        # x: N V Lo C -> N V 1 LoC
        x = x.reshape(-1, self.num_nodes, 1, self.output_len * self.d_model)

        # N V 1 LoC -> N LoC V 1
        x = x.permute(0, 3, 1, 2)

        # N (V)C'Lo V 1
        x = self.linear(x)

        if self.traditional:
            x = x.reshape(-1, self.c_out, self.output_len, self.num_nodes)
            x = torch.einsum('nclv->nlvc', x)
        else:
            x = x.reshape(-1, self.num_nodes, self.c_out, self.output_len, self.num_nodes)
            x = torch.einsum('nuclv->nclvu', x)

        # x: N C Lo V V
        return x
