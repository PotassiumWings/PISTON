import copy

import ptwt
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import logging
from torch.nn import Module

from configs.arguments import TrainingArguments
from models.md_block import MDBlock


class STDecomposition(Module):
    def __init__(self, config: TrainingArguments, supports, scaler, decomposition_batch=None):
        super(STDecomposition, self).__init__()
        # self.adj = supports[0]
        self.config = config
        self.tk = config.p
        self.sk = config.q
        self.num_nodes = config.num_nodes
        self.total = self.tk * self.sk
        self.node_emb = []
        self.mds = []
        self.conv = []

        self.decomposition_batch = decomposition_batch

        for i in range(self.tk):
            for j in range(self.sk):
                node_emb = nn.Parameter(torch.randn(self.num_nodes, config.node_emb), requires_grad=True)
                self.register_parameter(f"{i}_{j}_node_emb", node_emb)
                self.node_emb.append(node_emb)

                # st_encoder = ["STGCN", "MTGNN", "STSSL"][random.randint(0, 2)]
                st_encoder = config.st_encoder
                logging.info(f"{i} {j} ~ {st_encoder}")

                md_block = MDBlock(copy.deepcopy(config), supports, i, j, st_encoder, scaler)
                self.mds.append(md_block)
                self.add_module(f"{i}_{j}_md", md_block)

    def forward(self, x, trues):
        # N L V V -> N V V L
        x = self.decomposition_batch.get_data(x)

        pred = []
        for i in range(self.tk):
            # N V V l
            for j in range(self.sk):
                idx = i * self.sk + j
                # N V V l -> N V C' L / N V V L_o
                y = self.mds[idx](x[idx], [self.get_adj(self.node_emb[idx])], trues)
                # N L_o V V
                pred.append(y.permute(0, 3, 1, 2))
            # pred.append(sub_preds)
        # tksk N L_o V V
        return pred

    def get_adj(self, node_emb):
        return F.softmax(F.relu(torch.mm(node_emb, node_emb.transpose(0, 1))), dim=1)

    def get_forward_loss(self):
        res = 0
        for i in range(len(self.mds)):
            res += self.mds[i].get_forward_loss()
        return res
