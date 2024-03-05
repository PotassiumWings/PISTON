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
    def __init__(self, config: TrainingArguments, supports):
        super(STDecomposition, self).__init__()
        # self.adj = supports[0]
        self.tk = config.p
        self.sk = config.q
        self.num_nodes = config.num_nodes
        self.total = self.tk * self.sk
        self.node_emb = []
        self.mds = []

        for i in range(self.tk):
            for j in range(self.sk):
                node_emb = nn.Parameter(torch.randn(self.num_nodes, self.num_nodes), requires_grad=True)
                self.register_parameter(f"{i}_{j}_node_emb", node_emb)
                self.node_emb.append(node_emb)

                st_encoder = ["STGCN", "MTGNN", "GraphWavenet"][random.randint(0, 2)]
                logging.info(f"{i} {j} ~ {st_encoder}")

                md_block = MDBlock(copy.deepcopy(config), supports, i, j, st_encoder)
                self.mds.append(md_block)
                self.add_module(f"{i}_{j}_md", md_block)

    def forward(self, x):
        x = self.decomposition(x)

        pred = []
        for i in range(self.tk):
            # N C V V l
            for j in range(self.sk):
                pred.append(self.mds[i * self.sk + j](x[i][j], self.get_adj(self.node_emb[i * self.sk + j])))
            # pred.append(sub_preds)
        # tksk N C_h V L
        return pred

    def decomposition(self, x):
        # x: N L V V -> N 1 V V L
        # x -> tk
        x = x.permute(0, 2, 3, 1)
        x = x.unsqueeze(1)

        wavelet = pywt.Wavelet("haar")
        x = ptwt.wavedec(x, wavelet, mode='zero', level=self.tk - 1)
        x = x[::-1]

        # x: [y1, y2, ..., ytk], y NCVVl
        res = []
        for i in range(self.tk):
            u, sig, v = torch.svd(x[i].permute(0, 1, 4, 2, 3))  # NClVV
            y = []
            for j in range(self.sk - 1):
                ui, sigi, vi = u[..., j:j + 1], sig[..., j:j + 1], v[..., j:j + 1]
                # mat: NClVV
                mat = torch.matmul(torch.matmul(ui, torch.diag_embed(sigi)), vi.transpose(-2, -1))
                # N 2C l V
                y.append(self.get_flows(mat))

            # residual term
            mat = torch.matmul(torch.matmul(u[..., self.sk:], torch.diag_embed(sig[..., self.sk:])),
                               v[..., self.sk:].transpose(-2, -1))
            y.append(self.get_flows(mat))

            res.append(y)
        # res: tk sk  N 2C l V
        return res

    def get_adj(self, node_emb):
        return F.softmax(F.relu(torch.mm(node_emb, node_emb.transpose(0, 1))), dim=1)

    def get_flows(self, mat):
        # mat: NClVV
        # DEBUG 2: inflow outflow
        inflow = mat.sum(-2)
        outflow = mat.sum(-1)

        # N 2C l V
        flows = torch.cat([inflow, outflow], dim=1)
        return flows.permute(0, 1, 3, 2)
