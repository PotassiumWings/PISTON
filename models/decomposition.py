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
    def __init__(self, config: TrainingArguments, supports, scaler):
        super(STDecomposition, self).__init__()
        # self.adj = supports[0]
        self.tk = config.p
        self.sk = config.q
        self.num_nodes = config.num_nodes
        self.total = self.tk * self.sk
        self.node_emb = []
        self.mds = []
        self.conv = []

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
        x = self.decomposition(x)

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

    def decomposition(self, x):
        # x: N L V V -> N V V L
        # x -> tk
        x = x.permute(0, 2, 3, 1)

        wavelet = pywt.Wavelet("haar")
        # logging.info("wavedec start")
        x = ptwt.wavedec(x, wavelet, mode='zero', level=self.tk - 1)
        # logging.info("wavedec end")
        x = x[::-1]

        # x: [y1, y2, ..., ytk], y NVVl
        res = []
        for i in range(self.tk):
            if self.sk == 1:
                res.append(x[i])
                continue

            # logging.info("svd start")
            u, sig, v = torch.svd(x[i].permute(0, 3, 1, 2))  # NlVV
            # logging.info("svd end")
            for j in range(self.sk - 1):
                ui, sigi, vi = u[..., j:j + 1], sig[..., j:j + 1], v[..., j:j + 1]
                # mat: NlVV
                mat = torch.matmul(torch.matmul(ui, torch.diag_embed(sigi)), vi.transpose(-2, -1))
                # N V V l
                res.append(mat.permute(0, 2, 3, 1))

            # residual term
            mat = torch.matmul(torch.matmul(u[..., self.sk:], torch.diag_embed(sig[..., self.sk:])),
                               v[..., self.sk:].transpose(-2, -1))
            res.append(mat.permute(0, 2, 3, 1))
        # res: tk sk  N V V l
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

    def get_forward_loss(self):
        res = 0
        for i in range(len(self.mds)):
            res += self.mds[i].get_forward_loss()
        return res
