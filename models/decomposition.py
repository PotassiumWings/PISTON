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
        self.conv = []

        for i in range(self.tk):
            for j in range(self.sk):
                node_emb = nn.Parameter(torch.randn(self.num_nodes, config.node_emb), requires_grad=True)
                self.register_parameter(f"{i}_{j}_node_emb", node_emb)
                self.node_emb.append(node_emb)

                conv = nn.Parameter(torch.randn(self.num_nodes, config.c_hid), requires_grad=True)
                self.register_parameter(f"{i}_{j}_conv", conv)
                self.conv.append(conv)

                # st_encoder = ["STGCN", "MTGNN", "STSSL"][random.randint(0, 2)]
                st_encoder = "STGCN"
                logging.info(f"{i} {j} ~ {st_encoder}")

                md_block = MDBlock(copy.deepcopy(config), supports, i, j, st_encoder)
                self.mds.append(md_block)
                self.add_module(f"{i}_{j}_md", md_block)

    def forward(self, x):
        x = self.decomposition(x)

        pred = []
        for i in range(self.tk):
            # N V V l
            for j in range(self.sk):
                idx = i * self.sk + j
                # N V V l -> N V C' L
                # input_x = torch.mm(x[idx], self.conv[idx])  # N C_hid V L
                input_x = torch.einsum("nvwl,wd->nvdl", (x[idx], self.conv[idx]))  # N V C_h L
                # N C_h V L
                input_x = input_x.permute(0, 2, 1, 3)
                # N C_h V L_o
                y = self.mds[idx](input_x, [self.get_adj(self.node_emb[idx])])
                # N V C_h L_o
                y = y.permute(0, 2, 1, 3)
                # N V V L_o
                y = torch.einsum("nvcl,cw->nvwl", (y, self.conv[idx].t()))
                # y = torch.mm(y, self.conv[idx].t())
                # N L_o V V
                pred.append(y.permute(0, 3, 1, 2))
            # pred.append(sub_preds)
        # tksk N C_h V L
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
            u, sig, v = torch.svd(x[i].permute(0, 2, 3, 1))  # NlVV
            # logging.info("svd end")
            y = []
            for j in range(self.sk - 1):
                ui, sigi, vi = u[..., j:j + 1], sig[..., j:j + 1], v[..., j:j + 1]
                # mat: NlVV
                mat = torch.matmul(torch.matmul(ui, torch.diag_embed(sigi)), vi.transpose(-2, -1))
                # N V V l
                y.append(mat.permute(0, 2, 3, 1))

            # residual term
            mat = torch.matmul(torch.matmul(u[..., self.sk:], torch.diag_embed(sig[..., self.sk:])),
                               v[..., self.sk:].transpose(-2, -1))
            y.append(mat.permute(0, 2, 3, 1))

            res.append(y)
        # res: tk sk  N V V l
        # x: tk NCVVl
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
