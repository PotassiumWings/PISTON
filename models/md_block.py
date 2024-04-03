import torch.nn as nn

import torch
import logging
from configs.arguments import TrainingArguments
from configs.GraphWavenet_configs import GraphWavenetConfig
from configs.MSDR_configs import MSDRConfig
from configs.MTGNN_configs import MTGNNConfig
from configs.STSSL_configs import STSSLConfig
from configs.STGCN_configs import STGCNConfig
from configs.GEML_configs import GEMLConfig
from configs.CSTN_configs import CSTNConfig
from models.GraphWavenet.GraphWavenet import GraphWavenet
from models.MSDR.MSDR import MSDR
from models.MTGNN.MTGNN import MTGNN
from models.STGCN.STGCN import STGCN
from models.STSSL.STSSL import STSSL
from models.GEML.GEML import GEML
from models.CSTN.CSTN import CSTN
from models.abstract_st_encoder import AbstractSTEncoder


class MDBlock(nn.Module):
    def __init__(self, config: TrainingArguments, supports: list, temporal_index, spatio_index, st_encoder, scaler):
        super(MDBlock, self).__init__()
        self.conv: AbstractSTEncoder

        origin_config = config

        config = globals()[st_encoder + "Config"]()
        for k, v in origin_config:
            # if k in origin_config.__fields__:
            config.__setattr__(k, v)

        self.adj_conv = None
        self.adj_conv_back = None
        if not config.is_od_model:
            self.adj_conv = nn.Parameter(torch.randn(config.num_nodes, config.c_hid), requires_grad=True)
            self.register_parameter(f"{temporal_index}_{spatio_index}_conv", self.adj_conv)
            self.adj_conv_back = nn.Parameter(torch.randn(config.c_hid, config.num_nodes), requires_grad=True)
            self.register_parameter(f"{temporal_index}_{spatio_index}_conv2", self.adj_conv_back)

        origin_input_len = config.input_len
        if config.p > 1:
            config.input_len //= 2
        for i in range(min(temporal_index + 1, config.p - 1)):
            origin_input_len //= 2
        self.padding_len = config.input_len - origin_input_len

        logging.info(f"\t{temporal_index} {spatio_index} "
                     f"input_len={config.input_len} padding_len={self.padding_len}")
        config.c_in = config.c_out = config.c_hid
        config.st_encoder = st_encoder
        # in:   N C_hid V L()
        # out:  N C_hid V L_out

        self.conv = globals()[config.st_encoder](config, supports, scaler)

    def forward(self, x, subgraph, trues):
        # x: N V V L
        input_x = x
        if self.padding_len > 0:
            input_x = nn.functional.pad(input_x, (self.padding_len, 0, 0, 0))

        is_od = self.adj_conv is None

        # model is not OD model
        if not is_od:
            input_x = torch.einsum("nvwl,wd->nvdl", (input_x, self.adj_conv))  # N V C_h L
            # N C_h V L
            input_x = input_x.permute(0, 2, 1, 3)

        # N C_h V L_o
        # N V V L_o
        y = self.conv(input_x, subgraph, trues)

        if not is_od:
            # N V C_h L_o
            y = y.permute(0, 2, 1, 3)
            # N V V L_o
            y = torch.einsum("nvcl,cw->nvwl", (y, self.adj_conv_back))

        return y

    def get_embedding(self):
        return self.conv.get_embedding()

    def get_forward_loss(self):
        return self.conv.get_forward_loss()
