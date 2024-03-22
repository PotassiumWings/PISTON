import torch.nn as nn

import logging
from configs.arguments import TrainingArguments
from configs.GraphWavenet_configs import GraphWavenetConfig
from configs.MSDR_configs import MSDRConfig
from configs.MTGNN_configs import MTGNNConfig
from configs.STSSL_configs import STSSLConfig
from configs.STGCN_configs import STGCNConfig
from models.GraphWavenet.GraphWavenet import GraphWavenet
from models.MSDR.MSDR import MSDR
from models.MTGNN.MTGNN import MTGNN
from models.STGCN.STGCN import STGCN
from models.STSSL.STSSL import STSSL
from models.abstract_st_encoder import AbstractSTEncoder


class MDBlock(nn.Module):
    def __init__(self, config: TrainingArguments, supports: list, temporal_index, spatio_index, st_encoder):
        super(MDBlock, self).__init__()
        self.conv: AbstractSTEncoder

        if st_encoder == "STGCN":
            config = STGCNConfig()
        elif st_encoder == "GraphWavenet":
            config = GraphWavenetConfig()
        elif st_encoder == "MTGNN":
            config = MTGNNConfig()
        elif st_encoder == "STSSL":
            config = STSSLConfig()
        elif st_encoder == "MSDR":
            config = MSDRConfig()

        for i in range(min(temporal_index + 1, config.p - 1)):
            config.input_len //= 2
        logging.info(f"\t{temporal_index} {spatio_index} input_len={config.input_len}")
        config.c_in = config.c_out = config.c_hid
        config.st_encoder = st_encoder
        # in:   N C_hid V L()
        # out:  N C_hid V L_out

        if config.st_encoder == "STGCN":
            self.conv = STGCN(config, supports)
        elif config.st_encoder == "GraphWavenet":
            self.conv = GraphWavenet(config, supports)
        elif config.st_encoder == "STSSL":
            self.conv = STSSL(config, supports)
        elif config.st_encoder == "MSDR":
            self.conv = MSDR(config, supports)
        elif config.st_encoder == "MTGNN":
            self.conv = MTGNN(config, supports)
        else:
            raise NotImplementedError(f"ST Encoder {config.st_encoder} not implemented.")

    def forward(self, x, subgraph):
        # x: (batch_size, c_in, num_nodes, input_len)
        pred = self.conv(x, subgraph)
        return pred

    def get_embedding(self):
        return self.conv.get_embedding()

    def get_forward_loss(self):
        return self.conv.get_forward_loss()
