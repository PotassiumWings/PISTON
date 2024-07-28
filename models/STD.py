import logging

import numpy as np
import torch
import torch.nn as nn

from configs.arguments import TrainingArguments
from models import loss
from models.decomposition_block import DecompositionBlock
from models.freq_attention import CorrelationEncoder
from models.prediction_head import PredictionHead
from models.self_supervise_heads import ContrastiveHead, RecoverHead


class STDOD(nn.Module):
    def __init__(self, config: TrainingArguments, supports, scaler):
        super(STDOD, self).__init__()
        self.supports = supports
        self.scaler = scaler
        self.loss = loss.masked_mae_loss(config.mae_mask)

        input_len = config.input_len
        if config.p > 1:
            input_len //= 2

        self.transform_start_block = nn.Linear(config.origin_c_in, config.num_nodes) \
            if config.tradition_problem else None

        logging.info("Decomposition Block")
        self.decomposition_block = DecompositionBlock(input_len=input_len, sk=config.q, tk=config.p,
                                                      n=config.num_nodes, random_svd_k=config.random_svd_k,
                                                      rsvd=config.rsvd, use_rsvd_emb=config.use_rsvd_emb,
                                                      output_dim=config.d_model // 2)

        self.recover = config.recover
        self.do_mask = config.mask
        if config.recover:
            assert False, "Recover has been disabled."
            if config.mask:
                self.mask = Mask(mask_percent=config.mask_percent)
            self.recover_head = RecoverHead(sk=config.q, tk=config.p, num_nodes=config.num_nodes,
                                            d_model=config.d_encoder)

        logging.info("Correlation Encoder")
        self.encoder = CorrelationEncoder(input_len=input_len, num_nodes=config.num_nodes, sk=config.q,
                                          tk=config.p, layers=config.layers, n_heads=config.n_head,
                                          adp_emb=config.adp_emb, d_model=config.d_model, d_ff=config.d_ff,
                                          d_encoder=config.d_encoder, d_encoder_ff=config.d_encoder,  # same
                                          dropout=config.dropout, support_len=len(supports), order=config.order,
                                          d_out=config.d_encoder, output_len=config.output_len, only_1=config.only_1)
        logging.info("Prediction Head")
        self.prediction_head = PredictionHead(num_nodes=config.num_nodes,
                                              d_model=config.d_encoder, c_out=config.c_out,
                                              output_len=config.output_len, traditional=config.tradition_problem)
        self.contra = config.contra
        if config.contra:
            self.contrastive_head = ContrastiveHead(tk=config.p, sk=config.q, num_nodes=config.num_nodes,
                                                    d_model=config.d_encoder, input_len=input_len)

        self.contra_loss = 0
        self.recover_loss = 0
        self.regular_loss = 0
        self.use_dwa = config.use_dwa
        self.last_loss = None
        self.loss_weights = np.array([config.loss_lamb, config.recover_lamb, config.contra_lamb])

    def disable_ssl(self):
        self.contra = self.recover = False

    def forward(self, x):
        self.recover_loss = self.contra_loss = 1e-10

        if self.transform_start_block is not None:
            x = self.transform_start_block(x)  # N L V C -> N L V V

        # x: N L V V
        # decomposed: N V L tk sk C
        decomposed = self.decomposition_block(x)
        if self.transform_start_block is None:
            decomposed = self.scaler.transform(decomposed)

        # embedding: N V L tk sk C
        embedding = self.encoder(decomposed, self.supports)

        if self.recover:
            masked_decomposed = decomposed
            if self.do_mask:
                masked_decomposed = self.mask(decomposed)

            embedding_masked = self.encoder(masked_decomposed, self.supports)
            # recover: tk*sk N V V L
            recover = self.recover_head(embedding_masked)
            self.recover_loss = loss.mae_torch(recover, decomposed)
            # self.recover_loss = loss.mae_torch(self.scaler.inverse_transform(recover),
            #                                    self.scaler.inverse_transform(decomposed),
            #                                    1e-10)

        if self.contra:
            self.contra_loss = self.contrastive_head(embedding)

        # pred: N C_out L V V
        pred = self.prediction_head(embedding)

        return self.scaler.inverse_transform(pred)

    def calculate_loss(self, pred, true, update_dwa=False):
        self.regular_loss = self.loss(pred.flatten(), true.flatten())
        losses = [self.regular_loss, self.recover_loss, self.contra_loss]

        if self.use_dwa and self.last_loss is not None and update_dwa:
            self.loss_weights = dwa(self.last_loss, losses)

        if update_dwa:
            self.last_loss = losses

        return self.loss_weights[0] * losses[0] + self.loss_weights[1] * losses[1] + self.loss_weights[2] * losses[2]

    def get_loss_weights(self):
        return self.loss_weights

    def get_contra_loss(self):
        try:
            return self.contra_loss.item()
        except AttributeError:
            return 0

    def get_recover_loss(self):
        try:
            return self.recover_loss.item()
        except AttributeError:
            return 0


def dwa(L_old, L_new, T=2):
    L_old = torch.Tensor(L_old)
    L_new = torch.Tensor(L_new)
    N = len(L_old)
    r = L_old / L_new
    if L_new[2] < 1e-4:
        r[2] = -100
    w = N * torch.softmax(r / T, dim=0)
    return w.numpy()
