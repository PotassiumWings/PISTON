import copy

import torch
import logging
import pywt
import ptwt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from configs.arguments import TrainingArguments
from models import loss
from models.decomposition_batch import DecompositionBatch
from datetime import datetime
from models.decomposition import STDecomposition
from utils.normalize import Scaler


class STGDL(nn.Module):
    def __init__(self, config: TrainingArguments, supports, scaler: Scaler, device):
        super(STGDL, self).__init__()
        self.config = config
        self.device = device
        self.scaler = scaler
        self.num_nodes = config.num_nodes

        self.use_model_pool = config.use_model_pool
        self.stds = []
        self.used_time = None
        self.decomposition_batch = DecompositionBatch(config, config.p, config.q)
        if self.use_model_pool:
            model_pool = config.model_pool
            self.models = model_pool.split(",")
            self.lm = len(self.models)
            for i in range(self.lm):
                temp_config = copy.deepcopy(config)
                temp_config.st_encoder = self.models[i]
                temp_config.is_od_model = config.model_pool_od[i] == '1'
                std = STDecomposition(temp_config, supports, scaler, self.decomposition_batch)
                self.stds.append(std)
                self.add_module(f"std_{i}", std)
            self.used_time = np.zeros(self.lm + 1)
        else:
            std = STDecomposition(config, supports, scaler, self.decomposition_batch)
            self.stds.append(std)
            self.add_module(f"std", std)

        self.end_conv = nn.Conv2d(in_channels=self.config.c_hid, out_channels=self.config.c_out,
                                  kernel_size=(1, 1), bias=True)

        self.loss_func = loss.masked_mae_loss(config.mae_mask)

    def forward(self, x, trues):
        # x: NCVL
        # preds: tk*sk N L_o V V
        self.decomposition_batch.init_batch()

        if self.use_model_pool:
            res = []

            start_time = datetime.now()
            self.decomposition_batch.get_data(x)
            self.used_time[-1] += (datetime.now() - start_time).total_seconds()

            for i in range(self.lm):
                # logging.info(f"Proceeding {self.models[i]}...")
                start_time = datetime.now()
                preds = self.stds[i](x, trues)
                self.used_time[i] += (datetime.now() - start_time).total_seconds()

                for j in range(len(preds)):
                    res.append(preds[j])
            # res: tkskm N L_o V V
            res = torch.stack(res) / (self.config.p * self.config.q * self.lm)
        else:
            res = 0
            preds = self.stds[0](x, trues)
            for i in range(len(preds)):
                res += preds[i]

        return self.scaler.inverse_transform(res)

    def print_time_consumed(self):
        if not self.use_model_pool:
            return
        total_time = sum(self.used_time)
        partial_time = self.used_time / total_time * 100
        s = "Time consumption ratio: "
        for i in range(self.lm):
            s += "({}: {:.2f}%)".format(self.models[i], partial_time[i])
        s += "(Decom: {:.2f}%)".format(partial_time[-1])
        logging.info(s)

    def calculate_loss(self, ys, preds, get_forward_loss=True):
        if self.use_model_pool:
            ys = ys.unsqueeze(0)
            ys = ys.broadcast_to(preds.shape)
        res = self.loss_func(preds, ys)
        if get_forward_loss:
            for i in range(len(self.stds)):
                res += self.stds[i].get_forward_loss()
        return res


class DecompositionBlock(nn.Module):
    def __init__(self, input_len, sk, tk, n, k):
        super(DecompositionBlock, self).__init__()
        self.tk = tk
        self.sk = sk
        self.n = n
        self.k = k
        self.input_len = input_len

        self.p = nn.Parameter(torch.randn(self.n, self.k), requires_grad=True)
        self.register_parameter(f"p", self.p)

    def svd(self, x):
        # x: N L V V
        z = x @ self.p  # z: N L V k
        q, _ = torch.qr(z)  # q: N L V k
        y = q.t @ x  # y: N L k V
        u, sig, v = torch.svd(y)
        u = q @ u
        return u, sig, v

    def forward(self, x):
        # x: N L V V -> N V V L
        # x -> tk
        x = x.permute(0, 2, 3, 1)

        wavelet = pywt.Wavelet("haar")
        x = ptwt.wavedec(x, wavelet, mode='zero', level=self.tk - 1)
        x = x[::-1]  # high -> low

        # x: [y1, y2, ..., ytk], y NVVl
        res = []
        for i in range(self.tk):
            if self.sk == 1:
                res.append(x[i])
                continue

            u, sig, v = self.svd(x[i].permute(0, 3, 1, 2))  # NlVV
            self.lambdas.append(sig)

            for j in range(self.sk - 1):
                ui, sigi, vi = u[..., j:j + 1], sig[..., j:j + 1], v[..., j:j + 1]
                # mat: NlVV
                mat = torch.matmul(torch.matmul(ui, torch.diag_embed(sigi)), vi.transpose(-2, -1))
                # N V V l
                res.append(mat.permute(0, 2, 3, 1))

            # residual term
            mat = torch.matmul(torch.matmul(u[..., self.sk:], torch.diag_embed(sig[..., self.sk:])),
                               v[..., self.sk:].transpose(-2, -1))
            mat = mat.permute(0, 2, 3, 1)  # N V V l
            res.append(mat)
        # padding
        for i in range(len(res)):
            res[i] = nn.functional.pad(res[i], (self.input_len - res[i].size(3), 0, 0, 0))
        # res: tk*sk  N V V l
        return torch.stack(res)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, f"d_model {d_model} should be divided by n_heads {n_heads}."
        d_qkv = d_model // n_heads
        self.WQ = nn.Linear(d_model, d_qkv * n_heads)
        self.WK = nn.Linear(d_model, d_qkv * n_heads)
        self.WV = nn.Linear(d_model, d_qkv * n_heads)
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_qkv = d_qkv
        self.fc = nn.Linear(d_qkv * n_heads, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff),
                                nn.GELU(),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model))

    def forward(self, x):
        bs = x.size(0)
        # x: ? L Dm
        residual = x
        # Multi-Head Attention
        q = self.WQ(x).view(bs, -1, self.n_heads, self.d_qkv).permute(0, 2, 1, 3)  # ? H L D
        k = self.WK(x).view(bs, -1, self.n_heads, self.d_qkv).permute(0, 2, 1, 3)  # ? H L D
        v = self.WV(x).view(bs, -1, self.n_heads, self.d_qkv).permute(0, 2, 1, 3)  # ? H L D
        attention = q @ k.transpose(1, 2) / self.d_qkv ** 0.5  # ? H L L
        attention = self.dropout(F.softmax(attention, dim=-1))  # ? H L L
        result = attention @ v  # ? H L D
        result = result.transpose(1, 2).view(bs, -1, self.d_model)  # ? L Dm
        result = self.dropout(self.fc(result))

        # Add & Norm
        result += residual
        result = self.layer_norm(result)

        # Feed-Forward
        ff_res = self.ff(result)

        # Add & Norm
        result += self.dropout(ff_res)
        result = self.layer_norm(result)
        return result


class TemporalAttention(nn.Module):
    def __init__(self):
        super(TemporalAttention, self).__init__()


    def forward(self, x):
        pass


class Encoder(nn.Module):
    def __init__(self, input_len, tk, sk, layers, n, d_model):
        super(Encoder, self).__init__()
        self.input_len = input_len
        self.layers = layers
        self.start_linear = nn.Linear(n, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(tk, sk, d_model))
        self.temporal_attention = TemporalAttention()

    def forward(self, x):
        # x: tk*sk N V Vd L -> tk sk N V Vd L -> N V L tk sk Vd
        x = x.reshape(self.tk, self.sk, -1, self.n, self.n, self.input_len)
        x = x.permute(2, 3, 5, 0, 1, 4)
        # x: N V L tk sk C
        x = self.start_linear(x)
        x = x + self.position_embedding

        for i in self.layers:
            x = self.temporal_attention(x)


class PredictionHead(nn.Module):
    def __init__(self):
        super(PredictionHead, self).__init__()
        pass


class SelfSuperviseHead(nn.Module):
    def __init__(self):
        super(SelfSuperviseHead, self).__init__()
        pass


class STDOD(nn.Module):
    def __init__(self, config: TrainingArguments, supports, scaler: Scaler, device):
        super(STDOD, self).__init__()

        input_len = config.input_len
        if config.p > 1:
            input_len //= 2

        self.decomposition_block = DecompositionBlock()
        self.encoder = Encoder()
        self.prediction_head = PredictionHead()
        self.self_supervise_head = SelfSuperviseHead()

    def forward(self, x):
        # p q N C V L
        decomposed = self.decomposition_block(x)
        embedding = self.encoder(decomposed)
        pred = self.prediction_head(embedding)
        reconstruct = self.self_supervise_head(embedding)
