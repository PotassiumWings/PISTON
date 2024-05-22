import logging

import torch
import pywt
import ptwt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from configs.arguments import TrainingArguments
from models import loss
from utils.normalize import Scaler


class DecompositionBlock(nn.Module):
    def __init__(self, rsvd, input_len, sk, tk, n, random_svd_k):
        super(DecompositionBlock, self).__init__()
        self.rsvd = rsvd
        self.tk = tk
        self.sk = sk
        self.n = n
        self.k = random_svd_k
        self.input_len = input_len

        self.p = nn.Parameter(torch.randn(self.n, self.k), requires_grad=True)

    def svd(self, x):
        if not self.rsvd:
            u, sig, v = torch.svd(x)
            return u, sig, v

        # x: N L V V
        z = x @ self.p  # z: N L V k
        q, _ = torch.linalg.qr(z)  # q: N L V k
        y = q.transpose(-2, -1) @ x  # y: N L k V
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
            sum_val = torch.zeros_like(x[i])

            for j in range(self.sk - 1):
                ui, sigi, vi = u[..., j:j + 1], sig[..., j:j + 1], v[..., j:j + 1]
                # mat: NlVV
                mat = torch.matmul(torch.matmul(ui, torch.diag_embed(sigi)), vi.transpose(-2, -1)).permute(0, 2, 3, 1)
                sum_val += mat
                # N V V l
                res.append(mat)

            # residual term
            res.append(x[i] - sum_val)
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
        q = self.WQ(x).reshape(bs, -1, self.n_heads, self.d_qkv).permute(0, 2, 1, 3)  # ? H L D
        k = self.WK(x).reshape(bs, -1, self.n_heads, self.d_qkv).permute(0, 2, 1, 3)  # ? H L D
        v = self.WV(x).reshape(bs, -1, self.n_heads, self.d_qkv).permute(0, 2, 1, 3)  # ? H L D
        attention = q @ k.transpose(-1, -2) / self.d_qkv ** 0.5  # ? H L L
        attention = self.dropout(F.softmax(attention, dim=-1))  # ? H L L
        result = attention @ v  # ? H L D
        result = result.transpose(1, 2).reshape(bs, -1, self.d_model)  # ? L Dm
        result = self.dropout(self.fc(result))

        # Add & Norm
        result = result + residual
        result = self.layer_norm(result)

        # Feed-Forward
        ff_res = self.ff(result)

        # Add & Norm
        result = result + self.dropout(ff_res)
        result = self.layer_norm(result)
        return result


class TemporalAttention(nn.Module):
    def __init__(self, num_nodes, input_len, tk, sk, d_model, d_ff, n_heads, dropout):
        super(TemporalAttention, self).__init__()
        self.tk = tk
        self.sk = sk
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.d_model = d_model
        self.attentions = nn.ModuleList()
        assert d_model % n_heads == 0, f"d_model {d_model} should be divided by n_heads {n_heads}."
        for i in range(self.tk):
            self.attentions.append(MultiHeadAttention(d_model, d_ff,
                                                      n_heads, dropout))

    def forward(self, x):
        # x: N V L tk sk c
        # x: tk N V L sk c
        x = torch.einsum('nvltsc->tnvlsc', x)
        # x: tk N*V L*sk c
        x = x.reshape(self.tk, -1, self.input_len * self.sk, self.d_model)

        outputs = []
        for i in range(self.tk):
            outputs.append(self.attentions[i](x[i]))
        # outputs: tk N*V L*sk c
        outputs = torch.stack(outputs)

        outputs = outputs.reshape(self.tk, -1, self.num_nodes, self.input_len, self.sk, self.d_model)
        outputs = torch.einsum('tnvlsc->nvltsc', outputs)
        # outputs: N V L tk sk c
        return outputs


class SpatialConvolution(nn.Module):
    def __init__(self, sk, tk, adp_emb, num_nodes, input_len, c_in, c_out, dropout, support_len, order):
        super(SpatialConvolution, self).__init__()

        support_len = support_len * 2 + 1
        self.dropout = dropout
        self.order = order
        self.sk = sk
        self.tk = tk
        self.num_nodes = num_nodes
        self.adp_emb = adp_emb
        self.c_in = c_in
        self.c_out = c_out
        self.input_len = input_len

        self.emb1 = nn.Parameter(torch.randn(self.sk, self.num_nodes, self.adp_emb), requires_grad=True)
        self.emb2 = nn.Parameter(torch.randn(self.sk, self.adp_emb, self.num_nodes), requires_grad=True)
        self.linear = nn.Linear(c_in * (1 + order * support_len), c_out)
        self.dropout = nn.Dropout(self.dropout)

    def conv(self, x, a):
        assert a.shape[-1] == a.shape[-2] == self.num_nodes
        if len(a.shape) == 2:
            return torch.einsum('sncvl,vw->sncwl', x, a)
        if len(a.shape) == 3:
            return torch.einsum('sncvl,svw->sncwl', x, a)
        assert False, f"len(a.shape) should be in \\{2, 3}, got {a.shape}: {len(a.shape)} instead."

    def forward(self, x, origin_supports):
        # x: N V L tk sk c
        adp = F.softmax(F.relu(torch.bmm(self.emb1, self.emb2)), dim=1)

        supports = [adp]
        for adj in origin_supports:
            supports.append(calc_sym(adj))
            supports.append(calc_sym(adj.transpose(0, 1)))

        # x: sk N*t c V L
        x = torch.einsum('nvltsc->sntcvl', x)
        x = x.reshape(self.sk, -1, self.c_in, self.num_nodes, self.input_len)

        out = [x]
        for a in supports:
            x = self.conv(x, a)
            out.append(x)
            for k in range(2, self.order + 1):
                x = self.conv(x, a)
                out.append(x)

        # out: sk N*t c V L
        # h: sk N*t C' V L
        h = torch.cat(out, dim=2)

        # h: sk N*t co V L
        h = self.linear(h.transpose(2, 4)).transpose(2, 4)
        h = self.dropout(h)

        # h: N V L tk sk co
        h = h.reshape(self.sk, -1, self.tk, self.c_out, self.num_nodes, self.input_len)
        h = torch.einsum('sntcvl->nvltsc', h)
        return h


class FreqAttention(nn.Module):
    def __init__(self, tk, sk, d_model, d_ff, n_heads, num_nodes, input_len, dropout):
        super(FreqAttention, self).__init__()
        self.tk = tk
        self.sk = sk
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.d_model = d_model
        self.attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads,
                                            d_ff=d_ff, dropout=dropout)

    def forward(self, x):
        # x: N V L tk sk C -> NV tk*sk*L C
        # x = torch.einsum('nvltsc->nvtslc', x)
        x = x.reshape(-1, self.tk * self.sk, self.d_model)

        x = self.attention(x)
        x = x.reshape(-1, self.num_nodes, self.input_len, self.tk, self.sk, self.d_model)
        # x = torch.einsum('nvtslc->nvltsc', x)
        return x


class Normalization(nn.Module):
    def __init__(self, tk, sk, num_nodes, input_len, d_model):
        super(Normalization, self).__init__()
        self.norm = nn.BatchNorm2d(tk * sk * d_model)
        self.tk = tk
        self.sk = sk
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.d_model = d_model

    def forward(self, x):
        # x: N V L tk sk c <-> N V L C <-> N C V L
        x = x.reshape(-1, self.num_nodes, self.input_len, self.tk * self.sk * self.d_model)
        x = self.norm(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = x.reshape(-1, self.num_nodes, self.input_len, self.tk, self.sk, self.d_model)
        return x


class CorrelationEncoder(nn.Module):
    def __init__(self, input_len, num_nodes, tk, sk, layers, adp_emb, n_heads, d_out, d_model, d_ff,
                 d_encoder, d_encoder_ff, dropout, support_len, order):
        super(CorrelationEncoder, self).__init__()
        self.tk = tk
        self.sk = sk
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.layers = layers
        self.d_model = d_model
        self.d_out = d_out
        self.start_linear = nn.Linear(num_nodes, d_model)
        self.position_embedding = nn.Parameter(torch.randn(input_len, tk, sk, d_model))
        self.att_t_fil = nn.ModuleList()
        self.att_t_gate = nn.ModuleList()
        self.att_s = nn.ModuleList()
        self.skip_conns = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(self.layers):
            self.att_t_fil.append(TemporalAttention(num_nodes=num_nodes, input_len=input_len, tk=tk,
                                                    sk=sk, d_model=d_model, d_ff=d_ff, dropout=dropout,
                                                    n_heads=n_heads))
            self.att_t_gate.append(TemporalAttention(num_nodes=num_nodes, input_len=input_len, tk=tk,
                                                     sk=sk, d_model=d_model, d_ff=d_ff, dropout=dropout,
                                                     n_heads=n_heads))
            self.att_s.append(SpatialConvolution(sk=sk, tk=tk, adp_emb=adp_emb, num_nodes=num_nodes,
                                                 input_len=input_len, c_in=d_model, c_out=d_model, dropout=dropout,
                                                 support_len=support_len, order=order))
            self.skip_conns.append(nn.Linear(d_model, d_model))
            self.batch_norms.append(Normalization(d_model=d_model, sk=sk, tk=tk, num_nodes=num_nodes,
                                                  input_len=input_len))

        self.mlp = nn.Conv2d(in_channels=d_model * sk * tk, out_channels=d_model * sk * tk,
                             kernel_size=(1, 1), bias=True)
        self.output = nn.Conv2d(in_channels=d_model * sk * tk, out_channels=d_out * sk * tk,
                                kernel_size=(1, 1), bias=True)

        self.freq_attention = FreqAttention(tk=tk, sk=sk, d_model=d_encoder, d_ff=d_encoder_ff, dropout=dropout,
                                            n_heads=n_heads, num_nodes=num_nodes, input_len=input_len)

    def forward(self, x, supports):
        # x: tk*sk N V Vd L -> tk sk N V Vd L -> N V L tk sk Vd
        x = x.reshape(self.tk, self.sk, -1, self.num_nodes, self.num_nodes, self.input_len)
        x = torch.einsum('tsnvwl->nvltsw', x)  # x = x.permute(2, 3, 5, 0, 1, 4)
        # x: N V L tk sk C
        x = self.start_linear(x)
        x = x + self.position_embedding  # add for causal transformer

        skip = torch.zeros_like(x)

        for i in range(self.layers):
            residual = x
            # temporal attention
            fil = self.att_t_fil[i](x)
            gate = self.att_t_gate[i](x)
            x = torch.tanh(fil) * torch.sigmoid(gate)

            s = self.skip_conns[i](x)
            skip += s

            x = self.att_s[i](x, supports) + residual
            x = self.batch_norms[i](x)

        x = F.relu(skip)

        # x: N V L tk sk C -> N V L tkskC -> N C' V L
        x = x.reshape(-1, self.num_nodes, self.input_len, self.tk * self.sk * self.d_model).permute(0, 3, 1, 2)

        x = F.relu(self.mlp(x))
        output = self.output(x)  # N tk*sk*C V L

        output = output.permute(0, 2, 3, 1).reshape(-1, self.num_nodes, self.input_len, self.tk, self.sk, self.d_out)

        # output, freq_attn: N V L tk sk C
        freq_attn = self.freq_attention(output)
        return freq_attn


class PredictionHead(nn.Module):
    def __init__(self, sk, tk, d_model, num_nodes, c_out, input_len, output_len, traditional):
        super(PredictionHead, self).__init__()
        self.sk = sk
        self.tk = tk
        self.num_nodes = num_nodes
        self.c_out = c_out
        self.input_len = input_len
        self.output_len = output_len
        self.d_model = d_model

        self.traditional = traditional
        if traditional:
            self.linear = nn.Conv2d(in_channels=d_model * sk * tk * input_len,
                                    out_channels=c_out * output_len,
                                    kernel_size=(1, 1), bias=True)
        else:
            self.linear = nn.Conv2d(in_channels=d_model * sk * tk * input_len,
                                    out_channels=num_nodes * c_out * output_len,
                                    kernel_size=(1, 1), bias=True)

    def forward(self, x):
        # x: N V L tk sk C
        x = x.reshape(-1, self.num_nodes, 1, self.input_len * self.tk * self.sk * self.d_model)

        # N V 1 C' -> N C' V 1
        x = x.permute(0, 3, 1, 2)

        # N C" V 1
        x = self.linear(x)

        if self.traditional:
            x = x.reshape(-1, self.c_out, self.output_len, self.num_nodes)
            x = torch.einsum('nclv->nlvc', x)
        else:
            x = x.reshape(-1, self.num_nodes, self.c_out, self.output_len, self.num_nodes)
            x = torch.einsum('nuclv->nclvu', x)

        # x: N C L V V
        return x


class Mask(nn.Module):
    def __init__(self, mask_percent):
        super(Mask, self).__init__()
        self.mask_percent = mask_percent

    def forward(self, x):
        # tk*sk N V V L
        freq, batch_size, num_nodes, _, length = x.shape
        device = x.device
        assert x.shape[3] == num_nodes

        mask_len = int(self.mask_percent * freq)
        keep_len = freq - mask_len

        # random choose {mask_len} elements in each node pair to mask the entire input len
        random_data = torch.rand(freq, batch_size, num_nodes, num_nodes, device=device)
        ids_shuffle = torch.argsort(random_data, dim=0)
        ids_restore = torch.argsort(ids_shuffle, dim=0)

        mask_shuffle = torch.zeros([mask_len, batch_size, num_nodes, num_nodes, length], device=device)
        keep_shuffle = torch.ones([keep_len, batch_size, num_nodes, num_nodes, length], device=device)
        total_mask_shuffle = torch.cat([mask_shuffle, keep_shuffle], dim=0)

        total_restore = ids_restore.unsqueeze(-1).repeat(1, 1, 1, 1, length)
        mask = torch.gather(total_mask_shuffle, 0, total_restore)
        return x * mask


class RecoverHead(nn.Module):
    def __init__(self, tk, sk, d_model, num_nodes):
        super(RecoverHead, self).__init__()
        self.sk = sk
        self.tk = tk
        self.d_model = d_model
        self.num_nodes = num_nodes
        self.linear = nn.Linear(in_features=tk * sk * d_model,
                                out_features=tk * sk * num_nodes)

    def forward(self, x):
        input_len = x.shape[2]

        # x: N V L t s C -> N V L tsC
        x = x.view(-1, self.num_nodes, input_len, self.tk * self.sk * self.d_model)

        # N V L tsV
        x = self.linear(x)

        # N V L t s U
        x = x.view(-1, self.num_nodes, input_len, self.tk, self.sk, self.num_nodes)

        # t s N V U L
        x = torch.einsum('nvltsu->tsnvul', x)

        # ts N V U L
        x = x.view(self.tk * self.sk, -1, self.num_nodes, self.num_nodes, input_len)
        return x


class ContrastiveHead(nn.Module):
    def __init__(self, sk, tk, num_nodes, d_model, input_len):
        super(ContrastiveHead, self).__init__()
        self.sk = sk
        self.tk = tk
        self.num_nodes = num_nodes
        self.d_model = d_model
        self.input_len = input_len
        self.discriminator = nn.Bilinear(self.d_model * self.input_len, self.d_model * self.input_len, 1)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, x):
        # x: N V L t s C
        bs = x.shape[0]

        result_loss = 0
        time_indices = torch.tensor(range(0, self.tk), device=x.device)
        spatial_indices = torch.tensor(range(0, self.sk), device=x.device)

        # N' V 1
        real_logits = torch.cat((torch.ones((self.sk + self.tk - 2) * bs, self.num_nodes, 1, device=x.device),
                                 torch.zeros((self.sk - 1) * (self.tk - 1) * bs, self.num_nodes, 1, device=x.device)),
                                dim=0)
        for i in range(self.tk):
            for j in range(self.sk):
                tid = torch.cat((time_indices[:i], time_indices[i + 1:]))
                sid = torch.cat((spatial_indices[:j], spatial_indices[j + 1:]))

                # N V L C
                h = x[:, :, :, i, j, :].reshape(-1, self.num_nodes, self.input_len * self.d_model)

                # N V L s+t-2 C
                positive_sample = torch.cat((torch.index_select(x[:, :, :, :, j, :], 3, tid),
                                             torch.index_select(x[:, :, :, i, :, :], 3, sid)), dim=3)
                # N' V C'
                positive_sample = positive_sample.permute(0, 3, 1, 2, 4)\
                                                 .reshape(-1, self.num_nodes, self.input_len * self.d_model)

                # N V L t-1 s-1 C
                negative_sample = torch.index_select(torch.index_select(x, 3, tid), 4, sid)
                # N' V C'
                negative_sample = negative_sample.permute(0, 3, 4, 1, 2, 5)\
                                                 .reshape(-1, self.num_nodes, self.d_model * self.input_len)

                # N' V 1
                pred = torch.cat((self.discriminator(h.repeat(self.sk + self.tk - 2, 1, 1), positive_sample),
                                  self.discriminator(h.repeat((self.sk - 1) * (self.tk - 1), 1, 1), negative_sample)))
                result_loss += self.bce(pred, real_logits) / self.sk / self.tk

        return result_loss


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
                                                      rsvd=config.rsvd)

        self.recover = config.recover
        self.do_mask = config.mask
        if config.recover:
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
                                          d_out=config.d_encoder)
        logging.info("Prediction Head")
        self.prediction_head = PredictionHead(sk=config.q, tk=config.p, num_nodes=config.num_nodes,
                                              d_model=config.d_encoder, c_out=config.c_out, input_len=input_len,
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

    def forward(self, x):
        self.recover_loss = self.contra_loss = 1e-10

        if self.transform_start_block is not None:
            x = self.transform_start_block(x)  # N L V C -> N L V V

        # x: N L V V
        # decomposed: tk*sk N V V L
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


def dwa(L_old, L_new, T=2):
    L_old = torch.Tensor(L_old)
    L_new = torch.Tensor(L_new)
    N = len(L_old)
    r = L_old / L_new
    if L_new[2] < 1e-4:
        r[2] = -100
    w = N * torch.softmax(r / T, dim=0)
    return w.numpy()


def calc_sym(adj):
    adj = adj + torch.eye(adj.size(-1)).to(adj.device)
    row_sum = torch.sum(adj, dim=-2)
    inv_rs_diag = torch.diag_embed(1 / row_sum)
    return adj @ inv_rs_diag
