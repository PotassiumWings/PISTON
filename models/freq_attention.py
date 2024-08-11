import torch
import torch.nn as nn
import math
import logging
import torch.nn.functional as F


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
    def __init__(self, tk, sk, d_model, d_ff, n_heads, num_nodes, input_len, dropout, output_len):
        super(FreqAttention, self).__init__()
        self.tk = tk
        self.sk = sk
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len
        self.d_model = d_model
        self.attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads,
                                            d_ff=d_ff, dropout=dropout)
        self.linear = nn.Linear(input_len * sk * tk, output_len)

    def forward(self, x):
        # x: N V L_in tk sk C -> NVLo ts C
        x = x.reshape(-1, self.tk * self.sk, self.d_model)

        # NVLo ts C
        x = self.attention(x)

        # NV Lts C
        x = x.reshape(-1, self.input_len * self.tk * self.sk, self.d_model)

        # NV Lout C
        x = self.linear(x.permute(0, 2, 1)).permute(0, 2, 1)

        # N V Lout C
        x = x.reshape(-1, self.num_nodes, self.output_len, self.d_model)
        return x


class FreqSetDPPAttention(nn.Module):
    def __init__(self, sk, tk, input_len, d_model, num_nodes, dropout):
        super(FreqSetDPPAttention, self).__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model
        self.sk = sk
        self.tk = tk
        self.nc = self.sk * self.tk  # num_components
        self.d_attn = d_model
        self.d_ff = d_model
        self.input_len = input_len
        self.wq = nn.Parameter(torch.randn(self.d_model, self.d_attn),
                               requires_grad=True)
        self.wk = nn.Parameter(torch.randn(self.d_model, self.d_attn * self.d_attn),
                               requires_grad=True)
        self.wv = nn.Parameter(torch.randn(self.d_model, self.d_model), requires_grad=True)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, self.d_ff),
                                nn.GELU(),
                                nn.Dropout(dropout),
                                nn.Linear(self.d_ff, d_model))
        # self.cnt = 0

    def forward(self, x):
        residual = x
        batch_size = x.shape[0]
        # self.cnt += 1
        # if self.cnt == 18:
        #     import pdb
        #     pdb.set_trace()

        # x: N V L t s C -> t s N C V L
        x = torch.einsum('nvltsc->ntsvlc', x)
        # N ts VL C
        x = x.reshape(-1, self.tk * self.sk, self.input_len * self.num_nodes, self.d_model)
        # N ts C
        x_mean = x.mean(-2)

        # q: N ts C(d_attn)
        q = x_mean @ self.wq
        # k: N ts C'^2 -> N ts C' C'
        k = (x_mean @ self.wk).view(-1, self.nc, self.d_attn, self.d_attn)
        # N ts C'
        kt = k.mean(-1)
        # N tsC' C'
        k = k.view(-1, self.nc * self.d_attn, self.d_attn)
        # v: N ts VL C'
        v = x @ self.wv

        # N ts C' * N C' tsC' -> N ts tsC' -> N ts ts C'
        b = (q @ k.transpose(-1, -2)).view(-1, self.nc, self.nc, self.d_attn)

        # N ts ts 1
        r = b.norm(dim=-1, p=2, keepdim=True)
        # N ts ts C'
        f = b / r

        # N ts ts ts, 单位向量点积 [-1, 1]
        s = f @ f.transpose(-2, -1)
        # [-1, 1] -> [0, 1]
        s = (s + 1) / 2
        # Nts ts ts
        s = s.reshape(-1, self.nc, self.nc)
        r = r.reshape(-1, self.nc)

        res = []
        for ind in range(batch_size * self.nc):
            batch_ind = ind // self.nc
            node_ind = ind % self.nc

            l = s[ind].clone()
            rs = r[ind]
            diag = torch.diag(l)  # store d^2

            # exclude node_ind
            l[node_ind] -= l[node_ind]
            l[:, node_ind] -= l[:, node_ind]
            diag[node_ind] = -1e20

            j = torch.argmax(diag * rs)  # argmax log(det L_S+{j}) - log(det L_S)
            yg = [int(j.cpu().numpy())]
            c = torch.zeros((self.nc + 1, self.nc)).to(x.device)
            z_all = list(range(0, node_ind)) + list(range(node_ind + 1, self.nc))
            # import pdb
            # pdb.set_trace()
            iter_ind = 1
            while iter_ind < self.nc:
                z_y = set(z_all).difference(set(yg))
                for i in z_y:
                    e = (l[j][i] - c[:iter_ind, j].dot(c[:iter_ind, i])) / torch.sqrt(diag[j])
                    c[iter_ind, i] = e
                    diag[i] -= e * e
                diag[j] = -1e20
                j = torch.argmax(diag * rs)
                if diag[j] * rs[j] < 1:
                    break
                yg.append(int(j.cpu().numpy()))
                iter_ind += 1

            if len(yg) != self.nc - 1:
                logging.info(f"finally! {yg}")
            weights = []
            vals = []
            for i in yg:
                # C' * C' -> 1
                val = q[batch_ind][node_ind] @ kt[batch_ind][i]
                vals.append(v[batch_ind][i])
                weights.append(val)
            # |S| 1
            # import pdb
            # pdb.set_trace()
            weights = torch.softmax(torch.stack(weights).squeeze(-1), dim=0)
            # |S| VL C' -> VL C' |S|
            vals = torch.stack(vals).permute(1, 2, 0)

            res.append(vals @ weights)

        # Nts V L C -> N t s V L C' -> N V L t s C
        output = torch.stack(res).view(-1, self.tk, self.sk, self.num_nodes, self.input_len, self.d_attn)\
            .permute(0, 3, 4, 1, 2, 5)

        result = output + residual
        if result.isnan().any():
            import pdb
            pdb.set_trace()
        result = self.layer_norm(result)

        # Feed-Forward
        ff_res = self.ff(result)

        # Add & Norm
        result = result + self.dropout(ff_res)
        result = self.layer_norm(result)
        return result


class FreqSetAttention(nn.Module):
    def __init__(self, tk, sk, d_model, d_ff, n_heads, num_nodes, input_len, dropout, output_len, only_1):
        super(FreqSetAttention, self).__init__()
        self.tk = tk
        self.sk = sk
        self.set_size = int(math.pow(2, tk * sk - 1))
        self.mask_size = self.tk * self.sk
        self.only_1 = only_1

        self.masks = []
        for i in range(self.mask_size):
            masks = []
            for s in range(self.set_size):
                bitmap = format(s, f'0{self.mask_size - 1}b')
                mask_list = [int(bit) for bit in bitmap]
                mask_list.insert(i, 0)
                mask = torch.LongTensor(mask_list)
                mask = mask.nonzero().squeeze(-1)
                masks.append(mask)
            self.masks.append(masks)

        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len
        self.d_model = d_model
        self.d_qkv = d_model // n_heads
        self.n_heads = n_heads

        self.dropout = nn.Dropout(dropout)

        self.WQ, self.WK, self.WV = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i in range(self.mask_size):
            self.WQ.append(nn.Linear(d_model, d_model))
            self.WK.append(nn.Linear(d_model, d_model))
            self.WV.append(nn.Linear(d_model, d_model))

        self.attn_weights = nn.Parameter(torch.randn(self.mask_size, self.set_size))

    def forward(self, x):
        bs = x.size(0)

        # x: N V L_in tk sk C -> tk sk N L_in V C
        x = torch.einsum("nvltsc->tsnlvc", x)
        # x: tk sk N L_in V C -> tksk N L_in V C
        x = x.reshape(self.tk * self.sk, -1, self.input_len, self.num_nodes, self.d_model)

        qs, ks, vs = [], [], []
        for i in range(self.mask_size):
            q = self.WQ[i](x[i]).reshape(bs, -1, self.n_heads, self.d_qkv).permute(0, 2, 1, 3)
            k = self.WK[i](x[i]).reshape(bs, -1, self.n_heads, self.d_qkv).permute(0, 2, 1, 3)
            v = self.WV[i](x[i]).reshape(bs, -1, self.n_heads, self.d_qkv).permute(0, 2, 1, 3)
            qs.append(q)
            ks.append(k)
            vs.append(v)
        qs, ks, vs = torch.stack(qs), torch.stack(ks), torch.stack(vs)

        all_res = []
        for i in range(self.mask_size):
            residual = x[i].permute(0, 2, 1, 3)  # N L V C -> N V L C
            sum_exp = 0
            result_part = 0
            # 0: residual
            start = 1
            if self.only_1:
                start = self.set_size - 1
            for s in range(start, self.set_size):
                # calc mask: tk*sk
                mask = self.masks[i][s]

                q = qs[i]  # N H VL C
                k = ks[mask]  # m N H VL C
                v = vs[mask]
                m = k.size(0)

                # N VL / H / 1 / C
                q = q.permute(0, 2, 1, 3).reshape(-1, self.n_heads, 1, self.d_qkv)
                # N VL / H / C / m
                k = k.permute(1, 3, 2, 4, 0).reshape(-1, self.n_heads, self.d_qkv, m)
                v = v.permute(1, 3, 2, 4, 0).reshape(-1, self.n_heads, self.d_qkv, m)

                # N VL / H / 1 / m
                attention = q @ k / self.d_qkv ** 0.5
                attention = self.dropout(F.softmax(attention, dim=-1))
                # N VL / H / 1 / C
                res = attention @ v.transpose(-2, -1)
                # N V L C
                res = res.reshape(-1, self.num_nodes, self.input_len, self.d_model)

                result_part += res * torch.exp(self.attn_weights[i][s])
                sum_exp += torch.exp(self.attn_weights[i][s])
            # softmax
            result_part /= sum_exp

            all_res.append(result_part + residual)

        # ts N V L C
        all_res = torch.stack(all_res)

        # ts N V L C -> N V L ts C -> N V L t s C
        all_res = all_res.permute(1, 2, 3, 0, 4)
        all_res = all_res.reshape(-1, self.num_nodes, self.input_len, self.tk, self.sk, self.d_model)
        return all_res


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
    def __init__(self, input_len, output_len, num_nodes, tk, sk, layers, adp_emb, n_heads, d_out, d_model, d_ff,
                 d_encoder, d_encoder_ff, dropout, support_len, order, only_1):
        super(CorrelationEncoder, self).__init__()
        self.tk = tk
        self.sk = sk
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.layers = layers
        self.d_model = d_model
        self.d_out = d_out
        # self.start_linear = nn.Linear(num_nodes, d_model)
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

        # self.freq_attention = FreqAttention(tk=tk, sk=sk, d_model=d_encoder, d_ff=d_encoder_ff, dropout=dropout,
        #                                     n_heads=n_heads, num_nodes=num_nodes, input_len=input_len,
        #                                     output_len=output_len)

        # self.freq_attention = FreqSetAttention(tk=tk, sk=sk, d_model=d_model, d_ff=d_model, dropout=dropout,
        #                                        n_heads=n_heads, num_nodes=num_nodes, input_len=input_len,
        #                                        output_len=output_len, only_1=only_1)

        self.freq_attention = FreqSetDPPAttention(sk=sk, tk=tk, input_len=input_len, d_model=d_model,
                                                  num_nodes=num_nodes, dropout=dropout)

        self.output_layer = nn.Linear(input_len * sk * tk, output_len)

    def forward(self, x, supports):
        # # x: tk*sk N V Vd L -> tk sk N V Vd L -> N V L tk sk Vd
        # x = x.reshape(self.tk, self.sk, -1, self.num_nodes, self.num_nodes, self.input_len)
        # x = torch.einsum('tsnvwl->nvltsw', x)  # x = x.permute(2, 3, 5, 0, 1, 4)
        # # x: N V L tk sk C
        # # x = self.start_linear(x)

        # x: N V L tk sk C
        x = x + self.position_embedding  # add for causal transformer

        x = self.freq_attention(x)

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

        # N V L T S C
        output = output.permute(0, 2, 3, 1).reshape(-1, self.num_nodes, self.input_len, self.tk, self.sk, self.d_out)

        # output, freq_attn: N V Lo C
        # freq_attn = self.freq_attention(output)
        # N V C TSL -> N V C L -> N V L C
        output = output.permute(0, 1, 5, 2, 3, 4)\
                       .reshape(-1, self.num_nodes, self.d_out, self.tk * self.sk * self.input_len)
        output = self.output_layer(output)
        return output.transpose(-1, -2)


def calc_sym(adj):
    adj = adj + torch.eye(adj.size(-1)).to(adj.device)
    row_sum = torch.sum(adj, dim=-2)
    inv_rs_diag = torch.diag_embed(1 / row_sum)
    return adj @ inv_rs_diag
