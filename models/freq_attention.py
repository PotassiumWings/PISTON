import torch
import torch.nn as nn
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
                 d_encoder, d_encoder_ff, dropout, support_len, order):
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

        self.freq_attention = FreqAttention(tk=tk, sk=sk, d_model=d_encoder, d_ff=d_encoder_ff, dropout=dropout,
                                            n_heads=n_heads, num_nodes=num_nodes, input_len=input_len,
                                            output_len=output_len)

    def forward(self, x, supports):
        # # x: tk*sk N V Vd L -> tk sk N V Vd L -> N V L tk sk Vd
        # x = x.reshape(self.tk, self.sk, -1, self.num_nodes, self.num_nodes, self.input_len)
        # x = torch.einsum('tsnvwl->nvltsw', x)  # x = x.permute(2, 3, 5, 0, 1, 4)
        # # x: N V L tk sk C
        # # x = self.start_linear(x)

        # x: N V L tk sk C
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

        # output, freq_attn: N V Lo C
        freq_attn = self.freq_attention(output)
        return freq_attn


def calc_sym(adj):
    adj = adj + torch.eye(adj.size(-1)).to(adj.device)
    row_sum = torch.sum(adj, dim=-2)
    inv_rs_diag = torch.diag_embed(1 / row_sum)
    return adj @ inv_rs_diag
