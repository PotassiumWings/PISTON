import numpy as np
import torch
import torch.nn as nn

from configs.GEML_configs import GEMLConfig
from models.loss import mse_torch as mse
from models.abstract_st_encoder import AbstractSTEncoder


class SLSTM(nn.Module):
    def __init__(self, feature_dim, hidden_dim, device, p_interval):
        super(SLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.cell_dim = hidden_dim
        self.p_interval = p_interval
        self.f_gate = nn.Sequential(
            nn.Linear(feature_dim + hidden_dim, self.cell_dim),
            nn.Softmax(dim=1)
        )
        self.i_gate = nn.Sequential(
            nn.Linear(feature_dim + hidden_dim, self.cell_dim),
            nn.Softmax(dim=1)
        )
        self.o_gate = nn.Sequential(
            nn.Linear(feature_dim + hidden_dim, self.hidden_dim),
            nn.Softmax(dim=1)
        )
        self.g_gate = nn.Sequential(
            nn.Linear(feature_dim + hidden_dim, self.cell_dim),
            nn.Tanh()
        )

        self.tanh = nn.Tanh()

        self.device = device

    def forward(self, x):
        # (T, B * N, 2E)
        h = torch.zeros((x.shape[1], self.hidden_dim)).unsqueeze(dim=0).repeat(self.p_interval, 1, 1).to(self.device)
        # (P, B * N, 2E)
        c = torch.zeros((x.shape[1], self.hidden_dim)).unsqueeze(dim=0).repeat(self.p_interval, 1, 1).to(self.device)
        # (P, B * N, 2E)

        T = x.shape[0]

        for t in range(T):
            x_ = x[t, :, :]  # (B * N, 2E)
            x_ = torch.cat((x_, h[t % self.p_interval]), 1)  # (B * N, 2E + 2E)

            f = self.f_gate(x_)  # (B * N, 2E)

            i = self.i_gate(x_)  # (B * N, 2E)

            o = self.o_gate(x_)  # (B * N, 2E)

            g = self.g_gate(x_)  # (B * N, 2E)

            c = f * c[t % self.p_interval] + i * g  # (B * N, 2E)

            c = self.tanh(c)  # (B * N, 2E)

            h[t % self.p_interval] = o * c  # (B * N, 2E)

        return h[(T - 1) % self.p_interval]  # (B * N, 2E)


class MutiLearning(nn.Module):
    def __init__(self, fea_dim, device):
        super(MutiLearning, self).__init__()
        self.fea_dim = fea_dim
        self.transition = nn.Parameter(data=torch.randn(self.fea_dim, self.fea_dim).to(device), requires_grad=True)
        self.project_in = nn.Parameter(data=torch.randn(self.fea_dim, 1).to(device), requires_grad=True)
        self.project_out = nn.Parameter(data=torch.randn(self.fea_dim, 1).to(device), requires_grad=True)

    def forward(self, x: torch.Tensor):
        # (B, N, 2E)
        x_t = x.permute(0, 2, 1)  # (B, 2E, N)

        x_in = torch.matmul(x, self.project_in)  # (B, N, 1)

        x_out = torch.matmul(x, self.project_out)  # (B, N, 1)

        x = torch.matmul(x, self.transition)
        # (B, N, 2E)
        x = torch.matmul(x, x_t)
        # (B, N, N)

        x = x.unsqueeze(dim=-1).unsqueeze(dim=1)
        x_in = x_in.unsqueeze(dim=-1).unsqueeze(dim=1)
        x_out = x_out.unsqueeze(dim=-1).unsqueeze(dim=1)

        return x, x_in, x_out


class GraphConvolution(nn.Module):
    def __init__(self, feature_dim, embed_dim, device, use_bias=False):
        super(GraphConvolution, self).__init__()
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.activation = nn.ReLU()

        weight = torch.randn((self.feature_dim, self.embed_dim))
        self.weight = nn.Parameter(data=weight.to(device), requires_grad=True)
        if use_bias:
            self.bias = nn.Parameter(data=torch.zeros(self.embed_dim).to(device), requires_grad=True)
        else:
            self.bias = None

    def forward(self, x, a):
        # (B, N, N)
        embed = torch.matmul(a, x)
        # (B, N, N)
        embed = torch.matmul(embed, self.weight)
        # (B, N, E)
        if self.bias is not None:
            embed += self.bias

        return embed


class GCN(nn.Module):
    def __init__(self, feature_dim, embed_dim, device):
        super(GCN, self).__init__()
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.gcn = nn.ModuleList([
            GraphConvolution(feature_dim, embed_dim, device),
            GraphConvolution(embed_dim, embed_dim, device)
        ])

    def forward(self, input_seq, adj_seq):
        embed = []
        for i in range(input_seq.shape[1]):
            frame = input_seq[:, i, :, :]  # (B, N, N)
            adj = adj_seq[:, i, :, :]  # (B, N, N)

            for m in self.gcn:
                frame = m(frame, adj)
            # (B, N, E)

            embed.append(frame)

        return torch.stack(embed, dim=1)


def generate_geo_adj(distance_matrix: np.matrix):
    distance_matrix = torch.Tensor(distance_matrix)  # (N, N)
    distance_matrix = distance_matrix * distance_matrix
    sum_cost_vector = torch.sum(distance_matrix, dim=1, keepdim=True)  # (N, 1)
    weight_matrix = distance_matrix / sum_cost_vector
    weight_matrix[range(weight_matrix.shape[0]), range(weight_matrix.shape[1])] = 1
    return weight_matrix  # (N, N)


def generate_semantic_adj(demand_matrix, device):
    # (B, T, N, N)
    adj_matrix = demand_matrix.clone()
    in_matrix = adj_matrix.permute(0, 1, 3, 2)

    adj_matrix[adj_matrix > 0] = 1

    adj_matrix[in_matrix > 0] = 1

    degree_vector = torch.sum(adj_matrix, dim=3, keepdim=True)
    # (B, T, N, 1)

    sum_degree_vector = torch.matmul(adj_matrix, degree_vector)
    # (B, T, N, 1)

    weight_matrix = torch.matmul(1 / (sum_degree_vector + 1e-3), degree_vector.permute((0, 1, 3, 2)))  # (B, T, N, N)

    weight_matrix[:, :, range(weight_matrix.shape[2]), range(weight_matrix.shape[3])] = 1

    return weight_matrix


class GEML(AbstractSTEncoder):
    def __init__(self, config: GEMLConfig, gp_supports, scaler):
        super(GEML, self).__init__(config, gp_supports, scaler)
        self.num_nodes = config.num_nodes
        self.output_dim = config.c_out
        self.input_window = config.input_len
        self.output_window = config.output_len

        self.p_interval = config.p_interval
        self.embed_dim = config.embed_dim
        self.batch_size = config.batch_size
        self.loss_p1 = config.loss_p1
        self.loss_p2 = config.loss_p2

        dis_mx = gp_supports[0]
        self.geo_adj = generate_geo_adj(dis_mx) \
            .repeat(self.batch_size * self.input_window, 1) \
            .reshape((self.batch_size, self.input_window, self.num_nodes, self.num_nodes)) \
            .to(self.device)

        self.GCN_ge = GCN(self.num_nodes, self.embed_dim, self.device)
        self.GCN_se = GCN(self.num_nodes, self.embed_dim, self.device)

        # self.LSTM = nn.LSTM(2 * self.embed_dim, 2 * self.embed_dim)
        self.LSTM = SLSTM(2 * self.embed_dim, 2 * self.embed_dim, self.device, self.p_interval)

        self.mutiLearning = MutiLearning(2 * self.embed_dim, self.device)

    def forward(self, x, gp_supports, trues):
        # N V V L -> N L V V
        x = x.permute(0, 3, 1, 2)
        semantic_adj = generate_semantic_adj(x, self.device)

        # (B, T, N, N)
        x_ge_embed = self.GCN_ge(x, self.geo_adj[:x.shape[0], ...])
        # (B, T, N, E)

        x_se_embed = self.GCN_se(x, semantic_adj)

        # (B, T, N, E)
        x_embed = torch.cat([x_ge_embed, x_se_embed], dim=3)
        # (B, T, N, 2E)
        x_embed = x_embed.permute(1, 0, 2, 3)
        # (T, B, N, 2E)
        x_embed = x_embed.reshape((self.input_window, -1, 2 * self.embed_dim))
        # (T, B * N, 2E)

        # _, (h, _) = self.LSTM(x_embed)
        # x_embed_pred = h[0].reshape((self.batch_size, -1, 2 * self.embed_dim))
        x_embed_pred = self.LSTM(x_embed).reshape((x.shape[0], -1, 2 * self.embed_dim))
        # (B, N, 2E)

        y_pred, y_in, y_out = self.mutiLearning(x_embed_pred)

        y_true = trues  # (B, TO, N, N)
        y_in_true = torch.sum(y_true, dim=-1, keepdim=True)  # (B, TO, N)
        y_out_true = torch.sum(y_true, dim=-2, keepdim=True)  # (B, TO, N)

        y_in = self.scaler.inverse_transform(y_in)
        y_out = self.scaler.inverse_transform(y_out)
        loss_in = mse(y_in, y_in_true)
        loss_out = mse(y_out, y_out_true)
        self.forward_loss = self.loss_p1 * loss_in + self.loss_p2 * loss_out

        return y_pred
