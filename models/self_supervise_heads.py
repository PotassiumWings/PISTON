import torch
import torch.nn as nn


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
                positive_sample = positive_sample.permute(0, 3, 1, 2, 4) \
                    .reshape(-1, self.num_nodes, self.input_len * self.d_model)

                # N V L t-1 s-1 C
                negative_sample = torch.index_select(torch.index_select(x, 3, tid), 4, sid)
                # N' V C'
                negative_sample = negative_sample.permute(0, 3, 4, 1, 2, 5) \
                    .reshape(-1, self.num_nodes, self.d_model * self.input_len)

                # N' V 1
                pred = torch.cat((self.discriminator(h.repeat(self.sk + self.tk - 2, 1, 1), positive_sample),
                                  self.discriminator(h.repeat((self.sk - 1) * (self.tk - 1), 1, 1), negative_sample)))
                result_loss += self.bce(pred, real_logits) / self.sk / self.tk

        return result_loss
