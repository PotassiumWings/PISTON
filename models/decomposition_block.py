import ptwt
import pywt
import torch
import torch.nn as nn


class DecompositionBlock(nn.Module):
    def __init__(self, rsvd, input_len, sk, tk, n, random_svd_k, use_rsvd_emb, output_dim):
        super(DecompositionBlock, self).__init__()
        self.rsvd = rsvd
        self.tk = tk
        self.sk = sk
        self.n = n
        self.k = random_svd_k
        self.input_len = input_len
        self.use_rsvd_emb = use_rsvd_emb and rsvd

        self.p = nn.Parameter(torch.randn(self.n, self.k), requires_grad=True)
        self.start_linear_o = nn.ModuleList()
        self.start_linear_d = nn.ModuleList()
        self.output_dim = output_dim
        for i in range(self.tk):
            for j in range(self.sk - 1):
                if self.use_rsvd_emb:
                    self.start_linear_o.append(nn.Linear(2, output_dim))
                    self.start_linear_d.append(nn.Linear(2, output_dim))
                else:
                    self.start_linear_o.append(nn.Linear(n, output_dim))
                    self.start_linear_d.append(nn.Linear(n, output_dim))
            self.start_linear_o.append(nn.Linear(n, output_dim))
            self.start_linear_d.append(nn.Linear(n, output_dim))

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

        use_rsvd_emb = self.use_rsvd_emb and self.rsvd

        # x: [y1, y2, ..., ytk], y NVVl
        res = []
        for i in range(self.tk):
            if self.sk == 1:
                res.append(x[i])
                continue

            u, sig, v = self.svd(x[i].permute(0, 3, 1, 2))  # NlVV
            sum_val = torch.zeros_like(x[i])

            for j in range(self.sk - 1):
                # NlVk Nlkk NlVk
                ui, sigi, vi = u[..., j:j + 1], sig[..., j:j + 1], v[..., j:j + 1]
                # mat: NlVV -> NVVl
                mat = torch.matmul(torch.matmul(ui, torch.diag_embed(sigi)), vi.transpose(-2, -1)).permute(0, 2, 3, 1)
                sum_val += mat
                if use_rsvd_emb:
                    # NlV1*Nl11 NlV1 -> NlV2 -> NV2l
                    emb = torch.cat([u[..., j:j + 1] * sig[..., j:j + 1].unsqueeze(-1), v[..., j:j + 1]], -1)
                    res.append(emb.permute(0, 2, 3, 1))
                else:
                    res.append(mat)

            # residual term
            res.append(x[i])
            # res.append(x[i] - sum_val)
        # padding
        for i in range(len(res)):
            # NV?l -> NlV?
            res[i] = nn.functional.pad(res[i], (self.input_len - res[i].size(3), 0, 0, 0))
            res[i] = res[i].permute(0, 3, 1, 2)
            # N l V C
            res[i] = torch.cat([self.start_linear_o[i](res[i]),
                                self.start_linear_d[i](res[i].transpose(-2, -1))], dim=-1)
        # res: tk*sk  N l V C
        res = torch.stack(res)

        # N V L tksk C -> N V L tk sk C
        res = res.permute(1, 3, 2, 0, 4)
        res = res.view(-1, self.n, self.input_len, self.tk, self.sk, 2 * self.output_dim)
        return res
