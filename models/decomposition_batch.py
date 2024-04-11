import torch
import pywt
import ptwt
import logging


class DecompositionBatch:
    def __init__(self, config, p, q):
        self.data = None
        self.config = config
        self.tk = p
        self.sk = q
        self.data = None
        self.lambdas = None
        self.avg_indexes = None

    def init_batch(self):
        self.data = None
        self.lambdas = []

    def get_data(self, x):
        if self.data is None:
            self.decomposition(x)
        return self.data

    def calc_avg(self, sig):
        # average split in spatial dimension
        if self.avg_indexes is not None:
            return self.avg_indexes

        # sig: N l V
        _, _, V = sig.shape
        result = [0]

        # squared
        if self.config.squared_lambda:
            sig = sig.pow(2)

        sig_sum = sig.sum()
        for i in range(self.sk - 1):
            pointer = result[-1]
            cur_sum = 0
            while cur_sum < sig_sum / (self.sk - i):
                cur_sum += sig[..., pointer].sum()
                pointer += 1
            sig_sum -= cur_sum
            result.append(pointer)
        result.append(V)
        self.avg_indexes = result
        logging.info(f"Average indexes: {result}")
        return result

    def get_spatio_weight(self, t_ind, s_ind):
        if s_ind == self.sk - 1:
            return 1.0 / self.sk

        lamb = self.lambdas[t_ind]
        if self.config.squared_lambda:
            lamb = lamb.pow(2)

        return lamb[..., s_ind].sum() / lamb[..., :self.sk - 2].sum() * (self.sk - 1) / self.sk

    def decomposition(self, x):
        # x: N L V V -> N V V L
        # x -> tk
        x = x.permute(0, 2, 3, 1)

        wavelet = pywt.Wavelet("haar")
        # logging.info("wavedec start")
        x = ptwt.wavedec(x, wavelet, mode='zero', level=self.tk - 1)
        # logging.info("wavedec end")
        x = x[::-1]

        # x: [y1, y2, ..., ytk], y NVVl
        res = []
        for i in range(self.tk):
            if self.sk == 1:
                res.append(x[i])
                continue

            # logging.info("svd start")
            u, sig, v = torch.svd(x[i].permute(0, 3, 1, 2))  # NlVV
            self.lambdas.append(sig)

            # logging.info("svd end")
            if self.config.avg_q:
                avg_indexes = self.calc_avg(sig)
                for j in range(self.sk):
                    s = avg_indexes[j]
                    t = avg_indexes[j + 1]
                    ui, sigi, vi = u[..., s:t], sig[..., s:t], v[..., s:t]
                    # mat: NlVV
                    mat = torch.matmul(torch.matmul(ui, torch.diag_embed(sigi)), vi.transpose(-2, -1))
                    # N V V l
                    res.append(mat.permute(0, 2, 3, 1))
            else:
                for j in range(self.sk - 1):
                    ui, sigi, vi = u[..., j:j + 1], sig[..., j:j + 1], v[..., j:j + 1]
                    # mat: NlVV
                    mat = torch.matmul(torch.matmul(ui, torch.diag_embed(sigi)), vi.transpose(-2, -1))
                    # N V V l
                    res.append(mat.permute(0, 2, 3, 1))

                # residual term
                mat = torch.matmul(torch.matmul(u[..., self.sk:], torch.diag_embed(sig[..., self.sk:])),
                                   v[..., self.sk:].transpose(-2, -1))
                res.append(mat.permute(0, 2, 3, 1))
        # res: tk sk  N V V l
        self.data = res
        return self.data
