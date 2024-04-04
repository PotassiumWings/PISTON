import torch
import pywt
import ptwt


class DecompositionBatch:
    def __init__(self, p, q):
        self.data = None
        self.tk = p
        self.sk = q

    def init_batch(self):
        self.data = None

    def get_data(self, x):
        if self.data is None:
            self.decomposition(x)
        return self.data

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
            # logging.info("svd end")
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
