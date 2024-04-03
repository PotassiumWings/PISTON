import torch


def masked_mae_loss(mask_value):
    def loss(preds, labels):
        mae = mae_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae

    return loss


def masked_mse_loss(mask_value):
    def loss(preds, labels):
        mse = mse_torch(pred=preds, true=labels, mask_value=mask_value)
        return mse

    return loss


def masked_huber_loss(mask_value, delta):
    def loss(preds, labels):
        return huber_torch(preds, labels, mask_value, delta)

    return loss


def huber_torch(pred, true, mask_value, delta):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    a = torch.abs(pred - true)
    a_great = torch.masked_select(a, torch.gt(a, delta))
    a_small = torch.masked_select(a, torch.lt(a, delta))
    size = a_great.size(0) + a_small.size(0)
    return (torch.sum(a_small * a_small) * 0.5 + delta * torch.sum(a_great - 0.5 * delta)) / size


def mae_torch(pred, true, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(pred - true))


def mape_torch(pred, true, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(torch.div((pred - true), true)))


def rmse_torch(pred, true, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.mean(torch.square(pred - true)))


def mse_torch(pred, true, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.square(pred - true))
