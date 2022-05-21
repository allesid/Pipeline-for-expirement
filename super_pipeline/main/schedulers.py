import torch


def StepLR(optimizer, kwargs):
    return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
