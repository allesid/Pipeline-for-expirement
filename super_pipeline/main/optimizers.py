import torch


def Adam(modelpars, kwargs):
    return torch.optim.Adam(modelpars, **kwargs)

