import torch

def CrossEntropyLoss(kwargs):
    return torch.nn.CrossEntropyLoss(**kwargs)


def NLLLoss(kwargs):
    """
    The negative log likelihood loss. It is useful to train a classification problem with C classes.
    https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss
    """
    return torch.nn.NLLLoss(**kwargs)


def MultiLabelSoftMarginLoss(kwargs):
    """
    Creates a criterion that optimizes a multi-label one-versus-all loss based on max-entropy, between input x and target y of size (N,C). 
    https://pytorch.org/docs/stable/generated/torch.nn.MultiLabelSoftMarginLoss.html#torch.nn.MultiLabelSoftMarginLoss
    """
    return torch.nn.MultiLabelSoftMarginLoss(**kwargs)
