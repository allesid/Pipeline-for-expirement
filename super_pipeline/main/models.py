import torch
from torchvision import models


def resnext50_32x4d_1Line(lenclassnames, kwargs):

    model = models.resnext50_32x4d(**kwargs)
    model.fc = torch.nn.Linear(model.fc.in_features, lenclassnames)
    return model


def resnext50_32x4d_2D(kwargs):

    model = models.resnext50_32x4d(**kwargs)
    model.fc = torch.nn.Linear(model.fc.in_features, 512*512)
    return model
