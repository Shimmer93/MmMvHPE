import torch
import torch.nn as nn

from misc.registry import create_loss

class BaseHead(nn.Module):
    def __init__(self, losses):
        super(BaseHead, self).__init__()
        self.losses = {loss['name']: (create_loss(loss['name'], loss['params']), loss['weight']) for loss in losses}

    def forward(self, x):
        raise NotImplementedError

    def loss(self, x, data_batch):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError