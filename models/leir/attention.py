import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt


class SelfAttention(nn.Module):
    def __init__(self, dim_q=1024, dim_k=1024, dim_v=1024):
        super().__init__()
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v

        self.linear_q = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_q, dim_v, bias=False)
        self._norm_fact = 1.0 / sqrt(dim_k)

    def forward(self, x):
        batch, n, dim_q = x.shape
        if dim_q != self.dim_q:
            raise ValueError(f"Expected dim {self.dim_q}, got {dim_q}.")

        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)

        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact
        dist = torch.softmax(dist, dim=-1)
        return torch.bmm(dist, v)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in=1024, d_hid=1024, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x = x + residual
        return self.layer_norm(x)
