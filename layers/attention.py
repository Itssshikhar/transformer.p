import math
import torch
import torch.nn as nn
from torch import Tensor, BoolTensor
from torch.nn import functional as F


class SingleHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, bias: bool = True):
        super().__init__()
        self.Wqkv = nn.Linear(hidden_size, (hidden_size//4)*3, bias=bias)
        self.Wo = nn.Linear(hidden_size//4, hidden_size, bias=bias)

    def forward(self, x:Tensor):
        batch, s_length, channel = x.shape
        q, k, v = self.Wqkv(x).reshape(batch, s_length, 3, channel//4).unbind(dim=2)

        attn = q @ k.transpose(-2, -1)
        attn = attn / math.sqrt(k.size(-1))
        attn = attn.softmax(dim=-1)

        x = attn @ v
        return self.Wo(x)

class BidirectionalAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, attn_dropout: float = 0.1, out_dropout: float = 0.1, bias: bool = True):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.nh = num_heads
        self.Wqkv = nn.Linear(hidden_size, hidden_size*3, bias=bias)
        self.Wo = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_dropout = nn.Dropout(out_dropout)

    def forward(self, x:Tensor, mask: BoolTensor):
        batch, s_length, channel = x.shape

        x = self.Wqkv(x).reshape(batch, s_length, 3, self.nh, channel//self.nh)
        q, k, v = x.transpose(3, 1).unbind(dim=2)

        attn = q @ k.transpose(-2, -1)
        attn = attn / math.sqrt(k.size(-1))

        attn = attn.masked_fill(mask.view(batch, 1, 1, s_length), float('-inf'))

        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(batch, s_length, channel)

        return self.out_drop(self.Wo(x))

class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, context_size: int, attn_dropout: float = 0.1, out_dropout: float = 0.1, bias: bool = True):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.nh = num_heads
        self.Wqkv = nn.Linear(hidden_size, hidden_size*3, bias=bias)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.Wo = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.out_dropout = nn.Dropout(out_dropout)
        self.register_buffer('casual_mask', torch.triu(torch.ones([context_size, context_size], dtype=torch.bool), diagonal=1).view(1, 1, context_size, context_size), persistent=False)
    
    def forward(self, x:Tensor, mask: BoolTensor):
        batch, s_length, channel = x.shape

        x = self.Wqkv(x).reshape(batch, s_length, 3, self.nh, channel//self.nh)
        q, k, v = x.transpose(3, 1).unbind(dim=2)

        attn = q @ k.transpose(-2, -1)
        attn = attn / math.sqrt(k.size(-1))

        combined_mask = self.casual_mask[:, :, :s_length, :s_length]
        if mask is not None:
            combined_mask += mask.view(batch, 1, 1, s_length)
        attn = attn.masked_fill(combined_mask, float('-inf'))

        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(batch, s_length, channel)

        return self.out_drop(self.Wo(x))

class CrossAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, context_size: int, attn_dropout: float = 0.1, out_dropout: float = 0.1, bias: bool = True):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.nh = num_heads
        self.Wq = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.Wkv = nn.Linear(hidden_size, hidden_size * 2, bias=bias)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.Wo = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.out_dropout = nn.Dropout(out_dropout)
        self.register_buffer('casual_mask', torch.triu(torch.ones([context_size, context_size], dtype=torch.bool), diagonal=1).view(1, 1, context_size, context_size), persistent=False)

    def forward(self, x: Tensor, y: Tensor, mask: BoolTensor):
        batch, s_length, channel = x.shape

        q = self.Wq(x).reshape(batch, s_length, self.nh, channel//self.nh).transpose(1,2)

        y = self.Wkv(y).reshape(batch, s_length, 2, self.nh, channel//self.nh)
        k, v = y.transpose(3, 1).unbind(dim=2)

        attn = q @ k.transpose(-2, -1)
        attn = attn / math.sqrt(k.size(-1))

        combined_mask = self.casual_mask[:, :, :s_length, :s_length]
        if mask is not None:
            combined_mask += mask.view(batch, 1, 1, s_length)
        attn = attn.masked_fill(combined_mask, float('-inf'))

        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(batch, s_length, channel)

        return self.out_drop(self.Wo(x))
