import math

import torch
import torch.nn as nn
from torch import Tensor, BoolTensor
from torch.nn import functional as F
from layers.attention import BidirectionalAttention


class PositionalEncoding(nn.Module):
    def __init__(self, context_size: int, hidden_size: int):
        super().__init__()
        # create the positional encoding tensor of shape
        # maximum sequence length (MS) by embedding dimension (C)
        pe = torch.zeros(context_size, hidden_size, dtype=torch.float)

        # pre-populate the position and the div_terms
        position = torch.arange(context_size).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2) * (-math.log(10000) / hidden_size)
        )

        # even positional encodings use sine, odd cosine
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # register as a buffer so autograd doesn't modify
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor):
        # return the pre-calculated positional encodings
        # up to sequence length (S). output shape (1, S, C)
        return self.pe[:, :x.shape[1], :]


class FeedForward(nn.Module):
    def __init__(self, hidden_size:int, expand_size:int, act:nn.Module=nn.GELU,
                 drop:float=0.1, bias:bool=True):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, expand_size, bias=bias)

        self.act = act()

        self.fc2 = nn.Linear(expand_size, hidden_size, bias=bias)

        self.drop = nn.Dropout(drop)

    def forward(self, x:Tensor):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size:int, num_heads:int, expand_size:int,
                 attention:nn.Module=BidirectionalAttention, act:nn.Module=nn.GELU,
                 attn_drop:float=0.1, out_drop:float=0.1, ffn_drop:float=0.1,
                 bias:bool=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = attention(
            hidden_size=hidden_size, num_heads=num_heads, attn_drop=attn_drop,
            out_drop=out_drop, bias=bias
        )

        self.norm2 = nn.LayerNorm(hidden_size)
        # initialize the feed forward network (MLP)
        self.ffn = FeedForward(
            hidden_size=hidden_size, expand_size=expand_size, act=act,
            drop=ffn_drop, bias=bias,
        )

    def forward(self, x: Tensor):
        x = x + self.attn(self.norm1(x))

        return x + self.ffn(self.norm2(x))


class BERT(nn.Module):
    def __init__(self, num_layers:int, vocab_size:int, hidden_size:int, num_heads:int,
                 context_size:int, expand_size:int, attention:nn.Module=BidirectionalAttention,
                 act:nn.Module=nn.GELU, embed_drop:float=0.1, attn_drop:float=0.1,
                 out_drop:float=0.1, ffn_drop:float=0.1, head_norm:bool=True,
                 tie_weights:bool=True, head_bias:bool=True, bias:bool=True):
        super().__init__()
        # initialize vocab & positional encodings to convert numericalied tokens
        # & position indicies to token and position vectors, with optional dropout
        self.vocab_embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_encode = PositionalEncoding(context_size, hidden_size)
        self.embed_drop = nn.Dropout(embed_drop)

        self.tfm_blocks = nn.ModuleList([TransformerBlock(
                hidden_size=hidden_size, num_heads=num_heads, expand_size=expand_size,
                attention=attention, act=act, bias=bias, attn_drop=attn_drop,
                out_drop=out_drop, ffn_drop=ffn_drop)
            for _ in range(num_layers)])

        # optional pre-head normalization
        self.head_norm = nn.LayerNorm(hidden_size) if head_norm else nn.Identity()

        self.head = nn.Linear(hidden_size, vocab_size, bias=head_bias)

        if tie_weights:
            self.head.weight = self.vocab_embed.weight

        self.apply(self._init_weights)

    def forward(self, x: Tensor, return_preds:bool=True):
        tokens = self.vocab_embed(x)
      
        pos = self.pos_encode(x)

        x = self.embed_drop(tokens + pos)

        for block in self.tfm_blocks:
            x = block(x)

        x = self.head_norm(x)

        if return_preds:
            # converts input token vectors of shape (B, S, C) to probability
            # distribution of shape batch, sequence length, vocabulary size (B, S, VS)
            return self.head(x)
        else:
            return x

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


class BERTForMaskedLM(BERT):
    def __init__(self, loss_fn:nn.Module=nn.CrossEntropyLoss(),
                 mlm_prob:float|None=None, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = loss_fn
        self.mlm_prob = mlm_prob

    def forward(self, input_ids: Tensor, labels: Tensor, mlm_prob: float|None = None):
        x = super().forward(input_ids, False)

        # flatten both the labels and the intermediate outputs
        labels = labels.view(-1)
        x = x.view(labels.shape[0], -1)

        # only select the masked tokens for predictions
        mask_tokens = labels != self.loss_fn.ignore_index

        mlm_prob = self.mlm_prob if mlm_prob is None else mlm_prob
        if mlm_prob is not None:
            num_masks = math.floor(self.mlm_prob * mask_tokens.shape[0])
        else:
            num_masks = mask_tokens.sum().int()
        indices = torch.argsort(mask_tokens.int())[-num_masks:]

        x = x[indices]
        labels = labels[indices]

        logits = self.head(x)

        return {'logits': logits, 'loss': self.loss_fn(logits, labels)}
