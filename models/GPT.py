import math
import torch
import torch.nn as nn
from torch import Tensor, BoolTensor
from torch.nn import functional as F
from layers.attention import CausalSelfAttention

class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, expand_size: int, act=nn.GELU, drop:float=0.1, bias:bool=True):
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
    def __init__(self, hidden_size: int, expand_size: int, context_size: int, num_heads: int, attention=CausalSelfAttention, act=nn.GELU, attn_drop:float=0.1, out_drop:float=0.1, ffn_drop:float=0.1, bias:bool=True):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_size)

        self.attn = attention(hidden_size=hidden_size, num_heads=num_heads, context_size=context_size, attn_drop=attn_drop, out_drop=out_drop, bias=bias)

        self.norm2 = nn.LayerNorm(hidden_size)

        self.ffn = FeedForward(hidden_size=hidden_size, expand_size=expand_size, act=act, drop=ffn_drop, bias=bias)

    def forward(self, x:Tensor):
        x += self.attn(self.norm1(x))
        x += self.ffn(self.norm2(x))

        return x

class GPT(nn.Module):
    def __init__(self, num_layers:int, vocab_size:int, hidden_size:int, num_heads:int, context_size:int, expand_size:int, attention=CausalSelfAttention, act=nn.GELU, embed_drop:float=0.1, attn_drop:float=0.1, out_drop:float=0.1, ffn_drop:float=0.1, head_norm:bool=True, tie_weights:bool=True, head_bias:bool=True, bias:bool=True):
        super().__init__()

        self.vocab_embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.Embedding(context_size, hidden_size)
        self.embed_drop = nn.Dropout(embed_drop)

        self.tfm_blocks = nn.ModuleList([TransformerBlock(hidden_size=hidden_size, num_heads=num_heads, context_size=context_size, expand_size=expand_size, attention=attention, act=act, bias=bias, attn_drop=attn_drop, out_drop=out_drop, ffn_drop=ffn_drop)
            for _ in range(num_layers)])

        self.head_norm = nn.LayerNorm(hidden_size) if head_norm else nn.Identity()

        self.head = nn.Linear(hidden_size, vocab_size, bias=head_bias)

        if tie_weights:
            self.head.weight = self.vocab_embed.weight

        pos = torch.arange(0, context_size, dtype=torch.long)
        self.register_buffer('pos', pos, persistent=False)

        self.apply(self._init_weights)

    def forward(self, x:Tensor):
        tokens = self.vocab_embed(x)

        pos = self.pos_embed(self.pos[:x.shape[1]])

        x = self.embed_drop(tokens + pos)

        for block in self.tfm_blocks:
            x = block(x)

        X = self.head_norm(x)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if module._get_name() == 'fc2':
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02/math.sqrt(2 * self.num_layers))
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

class GPTForCausalLM(GPT):
    def __init__(self, loss_fn:nn.Module=nn.CrossEntropyLoss(), **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = loss_fn

    def forward(self, x: Tensor):

        inputs = x[:, :-1]
        labels = x[:, 1:]

        logits = super().forward(inputs)

        loss = self.loss_fn(logits.view(-1, logits.shape[-1], labels.view(-1)))

        return {'logits': logits, 'loss': loss}
