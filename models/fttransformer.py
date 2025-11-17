"""From https://github.com/sb-ai-lab/LightAutoML/blob/master/lightautoml/ml_algo/torch_based/fttransformer/fttransformer_utils.py"""

"""Feedforward and Attention blocks for FTTransformer (https://arxiv.org/abs/2106.11959v2) from https://github.com/lucidrains/tab-transformer-pytorch/tree/main."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

from .pooling import pooling_by_name, SequenceIndentityPooler
from .embed import LinearEmbedding


class GEGLU(nn.Module):
    """GEGLU activation for Attention block."""

    def forward(self, x):
        """Forward pass."""
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


def FeedForward(dim, mult=4, dropout=0.0):
    """Feedforward for Transformer block."""
    return nn.Sequential(
        nn.LayerNorm(dim), nn.Linear(dim, dim * mult * 2), GEGLU(), nn.Dropout(dropout), nn.Linear(dim * mult, dim)
    )


class Attention(nn.Module):
    """Attention Block.

    Args:
            dim: Embeddings dimension.
            heads: Number of heads in Attention.
            dim_head: Attention head dimension.
            dropout: Post-Attention dropout.
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Transform the input tensor with attention.

        Args:
            x : torch.Tensor
                3-d tensor; for example, embedded numeric and/or categorical values,
                or the output of a previous attention layer.

        Returns:
            torch.Tensor

        """
        batch_size, seq_len, dim = x.shape
        h = self.heads

        x = self.norm(x)
        qkv = self.to_qkv(x)
        q, k, v = torch.split(qkv, qkv.size(-1) // 3, dim=-1)

        q = q.view(batch_size, seq_len, h, -1).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, h, -1).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, h, -1).permute(0, 2, 1, 3)

        q = q * self.scale
        sim = torch.matmul(q, k.transpose(-2, -1))
        attn = torch.softmax(sim, dim=-1)
        dropped_attn = self.dropout(attn)

        out = torch.matmul(dropped_attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)
        out = self.to_out(out)

        return out, attn


# transformer
class Transformer(nn.Module):
    """Transformer Block.

    Args:
            dim: Embeddings dimension.
            depth: Number of Attention Blocks.
            heads: Number of heads in Attention.
            dim_head: Attention head dimension.
            attn_dropout: Post-Attention dropout.
            ff_dropout: Feed-Forward Dropout.
            return_attn: Return attention scores or not.
    """

    def __init__(self, dim, depth, heads, dim_head, attn_dropout, ff_dropout):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout),
                        FeedForward(dim, dropout=ff_dropout),
                    ]
                )
            )

    def forward(self, x):
        """Transform the input embeddings tensor with Transformer module.

        Args:
            x : torch.Tensor
                3-d tensor; embedded numeric and/or categorical values,
                or the output of a previous Transformer layer.

        Returns:
            torch.Tensor

        """
        post_softmax_attns = []

        for attn, ff in self.layers:
            attn_out, post_softmax_attn = attn(x)
            post_softmax_attns.append(post_softmax_attn)
            x = attn_out + x
            x = ff(x) + x

        return x

class FTTransformer(nn.Module):
    """FT Transformer (https://arxiv.org/abs/2106.11959v2) from https://github.com/lucidrains/tab-transformer-pytorch/tree/main.

    Args:
            pooling: Pooling used for the last step.
            n_out: Output dimension, 1 for binary prediction.
            embedding_size: Embeddings size.
            depth: Number of Attention Blocks inside Transformer.
            heads: Number of heads in Attention.
            attn_dropout: Post-Attention dropout.
            ff_dropout: Feed-Forward Dropout.
            dim_head: Attention head dimension.
            num_enc_layers: Number of Transformer layers.
            device: Device to compute on.
    """

    def __init__(
        self,
        # *,
        in_shape,
        num_classes: int = 10,
        pooling: str = "mean",
        # n_out: int = 1,
        embedding_size: int = 32,
        depth: int = 4,
        heads: int = 1,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        dim_head: int = 32,
        num_enc_layers: int = 2,
        device: Union[str, torch.device] = "cuda:0",
        **kwargs,
    ):
        super(FTTransformer, self).__init__()
        self.device = device
        self.pooling = pooling_by_name[pooling]()

        # transformer
        self.transformer = nn.Sequential(
            *nn.ModuleList(
                [
                    Transformer(
                        dim=embedding_size,
                        depth=depth,
                        heads=heads,
                        dim_head=dim_head,
                        attn_dropout=attn_dropout,
                        ff_dropout=ff_dropout,
                    )
                    for _ in range(num_enc_layers)
                ]
            )
        )

        # to logits
        if pooling == "concat":
            self.to_logits = nn.Sequential(nn.BatchNorm1d(embedding_size * 2), nn.Linear(embedding_size * 2, num_classes))
        else:
            self.to_logits = nn.Sequential(nn.BatchNorm1d(embedding_size), nn.Linear(embedding_size, num_classes))

        self.cls_token = nn.Embedding(2, embedding_size)
        self.embed = LinearEmbedding(num_dims=in_shape)

    def forward(self, embedded):
        """Transform the input tensor.

        Args:
            embedded : torch.Tensor
                embedded fields

        Returns:
            torch.Tensor

        """
        embedded = self.embed(embedded)
        cls_token = torch.unsqueeze(
            self.cls_token(torch.ones(embedded.shape[0], dtype=torch.int).to(self.device)), dim=1
        )
        x = torch.cat((cls_token, embedded), dim=1)
        x = self.transformer(x)
        x_mask = torch.ones(x.shape, dtype=torch.bool).to(self.device)
        pool_tokens = self.pooling(x=x, x_mask=x_mask)
        if isinstance(self.pooling, SequenceIndentityPooler):
            pool_tokens = pool_tokens[:, 0]

        logits = self.to_logits(pool_tokens)
        return logits