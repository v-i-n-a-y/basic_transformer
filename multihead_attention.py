"""
Vinay Williams
8th May 2024
Basic Transformer - Multihead attention
"""

import math

from torch import nn
import torch


class MultiHeadAttention(nn.Module):
    """
    Multihead Attention
    """

    def __init__(self, d_model, num_heads):
        """
        Constructor

        Arguments:

        d_model - int
            Model's Embedding Dimension
        num_heads - int
            Number of attention heads
        """
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """
        Scaled dot product attention

        Arguments

        Q - PyTorch Tensor
            Queries
        K - PyTorch Tensor
            Keys
        V - PyTorch Tensor
            Values
        """

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, v)
        return output

    def split_heads(self, x):
        """
        Splits Heads

        Arguments

        x - PyTorch Tensor
            Input
        """

        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        """
        Combine Attention Heads

        Arguments

        x - PyTorch Tensor
            Input
        """

        batch_size, _, seq_length, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, q, k, v, mask=None):
        """
        Forward Call

        Arguments

        Q - PyTorch Tensor
            Queries
        K - PyTorch Tensor
            Keys
        V - PyTorch Tensor
            Values
        """
        q = self.split_heads(self.w_q(q))
        k = self.split_heads(self.w_k(k))
        v = self.split_heads(self.w_v(v))

        attn_output = self.scaled_dot_product_attention(q, k, v, mask)

        return self.w_o(self.combine_heads(attn_output))
