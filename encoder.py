"""
Vinay Williams
8th May 2024
Basic Transformer - encoder
"""

from torch import nn

from multihead_attention import MultiHeadAttention
from positionwisefeedforward import PositionWiseFeedForward


class EncoderLayer(nn.Module):
    """
    Transformer Encoder
    """

    def __init__(self, d_model, num_heads, d_ff, dropout):
        """
        Constructor

        Arguments

        d_model -
            Dimensionality of input
        num_heads - int
            Number of Attention heads
        d_ff -
            Dimensionality of feed forward latent space
        dropout - float
            Dropout rate

        """
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        Forward Call

        Arguments:

        x - PyTorch Tensor
            Input
        mask - PyTorch Tensor
            Mask
        """
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
