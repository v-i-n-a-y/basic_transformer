"""
Vinay Williams
8th May 2024
Basic Transformer - decoder
"""

from torch import nn

from multihead_attention import MultiHeadAttention
from positionwisefeedforward import PositionWiseFeedForward


class DecoderLayer(nn.Module):
    """
    Tranformer Decoder
    """

    def __init__(self, d_model, num_heads, d_ff, dropout):
        """
        Constructor

        Arguments

        d_model
            Dimensionality of input
        num_heads - int
            Number of attention heads
        d_ff - int
            Dimensionality of feed forward latent space
        dropout - float
            Rate of dropout
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        """
        Forward Call

        Arguments

        x - PyTorch Tensor
            Input
        enc_output - PyTorch Tensor
            Encoder's Output
        src_mask - PyTorch Tensor
            Source Mask
        tgt_mask - PyTorch Tensor
            Target Mask
        """
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x
