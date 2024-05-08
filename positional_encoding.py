"""
Vinay Williams
8th May 2024
Basic Transformer - Positional Encoding
"""

import math

from torch import nn
import torch


class PositionalEncoding(nn.Module):
    """
    Positional Encoding
    """

    def __init__(self, d_model, max_seq_length):
        """
        Constructor

        Arguments

        d_model - int
            Model's embedding dimension
        max_seq_length - int
            Maximum sequence length
        """
        super().__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        """
        Forward Call

        Arguments:

        x - PyTorch Tensor
            Input
        """
        return x + self.pe[:, : x.size(1)]
