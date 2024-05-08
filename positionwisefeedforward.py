"""
Vinay Williams
8th May 2024
Basic Transformer - Position wise feed forward
"""

from torch import nn


class PositionWiseFeedForward(nn.Module):
    """
    Position Wise Feed Forward
    """

    def __init__(self, d_model, d_ff):
        """
        Constructor

        Arguments:

        d_model - int
            Model's embedding dimension
        d_ff - int
            Feed Forward's latent dimension
        """
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward Call

        Arguments:

        x - PyTorch Tensor
            Input
        """
        return self.fc2(self.relu(self.fc1(x)))
