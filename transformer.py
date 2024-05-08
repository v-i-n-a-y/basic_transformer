"""
Vinay Williams
8th May 2024
Basic Transformer - Transformer
"""

from torch import nn
import torch

from encoder import EncoderLayer
from decoder import DecoderLayer

from positional_encoding import PositionalEncoding


class Transformer(nn.Module):
    """
    Transformer
    """

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        max_seq_length,
        dropout,
    ):
        """
        Constructor

        Arguments

        src_vocab_size - int
            Source vocabulary Size
        tgt_vocab_size - int
            Target vocabulary size
        d_model -
            Dimensionality of input
        num_heads - int
            Number of attention heads
        num_layers - int
            Number of encoder and decoder layers
        d_ff -
            Dimensionality of feed forward latent space
        max_seq_length - int
            Maximum sequence length
        dropout - float
            Dropout rate
        """
        super().__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        """
        Generate mask

        Arguments

        src - PyTorch Tensor
            source data
        tgt - PyTorch Tensor
            target data
        """
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (
            1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)
        ).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        """
        Forward Call

        Arguments

        src - PyTorch Tensor
            source data
        tgt - PyTorch Tensor
            target data
        """

        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(
            self.positional_encoding(self.encoder_embedding(src))
        )
        tgt_embedded = self.dropout(
            self.positional_encoding(self.decoder_embedding(tgt))
        )

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        return self.fc(dec_output)


if __name__ == "__main__":
    src_vocab_size = 5000
    tgt_vocab_size = 5000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_length = 100
    dropout = 0.1

    transformer = Transformer(
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        max_seq_length,
        dropout,
    )

    src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))
    tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))

    output = transformer(src_data, tgt_data)

    print(output.size())
