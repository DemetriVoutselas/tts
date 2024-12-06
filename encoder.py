from typing import Tuple
import torch as th
import torch.nn as nn

from common import Speaker, TTSConv


class Encoder(nn.Module):
    """
    Encoder network as described in Section 3.4 of the paper.
    
    Args:
        n_vocab: Size of the symbol vocabulary
        symbol_embedding_dim: Dimension of symbol embeddings
        hidden_dim: Hidden dimension of the network
        speaker_embedding_dim: Dimension of speaker embeddings
        dropout: Dropout probability
        n_conv: Number of convolution blocks
    """
        
    def __init__(self, 
        n_vocab,
        symbol_embedding_dim,
        hidden_dim,
        speaker_embedding_dim,
        dropout,
        n_conv,         
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.prenet_embedding = nn.Embedding(n_vocab, symbol_embedding_dim)
        self.prenet_speaker = Speaker(speaker_embedding_dim, hidden_dim)
        self.prenet_fc = nn.Linear(symbol_embedding_dim, hidden_dim)
        self.prenet_fc = nn.utils.parametrizations.weight_norm(self.prenet_fc)

        self.conv_block = nn.ModuleList(
            TTSConv(hidden_dim, speaker_embedding_dim, dropout) 
            for _ in range(n_conv)
        )

        self.postnet_speaker = Speaker(speaker_embedding_dim, hidden_dim)
        self.postnet_fc = nn.Linear(hidden_dim, hidden_dim)
        self.postnet_fc = nn.utils.parametrizations.weight_norm(self.postnet_fc)
           

    def forward(self, symbols, speaker_embedding):
        prenet_out: th.Tensor
        symbol_embedding: th.Tensor

        #prenet
        symbol_embedding = self.prenet_embedding(symbols)     
        prenet_speaker_bias = self.prenet_speaker(speaker_embedding)   
        prenet_out = symbol_proj + prenet_speaker_bias

        symbol_proj = self.prenet_fc(symbol_embedding) # (batch, timesteps, conv channels)

        
        conv_out = symbol_proj.transpose(1, 2)

        for conv in self.conv_block:
            conv_out = conv(conv_out, speaker_embedding)

        #unwind the transposition    
        conv_out = conv_out.transpose(1,2)

        #postnet
        postnet_speaker_bias = self.postnet_speaker(speaker_embedding)
        postnet_keys = self.postnet_fc(conv_out) + postnet_speaker_bias
        postnet_values = th.sqrt(.5) * (postnet_keys + symbol_embedding)

        return (postnet_keys, postnet_values)
