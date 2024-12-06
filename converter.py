import torch as th
import torch.nn as nn

from common import TTSConv

class Converter(nn.Module):
    def __init__(self, hidden_dim, speaker_embedding_dim, spectrogram_dim, conv_layers, sharpening_factor, dropout, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv = nn.ModuleList(
            TTSConv(
                dropout= dropout,
                conv_channels= hidden_dim,
                speaker_embedding_dim= speaker_embedding_dim,
                causal_padding= False
            )
            for _ in range(conv_layers)            
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, spectrogram_dim),
        )

        

    def forward(self, decoder_out, speaker_embedding = None):
        conv_out = decoder_out
        for i in range(len(self.conv)):
            conv_out = self.conv[i](conv_out, speaker_embedding)
        
        fc_out = self.fc(conv_out)

        # fc_out should be the lin spectrogram
        return fc_out 