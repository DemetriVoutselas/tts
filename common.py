import torch as th
import torch.nn as nn

SPEAKER_EMBEDDING_DIM = 128
HIDDEN_DIM = 128

class Speaker(nn.Module):
    def __init__(self, speaker_embedding_dim, out_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fc = nn.Linear(speaker_embedding_dim, out_dim)
        self.softsign = nn.Softsign()

    def forward(self, speaker_embedding):
        return self.softsign(self.fc(speaker_embedding))
    
#TODO: hyperparameters
class TTSConv(nn.Module):
    def __init__(
        self, 
        dropout, 
        hidden_dim, 
        speaker_embedding_dim,
        
        causal_padding = False,
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.dropout = nn.Dropout(dropout)
        self.padding = nn.ZeroPad1d((4, 0) if causal_padding else 2)
        self.convolution = nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size= 5, stride=1, padding = 0)        
        self.sigmoid = nn.Sigmoid()
        self.speaker = Speaker(speaker_embedding_dim, hidden_dim)
        
    def forward(self, input, speaker_embedding = None) -> th.Tensor:
        do_input = self.dropout(input)        
        padded_do_input = self.padding(do_input)
        conv_out = self.convolution(padded_do_input)

        values, gates = conv_out.chunk(2, dim=1)

        if speaker_embedding:
            speaker_bias = self.speaker_embedding_fc(speaker_embedding)
            speaker_bias = self.softsign(speaker_bias)
            values += speaker_bias

        output = values * self.sigmoid(gates) + input
        
        return output * th.sqrt(.5)