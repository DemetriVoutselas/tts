import torch as th
import torch.nn as nn


class Speaker(nn.Module):
    """
    Speaker embedding block illustrated in figure 1 use throughout
    
    Args:
        speaker_embedding_dim (int): Input speaker embedding dimension
        out_dim (int): Output dimension after projection
    """
    def __init__(self, speaker_embedding_dim, hidden_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fc = nn.Linear(speaker_embedding_dim, hidden_dim)
        self.fc = nn.utils.parametrizations.weight_norm(self.fc)

        self.softsign = nn.Softsign()

    def forward(self, speaker_embedding):
        out = self.fc(speaker_embedding)
        
        return self.softsign(out)
    

class TTSConv(nn.Module):
    """
    The default convolution block as described in Section 3.3.
    
    Args:
        conv_channels (int): Number of conv channels
        speaker_embedding_dim (int): Dimension of speaker embeddings
        dropout (float): Dropout probability
        causal_padding (bool): Applies causal left padding as required by the decoder
    """
        
    def __init__(
        self, 
        conv_channels: int,
        speaker_embedding_dim: int,
        dropout: float,
        causal_padding: bool = False,
        kernel_size: int = 5,  
        *args, 
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        
        assert kernel_size % 2, "kernel size must be odd"

        self.dropout = nn.Dropout(dropout)
        self.padding = nn.ZeroPad1d((kernel_size - 1, 0) if causal_padding else ((kernel_size - 1) // 2, (kernel_size - 1) // 2))
        self.conv = nn.Conv1d(conv_channels, conv_channels*2, kernel_size= kernel_size, stride=1, padding = 0)        
        self.conv = nn.utils.parametrizations.weight_norm(self.conv)
        self.sigmoid = nn.Sigmoid()
        self.speaker = Speaker(speaker_embedding_dim, conv_channels)
        
    def forward(self, input: th.Tensor, speaker_embedding: th.tensor) -> th.Tensor:
        do_input = self.dropout(input)        
        padded_do_input = self.padding(do_input)

        conv_out = self.conv(padded_do_input)
        values, gates = conv_out.chunk(2, dim=1)

        speaker_bias = self.speaker(speaker_embedding)
        
        output = (values + speaker_bias) * self.sigmoid(gates) + input
        
        return output * th.sqrt(.5)