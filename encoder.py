from typing import Tuple
import torch as th
import torch.nn as nn

from common import Speaker, TTSConv

class Prenet(nn.Module):
    def __init__(self, 
        vocabulary_size, 
        text_embedding_dim, 
        speaker_embedding_dim, 
        hidden_dim,             
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.projection_fc = nn.Linear(text_embedding_dim, hidden_dim)
        self.speaker = Speaker(speaker_embedding_dim, hidden_dim)

        #TODO: custom embeddings, phoneme support? 
        #TODO: paper doesnt mention positional encodings? which seems weird. 
        self.embedding_layer = nn.Embedding(vocabulary_size, text_embedding_dim)

    
    def forward(self, text, speaker_embedding = None) -> Tuple[th.Tensor, th.Tensor]:
        text_embedding = self.embedding_layer(text)

        if speaker_embedding:
            speaker_bias = self.speaker(speaker_embedding)
            text_embedding = text_embedding + speaker_bias

        output = self.projection_fc(text_embedding)

        return (output, text_embedding)
    
class PostNet(nn.Module):        
    def __init__(self,
        speaker_embedding_dim, 
        hidden_dim, 
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.speaker = Speaker(speaker_embedding_dim, hidden_dim)
        self.input_fc = nn.Linear(hidden_dim, hidden_dim)
        

    def forward(self, input, speaker_embedding = None) -> th.Tensor:
        output = self.input_fc(input)
        if speaker_embedding:
            speaker_bias = self.speaker(speaker_embedding)            
            output += speaker_bias

        return output

class Encoder(nn.Module):
    def __init__(self, 
        vocabulary_size,
        text_embedding_dim,
        speaker_embedding_dim,
        dropout,
        convolutions, 
        conv_dim,
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.pre_net = Encoder.Prenet(vocabulary_size, text_embedding_dim, speaker_embedding_dim, conv_dim)

        self.conv_block = nn.ModuleList(
            TTSConv(dropout, conv_dim, speaker_embedding_dim) 
            for _ in range(convolutions)
        )

        self.post_net = Encoder.PostNet(speaker_embedding_dim, conv_dim)        

    def forward(self, input, speaker_embedding = None):
        pre_out: th.Tensor
        pre_embedding: th.Tensor
        pre_out, pre_embedding = self.pre_net(input, speaker_embedding)

        # pre_out outputs       (batch, seq_len, emb_dim)
        # convolution expects   (batch, emb_dim, seq_len)
        conv_out = pre_out.transpose(1, 2)
        for conv in self.conv_block:
            conv_out = conv(conv_out, speaker_embedding)

        #unwind the transposition    
        conv_out = conv_out.transpose(1,2)
        post_keys = self.post_net(conv_out, speaker_embedding)
        post_values = th.sqrt(.5) * (post_keys + pre_embedding)

        return (post_keys, post_values)
