from typing import Tuple
import torch as th
import torch.nn as nn

class Encoder(nn.Module):


    class Prenet(nn.Module):
        def __init__(self, 
            vocabulary_size, 
            text_embedding_dim, 
            speaker_embedding_dim, 
            conv_dim,             
            *args, 
            **kwargs
        ):
            super().__init__(*args, **kwargs)

            self.projection_fc = nn.Linear(text_embedding_dim, conv_dim)
            self.speaker_embedding_fc = nn.Linear(speaker_embedding_dim, text_embedding_dim)
            self.softsign = nn.Softsign()

            #TODO: custom embeddings, phoneme support? 
            #TODO: paper doesnt mention positional encodings? which seems weird. 
            self.embedding_layer = nn.Embedding(vocabulary_size, text_embedding_dim)

        
        def forward(self, text, speaker_embedding = None) -> Tuple[th.Tensor, th.Tensor]:
            text_embedding = self.embedding_layer(text)

            if speaker_embedding:
                speaker_bias = self.speaker_embedding_fc(speaker_embedding)
                speaker_bias = self.softsign(speaker_bias)
                text_embedding = text_embedding + speaker_bias

            output = self.projection_fc(text_embedding)

            return (output, text_embedding)
        
    class PostNet(nn.Module):        
        def __init__(self,
            speaker_embedding_dim, 
            conv_dim, 
            *args, 
            **kwargs
        ):
            super().__init__(*args, **kwargs)

            self.speaker_embedding_fc = nn.Linear(speaker_embedding_dim, conv_dim)
            self.input_fc = nn.Linear(conv_dim, conv_dim)
            self.softsign = nn.Softsign()

        def forward(self, input, speaker_embedding = None) -> th.Tensor:
            output = self.input_fc(input)
            if speaker_embedding:
                speaker_bias = self.speaker_embedding_fc(speaker_embedding)
                speaker_bias = self.softsign(speaker_bias)
                output += speaker_bias

            return output


    class EmbeddingConvolution(nn.Module):
        def __init__(
            self, 
            dropout, 
            conv_dim, 
            speaker_embedding_dim,
            
            *args, 
            **kwargs
        ):
            super().__init__(*args, **kwargs)
            
            self.dropout = nn.Dropout(dropout)
            self.convolution = nn.Conv1d(conv_dim, conv_dim*2, kernel_size= 5, stride=1, padding = 2)
            self.softsign = nn.Softsign()
            self.sigmoid = nn.Sigmoid()
            self.speaker_embedding_fc = nn.Linear(speaker_embedding_dim, conv_dim)

        def forward(self, input, speaker_embedding = None) -> th.Tensor:
            do_input = self.dropout(input)        
            conv_out = self.convolution(do_input)

            values, gates = conv_out.chunk(2, dim=1)

            if speaker_embedding:
                speaker_bias = self.speaker_embedding_fc(speaker_embedding)
                speaker_bias = self.softsign(speaker_bias)
                values += speaker_bias

            output = values * self.sigmoid(gates) + input
            return output * th.sqrt(.5)


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
            Encoder.EmbeddingConvolution(dropout, conv_dim, speaker_embedding_dim) 
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
