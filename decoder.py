import torch as th
import torch.nn as nn
from common import Speaker, TTSConv

class Prenet(nn.Module):
    def __init__(self, speaker_embedding_dim, out_dim, dropout, n_layers, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.speaker = Speaker(speaker_embedding_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.n_layers = n_layers
        self.fc = nn.ModuleList(nn.Linear(out_dim, out_dim) for _ in range(n_layers))
        self.relu = nn.ModuleList(nn.ReLU() for _ in range(n_layers))

    def forward(self, input, speaker_embedding = None):
        output = input
        
        for i in range(self.n_layers):
            #no dropout in the first layer
            if i != 0:
                output = self.dropout(output) 

            if speaker_embedding:
                speaker_bias = self.speaker(speaker_embedding)
                output += speaker_bias

            output = self.fc[i](output)
            output = self.relu[i](output)

        return output

class Attention(nn.Module):
    def __init__(self, speaker_embedding_dim, hidden_dim, dropout, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.speakers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(speaker_embedding_dim, hidden_dim),
                nn.Sigmoid()
            )
            for _ in range(2)
        )

        self.query_fc = nn.Linear(hidden_dim, hidden_dim)
        self.key_fc = nn.Linear(hidden_dim, hidden_dim)
        self.value_fc = nn.Linear(hidden_dim, hidden_dim)
        self.output_fc = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(-1)
        
        
    def forward(self, query, keys, values, speaker_embedding = None):
        """
        Parameters
        -
        query: tensor from decoder conv 
            (batch, time_steps, hidden_dim)
        keys: tensor from encoder 
            (batch, text_length, hidden_dim)
        values: tensor from encoder 
            (batch, text_length, hidden_dim)
        """
        if speaker_embedding:
            speaker_key_weight = self.speakers[0](speaker_embedding)
            speaker_key_weight *= self.w_initial
            speaker_key_weight += self.position_encoding(speaker_key_weight)  
            keys += speaker_key_weight

            speaker_query_weight = self.speakers[1](speaker_embedding)
            speaker_query_weight *= 2
            speaker_query_weight += self.position_encoding(speaker_query_weight)  
            query += speaker_query_weight

        keys = self.key_fc(keys)
        query = self.query_fc(query)

        output = th.bmm(query, keys.transpose(1, 2))
        output = self.softmax(output)
        output = self.dropout(output)
        
        output = th.bmm(output, values)
        output = output / th.sqrt( values.size(1) ) # not really sure but it looks like timesteps is synonymous with 
        output = self.output_fc(output)
        return output


    def position_encoding(self, length, channels, position_rate):
        position = th.arange(length).unsqueeze(1)
        factor = th.exp( -th.arange(channels) * th.log(10000) / (channels - 1) )

        pe = th.zeros(length, channels)
        pe[:, 0::2] = th.sin(position_rate * position * factor)[0::2]
        pe[:, 1::2] = th.cos(position_rate * position * factor)[1::2]

class DecoderCore(nn.Module):
    def __init__(self, speaker_embedding_dim, hidden_dim, dropout, n_layers, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.convs = nn.ModuleList(
            TTSConv(dropout, hidden_dim, speaker_embedding_dim, causal_padding=True)
            for _ in range(n_layers)
        )

        self.attns = nn.ModuleList(
            Attention(speaker_embedding_dim, hidden_dim)
            for _ in range(n_layers)
        )

    def forward(self, input, enc_keys, enc_values, speaker_embedding = None):
        out = input
        for i in range(self.n_layers):
            conv_out = self.convs[i](out, speaker_embedding)
            att_out = self.attns[i](out, enc_keys, enc_values, speaker_embedding)
            out = th.sqrt(.5) * (att_out + conv_out)

        return out
    
class Done(nn.Module):
    def __init__(self, hidden_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.fc(input)
        output = self.sigmoid(output)

        return output > .5

class Decoder(nn.Module):
    def __init__(self, speaker_embedding_dim, hidden_dim, dropout, n_layers, mel_dim, red_factor, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_layers = n_layers
        self.red_factor = red_factor

        self.done = Done(hidden_dim)        
        self.prenet = Prenet(speaker_embedding_dim, hidden_dim, dropout, n_layers)
        self.decoder_core = DecoderCore(speaker_embedding_dim, hidden_dim, dropout, n_layers)
        
        self.mel_fc = nn.Linear(hidden_dim, mel_dim)
        self.mel_sigmoid_fc = nn.Linear(hidden_dim, mel_dim)
        self.mel_sigmoid = nn.Sigmoid()

    def forward(self, mel_input, encoder_keys, encoder_values, speaker_embedding_dim = None):
        prenet_out = self.prenet(mel_input, speaker_embedding_dim)
        dec_out = self.decoder_core(prenet_out, encoder_keys, encoder_values)

        done_out = self.done(dec_out)

        mel_out_gate = self.mel_sigmoid_fc(dec_out)
        mel_out_gate = self.mel_sigmoid(mel_out_gate)

        mel_out = self.mel_fc(dec_out)

        return (dec_out, mel_out * mel_out_gate, done_out)

        #it is unclear what is fed into the mel_sigmoid_fc. will assume its decoder_core out


