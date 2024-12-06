import torch as th
import torch.nn as nn
from common import Speaker, TTSConv

class Prenet(nn.Module):
    """
    Mixes mel spec with speaker embeddings to pass it off to convolutions
    """
    def __init__(self, mel_spec_dim, hidden_dim, speaker_embedding_dim, dropout, n_layers, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dropout = nn.Dropout(dropout)
        
        self.n_layers = n_layers
        self.fc = nn.ModuleList()
        self.relu = nn.ModuleList()
        self.speaker = nn.ModuleList()

        for i in range(n_layers):
            self.fc.append(nn.Linear(mel_spec_dim if i == 0 else hidden_dim, hidden_dim))
            self.relu.append(nn.ReLU())
            self.speaker.append(Speaker(speaker_embedding_dim, hidden_dim))

    def forward(self, input, speaker_embedding):
        output = input
        
        for i in range(self.n_layers):
            if i != 0:
                output = self.dropout(output) 
                        
            output += self.speaker[i](speaker_embedding)
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
        
        
    def forward(self, query, keys, values, speaker_embedding, current_mel_pos ):
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
        
        speaker_key_weight = self.speakers[0](speaker_embedding)
        speaker_key_weight *= self.w_initial 
        keys += self.position_encoding(speaker_key_weight) 

        speaker_query_weight = self.speakers[1](speaker_embedding)
        speaker_query_weight *= 2
        query += self.position_encoding(speaker_query_weight, current_mel_pos) 

        keys = self.key_fc(keys)
        query = self.query_fc(query)

        output = th.bmm(query, keys.transpose(1, 2))
        output = self.softmax(output)
        output = self.dropout(output)
        
        output = th.bmm(output, values)
        output = output / th.sqrt( values.size(1) ) # not really sure but it looks like timesteps is synonymous with 
        output = self.output_fc(output)
        return output


    def position_encoding(self, length, timesteps, position_rate, current_pos = 0):
        position = th.arange(length).unsqueeze(1)
        factor = th.exp( -th.arange(timesteps) * th.log(10000) / (timesteps - 1) )

        pe = th.zeros(length, timesteps)
        pe[:, 0::2] = th.sin(position_rate * position * factor)[0::2]
        pe[:, 1::2] = th.cos(position_rate * position * factor)[1::2]

class DecoderCore(nn.Module):
    def __init__(self, speaker_embedding_dim, hidden_dim, dropout, n_layers, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.convs = nn.ModuleList()
        self.attns = nn.ModuleList()

        for _ in range(n_layers):
            self.convs.append(TTSConv(dropout, hidden_dim, speaker_embedding_dim, causal_padding=True))
            self.attns.append(Attention(speaker_embedding_dim, hidden_dim))

    def forward(self, input, enc_keys, enc_values, speaker_embedding):
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

    def forward(self, mel_input, encoder_keys, encoder_values, speaker_embedding ):
        
        prenet_out = self.prenet(mel_input, speaker_embedding)
        attn_out = self.decoder_core(prenet_out, encoder_keys, encoder_values)

        done_out = self.done(attn_out)

        mel_out_gate = self.mel_sigmoid_fc(attn_out)
        mel_out_gate = self.mel_sigmoid(mel_out_gate)

        mel_out = self.mel_fc(attn_out)

        return (attn_out, mel_out * mel_out_gate, done_out)




