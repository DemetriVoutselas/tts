from dataclasses import astuple
from pprint import pprint
from random import random
import numpy as np
import torch as th
import torch.nn as nn

from hparams import HParams
import phoneme
from preprocessing import get_saved_data, get_speaker_id_map, get_symbol_sequence, pad_tts_data, reconstruct_audio


from torch.utils.data import Dataset, DataLoader
class TTSDataset(Dataset):
    def __init__(self, tts_data):
        super().__init__()

        self.data = tts_data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]



def collate_fn(batch):
    n = len(batch)
    speaker_ids = []
    texts = []
    T_actual = []
    T_max = max(it.lin_spec.size(-1) for it in batch)
    
    padded_lin_specs = th.zeros((n, batch[0].lin_spec.size(0), T_max))
    padded_mel_specs = th.zeros((n, batch[0].mel_spec.size(0), T_max))

    for i, it in enumerate(batch):
        speaker_ids.append(it.speaker_id)
        texts.append(it.phoneme if it.phoneme and random() < .5 else it.text)
        T_actual.append(it.T)
        padded_lin_specs[i, :, :it.T] = it.lin_spec
        padded_mel_specs[i, :, :it.T] = it.mel_spec

    ##transpose specs so that T is the 2nd dimension
    #padded_lin_specs = padded_lin_specs.transpose(1, 2)
    #padded_mel_specs = padded_mel_specs.transpose(1, 2)

    symbol_seqs = get_symbol_sequence(texts)

    text_max = max(len(seq) for seq in symbol_seqs)
    symbols = th.full((n, text_max), phoneme.PAD)
    for i, seq in enumerate(symbol_seqs):
        symbols[i, :len(seq)] = th.tensor(seq).unsqueeze(0)

    return (
        speaker_ids,
        symbols, 
        th.tensor(T_actual),
        padded_lin_specs,
        padded_mel_specs
    )

class Speaker(nn.Module):
    """
    Projects speaker embedding into a hidden dimension space with softsign activation.
    Used to add a speaker-dependent bias to convolutional blocks.

    Input shape:
      speaker_embedding: (B, speaker_embedding_dim)

    Output shape:
      (B, hidden_dim)
    """
    def __init__(self, speaker_embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.fc = nn.utils.weight_norm(module=nn.Linear(in_features=speaker_embedding_dim, out_features=hidden_dim))
        self.softsign = nn.Softsign()

    def forward(self, speaker_embedding: th.Tensor) -> th.Tensor:
        # speaker_embedding: (B, speaker_embedding_dim)
        return self.softsign(self.fc(speaker_embedding))  # (B, hidden_dim)

class ConvBlock(nn.Module):
    """
    The default convolution block as described in Section 3.3 of Deep Voice 3.

    Args:
        kernel_size (int): Filter kernel size (must be odd)
        hidden_dim (int): Hidden dimension of input/output channels
        speaker_embedding_dim (int): Dimension of speaker embeddings
        dropout (float): Dropout probability
        causal_padding (bool): Apply causal padding for decoder blocks

    Input shape:
      x: (B, hidden_dim, T)
      speaker_embedding: (B, speaker_embedding_dim)

    Output shape:
      (B, hidden_dim, T)
    """
        
    def __init__(
        self,
        kernel_size: int,
        hidden_dim: int,
        speaker_embedding_dim: int,
        dropout: float,
        causal_padding: bool = False
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd for symmetrical padding."

        self.sqrt12 = np.sqrt(0.5)
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout)

        # Padding for causal or non-causal convolutions
        if causal_padding:
            # For causal: all padding on the left
            padding = (kernel_size - 1, 0)
        else:
            # Symmetrical padding
            p = (kernel_size - 1) // 2
            padding = (p, p)

        self.pad = nn.ZeroPad1d(padding)
        self.conv = nn.utils.weight_norm(
            module=nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim * 2,
                kernel_size=kernel_size,
                stride=1,
                padding=0
            )
        )
        self.sigmoid = nn.Sigmoid()
        self.speaker = Speaker(
            speaker_embedding_dim=speaker_embedding_dim,
            hidden_dim=hidden_dim
        )

        
    def forward(self, x: th.Tensor, speaker_embedding: th.Tensor) -> th.Tensor:
        # x: (B, C=hidden_dim, T)
        B, C, T = x.size()
        do_x = self.dropout(x)  # (B, C, T)
        pad_x = self.pad(do_x)
        conv_out = self.conv(pad_x)  # (B, 2*hidden_dim, T)
        values, gates = conv_out.chunk(2, dim=1)  # (B, hidden_dim, T), (B, hidden_dim, T)

        speaker_bias = self.speaker(speaker_embedding)  # (B, hidden_dim)
        speaker_bias = speaker_bias.unsqueeze(2)  # (B, hidden_dim, 1) for broadcasting over T

        # Gated linear unit with speaker bias
        output = (values + speaker_bias) * self.sigmoid(gates) + x  # Residual connection

        return output * self.sqrt12  # (B, hidden_dim, T)

class Encoder(nn.Module):
    """
    Encoder network: transforms embedded symbols into a set of keys and values for the decoder.

    Args:
        symbol_embedding_dim (int)
        hidden_dim (int)
        speaker_embedding_dim (int)
        dropout (float)
        n_conv (int)
        kernel_size (int)

    Input shape:
      symbol_embedding: (B, T_text, symbol_embedding_dim)
      speaker_embedding: (B, speaker_embedding_dim)

    Output shape:
      keys, values: (B, T_text, hidden_dim)
    """
        
    def __init__(self, 
        symbol_embedding_dim: int,
        hidden_dim: int,
        speaker_embedding_dim: int,
        dropout: float,
        n_conv: int,
        kernel_size: int
    ):
        super().__init__()

        self.prenet_speaker = Speaker(
            speaker_embedding_dim=speaker_embedding_dim,
            hidden_dim=hidden_dim
        )

        self.postnet_speaker = Speaker(
            speaker_embedding_dim=speaker_embedding_dim,
            hidden_dim=hidden_dim
        )

        self.prenet_fc = nn.utils.weight_norm(
            module=nn.Linear(
                in_features=symbol_embedding_dim,
                out_features=hidden_dim
            )
        )

        self.conv_block = nn.ModuleList(
            [
                ConvBlock(
                    kernel_size=kernel_size,
                    hidden_dim=hidden_dim,
                    speaker_embedding_dim=speaker_embedding_dim,
                    dropout=dropout,
                    causal_padding=False
                )
                for _ in range(n_conv)
            ]
        )

        self.postnet_fc = nn.utils.weight_norm(
            module=nn.Linear(
                in_features=hidden_dim,
                out_features=hidden_dim
            )
        )

        self.sqrt12 = np.sqrt(0.5)
           
    def forward(self, symbol_embedding: th.Tensor, speaker_embedding: th.Tensor, symbol_lengths: th.Tensor = None):
        # symbol_embedding: (B, T_text, symbol_embedding_dim)
        # speaker_embedding: (B, speaker_embedding_dim)
        #symbol_lengths: (B,) tensor of actual lengths before padding
        B, T_text, _ = symbol_embedding.size()
        
        # padding mask based on symbols true size
        # TODO do we need symbol masking 
        #symbol_mask = (th.arange(T_text).unsqueeze(0).to(symbol_embedding.device) < symbol_lengths.unsqueeze(1)).unsqueeze(-1)
        
        # Prenet
        symbol_proj = self.prenet_fc(symbol_embedding)  # (B, T_text, hidden_dim)
        prenet_speaker_bias = self.prenet_speaker(speaker_embedding)  # (B, hidden_dim)
        prenet_speaker_bias = prenet_speaker_bias.unsqueeze(1)  # (B,1,hidden_dim) for broadcasting

        prenet_out = symbol_proj  + prenet_speaker_bias  # (B, T_text, hidden_dim)
        conv_out = prenet_out.transpose(1, 2)  # (B, hidden_dim, T_text)

        # Stacked ConvBlocks
        for conv in self.conv_block:
            conv_out = conv(x=conv_out, speaker_embedding=speaker_embedding)

        conv_out = conv_out.transpose(1, 2)  # (B, T_text, hidden_dim)

        # Postnet
        postnet_speaker_bias = self.postnet_speaker(speaker_embedding).unsqueeze(1)  # (B,1,hidden_dim)
        postnet_keys = self.postnet_fc(conv_out) + postnet_speaker_bias  # (B, T_text, hidden_dim)
        postnet_values = self.sqrt12 * (postnet_keys + symbol_embedding)  # (B, T_text, hidden_dim)

        return (postnet_keys, postnet_values)

class Prenet(nn.Module):
    """
    Prenet module for the decoder:
    Mixes the input mel spectrograms with speaker embeddings before passing to convolutions.

    Args:
      mel_spec_dim (int)
      hidden_dim (int)
      speaker_embedding_dim (int)
      dropout (float)
      n_proj_layers (int)

    Input shape:
      input: (B, T_mel, mel_spec_dim)
      speaker_embedding: (B, speaker_embedding_dim)

    Output shape:
      (B, T_mel, hidden_dim)
    """
    def __init__(
        self,
        mel_spec_dim: int,
        hidden_dim: int,
        speaker_embedding_dim: int,
        dropout: float,
        n_proj_layers: int
    ):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.n_proj_layers = n_proj_layers
        self.fc = nn.ModuleList()
        self.relu = nn.ModuleList()
        self.speaker = nn.ModuleList()

        for i in range(n_proj_layers):
            in_dim = mel_spec_dim if i == 0 else hidden_dim
            self.fc.append(nn.Linear(in_features=in_dim, out_features=hidden_dim))
            self.relu.append(nn.ReLU())
            self.speaker.append(Speaker(speaker_embedding_dim=speaker_embedding_dim, hidden_dim=hidden_dim))


    def forward(self, input: th.Tensor, speaker_embedding: th.Tensor) -> th.Tensor:
        # input: (B, T_mel, mel_spec_dim)
        # speaker_embedding: (B, speaker_embedding_dim)
        output = input
        for i in range(self.n_proj_layers):
            if i != 0:
                output = self.dropout(output)

            output = self.fc[i](output)  # (B, T_mel, hidden_dim)   
            output += self.speaker[i](speaker_embedding).unsqueeze(1)  # Add speaker bias (B,1,hidden_dim)
            output = self.relu[i](output)
        return output

class Attention(nn.Module):
    """
    Attention module for the decoder to attend over encoder outputs.

    Uses sinusoidal positional encoding with learned speaker-dependent rates.

    Args:
      speaker_embedding_dim (int)
      hidden_dim (int)
      dropout (float)

    Inputs:
      query: (B, T_dec, hidden_dim)
      keys: (B, T_text, hidden_dim)
      values: (B, T_text, hidden_dim)
      speaker_embedding: (B, speaker_embedding_dim)
      current_mel_pos: int (used in positional encoding)

    Output:
      (B, T_dec, hidden_dim)
    """
        
    def __init__(self, speaker_embedding_dim, hidden_dim, dropout):
        super().__init__()

        self.hidden_dim = hidden_dim

        # we assume multi speaker weights
        self.speaker_query_proj = nn.Sequential(
            nn.Linear(in_features=speaker_embedding_dim, out_features=1),
            nn.Sigmoid()
        )

        self.speaker_key_proj = nn.Sequential(
            nn.Linear(in_features=speaker_embedding_dim, out_features=1),
            nn.Sigmoid()
        )

        self.query_proj = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.key_proj = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.value_proj = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.output_proj = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)

        # Shared weights for query/key projection as in the paper
        # roughly diagonal initialization since thats roughly what we want attentiont o look like
        shared_weights = th.eye(hidden_dim) + th.randn(hidden_dim, hidden_dim)
        shared_weights = shared_weights / np.sqrt(hidden_dim)
        self.query_proj.weight.data = shared_weights
        self.key_proj.weight.data = shared_weights

        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=-1)

        self.sqrt_hidden = th.sqrt(th.tensor(self.hidden_dim))
    
    def position_encoding(self, length: int, d_model: int, position_rate: th.tensor, current_pos: int = 0):
        """
        length: number of positions (timesteps)
        d_model: number of channels (dimensions) in the positional encoding
        position_rate (ω_s): scaling factor for the positions
        device: torch device

        Returns:
        pe: (length, d_model) tensor of positional encodings
        """
        # i: positions from 0 to length-1
        # k: dimension indices from 0 to d_model-1
        B = position_rate.shape[0]
        position = th.arange(current_pos, current_pos + length).unsqueeze(1).float()        # shape (length, 1)
        k = th.arange(d_model).float()                                                      # shape (d_model)

        # denominator 10000^(k/d)
        div_term = (10000.0 ** (k / d_model)).unsqueeze(0)  # (d_model,)

        # 
        position = position.unsqueeze(0)
        position_rate = position_rate.view(-1, 1, 1)

        # angle(i, k) = (ω_s * i) / (10000^(k/d))
        angle = (position_rate * position) / div_term.unsqueeze(0)  # (length, d_model)

        pe = th.zeros(B, length, d_model)
        pe[:, :, 0::2] = th.sin(angle[:, :, 0::2])
        pe[:, :, 1::2] = th.cos(angle[:, :, 1::2])

        return pe

    def get_attn_mask(self, T_dec:int , T_text: int, window = 3):
        """
        Creates a windowed attention mask that enforces monotonicity during inference.
        
        Args:
            T_dec: Decoder sequence length
            T_text: Encoder sequence length (text length)
            window_size: Size of the attention window (paper suggests 3)
            
        Returns:
            Boolean mask of shape (T_dec, T_text) where True values are allowed
        """
        decoder_positions = th.arange(T_dec).unsqueeze(-1)
        lb = (decoder_positions - window).clamp(min=0)
        ub = (decoder_positions + window).clamp(max = T_text - 1)

        text_positions = th.arange(T_text).unsqueeze(0)

        return (text_positions >= lb) & (text_positions <= ub)

    def forward(
        self,
        query: th.Tensor,
        keys: th.Tensor,
        values: th.Tensor,
        speaker_embedding: th.Tensor,
        current_mel_pos: int
    ) -> th.Tensor:
        # query: (B, T_dec, hidden_dim)
        # keys: (B, T_text, hidden_dim)
        # values: (B, T_text, hidden_dim)
        # speaker_embedding: (B, speaker_embedding_dim)
        speaker_key_weight = self.speaker_key_proj(speaker_embedding) # (B, 1)

        speaker_query_weight = self.speaker_query_proj(speaker_embedding) # (B, 1)

        # Add position encoding to keys and query
        B, T_text, D = keys.size()
        _, T_dec, _ = query.size() 

        # position_encoding returns (length, dim), we need to make it (B, length, dim)
        #TODO: validate if current_pos is correct for both of these
        pe_keys = self.position_encoding(length=T_text, d_model=D, position_rate=speaker_key_weight, current_pos=0)
        pe_query = self.position_encoding(length=T_dec, d_model=D, position_rate=speaker_query_weight, current_pos=current_mel_pos)

        keys = keys + pe_keys
        query = query + pe_query

        keys = self.key_proj(keys)       # (B, T_text, hidden_dim)
        query = self.query_proj(query)   # (B, T_dec, hidden_dim)
        values = self.value_proj(values) # (B, T_text, hidden_dim)

        # Compute attention scores: (B, T_dec, T_text)
        attn_scores = th.bmm(query, keys.transpose(1, 2)) 
        attn_mask = self.get_attn_mask(T_dec= T_dec, T_text= T_text, window = 5)
        attn_scores.masked_fill(~attn_mask, -th.inf)
        attn_scores /= self.sqrt_hidden
        attn_weights = self.softmax(attn_scores) # (B, T_dec, T_text)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values: (B, T_dec, hidden_dim)
        output = th.bmm(attn_weights, values)

        # Scale by sqrt(hidden_dim) instead of sequence length for correctness
        #TODO: validate that we are dividing by the appropriate factor here
        output = output / self.sqrt_hidden

        output = self.output_proj(output) # (B, T_dec, hidden_dim)
        return output

class DecoderCore(nn.Module):
    """
    Decoder core: stacked causal ConvBlocks + final attention.

    Args:
      speaker_embedding_dim (int)
      hidden_dim (int)
      dropout (float)
      n_conv_layers (int)
      kernel_size (int)

    Input shape:
      input: (B, T_dec, hidden_dim)
      enc_keys: (B, T_text, hidden_dim)
      enc_values: (B, T_text, hidden_dim)
      speaker_embedding: (B, speaker_embedding_dim)

    Output shape:
      (B, T_dec, hidden_dim)
    """
    def __init__(
        self,
        speaker_embedding_dim: int,
        hidden_dim: int,
        dropout: float,
        n_conv_layers: int,
        kernel_size: int
    ):
        super().__init__()

        self.n_conv_layers = n_conv_layers
        self.attn_final = Attention(
            speaker_embedding_dim=speaker_embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        self.convs = nn.ModuleList()
        for _ in range(n_conv_layers):
            self.convs.append(
                ConvBlock(
                    kernel_size=kernel_size,
                    hidden_dim=hidden_dim,
                    speaker_embedding_dim=speaker_embedding_dim,
                    dropout=dropout,
                    causal_padding=True
                )
            )
            

    def forward(
        self,
        input: th.Tensor,
        enc_keys: th.Tensor,
        enc_values: th.Tensor,
        speaker_embedding: th.Tensor,
        current_mel_pos: int
    ) -> th.Tensor:
        # input: (B, T_dec, hidden_dim)
        conv_out = input.transpose(1, 2)  # (B, hidden_dim, T_dec)
        for conv in self.convs:
            conv_out = conv(x=conv_out, speaker_embedding=speaker_embedding)
        conv_out = conv_out.transpose(1, 2)  # (B, T_dec, hidden_dim)

        att_out = self.attn_final(
            query=conv_out,
            keys=enc_keys,
            values=enc_values,
            speaker_embedding=speaker_embedding,
            current_mel_pos=current_mel_pos
        )

        out = np.sqrt(0.5) * (att_out + conv_out)
        return out
    
class Done(nn.Module):
    """
    Predicts the 'done' token: signals when the decoder should stop.

    Args:
      hidden_dim (int)

    Input shape:
      input: (B, T_dec, hidden_dim)

    Output shape:
      done_prob: (B, T_dec, 1) - probability of being done
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_features=hidden_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input: th.Tensor) -> th.Tensor:
        # input: (B, T_dec, hidden_dim)
        output = self.fc(input)      # (B, T_dec, 1)
        output = self.sigmoid(output) # (B, T_dec, 1), probability in [0,1]
        return output

class Decoder(nn.Module):
    """
    Decoder: autoregressively produces mel spectrogram frames from encoder outputs.
    Uses Prenet -> DecoderCore -> Output layer.

    Args:
      speaker_embedding_dim (int)
      hidden_dim (int)
      dropout (float)
      n_conv_layers (int)
      n_proj_layers (int)
      mel_dim (int)
      reduction_factor (int)
      kernel_size (int)
    """
    def __init__(
        self,
        speaker_embedding_dim: int,
        hidden_dim: int,
        dropout: float,
        n_conv_layers: int,
        n_proj_layers: int,
        mel_dim: int,
        reduction_factor: int,
        kernel_size: int
    ):
        super().__init__()

        self.n_layers = n_proj_layers
        self.reduction_factor = reduction_factor
        self.mel_dim = mel_dim

        self.done = Done(hidden_dim=hidden_dim)
        self.prenet = Prenet(
            mel_spec_dim=mel_dim,
            hidden_dim=hidden_dim,
            speaker_embedding_dim=speaker_embedding_dim,
            dropout=dropout,
            n_proj_layers=n_proj_layers
        )
        self.decoder_core = DecoderCore(
            speaker_embedding_dim=speaker_embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            n_conv_layers=n_conv_layers,
            kernel_size=kernel_size
        )

        self.mel_fc = nn.Linear(in_features=hidden_dim, out_features=mel_dim * reduction_factor)
        #self.mel_sigmoid_fc = nn.Linear(in_features=hidden_dim, out_features=mel_dim * reduction_factor)
        #self.mel_sigmoid = nn.Sigmoid()

    def forward(
        self,
        mel_specs: th.Tensor,
        encoder_keys: th.Tensor,
        encoder_values: th.Tensor,
        speaker_embedding: th.Tensor,
        current_mel_pos: int
    ):
        # mel_specs: (B, T_dec, mel_dim)
        # encoder_keys/values: (B, T_text, hidden_dim)
        # speaker_embedding: (B, speaker_embedding_dim)

        prenet_out = self.prenet(input=mel_specs, speaker_embedding=speaker_embedding) # (B, T_dec, hidden_dim)
        
        # Pass through decoder core (convs + attention)
        attn_out = self.decoder_core(
            input=prenet_out,
            enc_keys=encoder_keys,
            enc_values=encoder_values,
            speaker_embedding=speaker_embedding,
            current_mel_pos=current_mel_pos
        )  # (B, T_dec, hidden_dim)

        done_out = self.done(input=attn_out)  # (B, T_dec, 1)

        # # Gate for mel output
        # mel_out_gate = self.mel_sigmoid_fc(attn_out)   # (B, T_dec, mel_dim)
        # mel_out_gate = self.mel_sigmoid(mel_out_gate)  # (B, T_dec, mel_dim)

        # mel_out = self.mel_fc(attn_out) * mel_out_gate # (B, T_dec, mel_dim)
        mel_out = self.mel_fc(attn_out)
        B, T_dec, _ = mel_out.shape
        mel_out = mel_out.view(B, T_dec * self.reduction_factor, self.mel_dim) # reduction factor
        return (attn_out, mel_out, done_out)

class Converter(nn.Module):
    """
    Converter: refines decoder states to produce linear spectrograms (and possibly sharpen mel).

    Args:
      hidden_dim (int)
      speaker_embedding_dim (int)
      linear_spec_dim (int)
      n_conv_layers (int)
      sharpening_factor (float)
      kernel_size (int)
      dropout (float)
    """
    def __init__(
        self,
        hidden_dim: int,
        speaker_embedding_dim: int,
        linear_spec_dim: int,
        n_conv_layers: int,
        sharpening_factor: float,
        kernel_size: int,
        dropout: float
    ):
        super().__init__()

        self.sharpening_factor = sharpening_factor

        self.conv = nn.ModuleList(
            [
                ConvBlock(
                    kernel_size=kernel_size,
                    hidden_dim=hidden_dim,
                    speaker_embedding_dim=speaker_embedding_dim,
                    dropout=dropout,
                    causal_padding=False
                )
                for _ in range(n_conv_layers)
            ]
        )

        self.linear_spec_proj = nn.Linear(in_features=hidden_dim, out_features=linear_spec_dim)

    def forward(self, decoder_out: th.Tensor, speaker_embedding: th.Tensor) -> th.Tensor:
        # decoder_out: (B, T_dec_out, hidden_dim)
        x = decoder_out.transpose(1, 2)  # (B, hidden_dim, T_dec_out)
        for conv in self.conv:
            x = conv(x=x, speaker_embedding=speaker_embedding)
        x = x.transpose(1, 2)  # (B, T_dec_out, hidden_dim)

        linear_spec = self.linear_spec_proj(x)  # (B, T_dec_out, linear_spec_dim)

        # Clamp and sharpen
        linear_spec = th.clamp(linear_spec, min=1e-7)

        return linear_spec ** self.sharpening_factor

class DeepVoice3(nn.Module):
    """
    Full Deep Voice 3 model:
    Encoder: text -> keys, values
    Decoder: autoregressive mel spec / done prediction 
    Converter: refine to linear spectrogram and sharpen

    Args:
      speaker_id_map (dict): map from speaker ID strings to speaker indices
      n_symbols (int): vocabulary size
      symbol_embedding_dim (int)
      hidden_dim (int)
      speaker_embedding_dim (int)
      encoder_layers (int)
      decoder_proj_layers (int)
      decoder_conv_layers (int)
      converter_layers (int)
      reduction_factor (int)
      mel_dim (int)
      linear_dim (int)
      dropout (float)
      sharpening_factor (float)
      kernel_size (int)
    """
    def __init__(
        self,
        speaker_id_map: dict,
        n_symbols: int,
        symbol_embedding_dim: int,
        hidden_dim: int,
        speaker_embedding_dim: int,
        encoder_layers: int,
        decoder_proj_layers: int,
        decoder_conv_layers: int,
        converter_layers: int,
        reduction_factor: int,
        mel_dim: int,
        lin_dim: int,
        dropout: float,
        sharpening_factor: float,
        kernel_size: int
    ):
        super().__init__()
        self.reduction_factor = reduction_factor
        self.mel_dim = mel_dim
        self.lin_dim = lin_dim
        self.hidden_dim = hidden_dim

        self.speaker_id_map = speaker_id_map
        
        self.symbol_embedder = nn.Embedding(num_embeddings=n_symbols, embedding_dim=symbol_embedding_dim)
        self.speaker_embedder = nn.Embedding(num_embeddings=len(speaker_id_map), embedding_dim=speaker_embedding_dim)

        self.encoder = Encoder(
            symbol_embedding_dim=symbol_embedding_dim,
            hidden_dim=hidden_dim,
            speaker_embedding_dim=speaker_embedding_dim,
            dropout=dropout,
            n_conv=encoder_layers,
            kernel_size=kernel_size
        )

        self.decoder = Decoder(
            speaker_embedding_dim=speaker_embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            n_conv_layers=decoder_conv_layers,
            mel_dim=mel_dim,
            n_proj_layers=decoder_proj_layers,
            reduction_factor=reduction_factor,
            kernel_size=kernel_size
        )

        self.converter = Converter(
            hidden_dim=hidden_dim,
            speaker_embedding_dim=speaker_embedding_dim,
            linear_spec_dim=lin_dim,
            n_conv_layers=converter_layers,
            sharpening_factor=sharpening_factor,
            kernel_size=kernel_size,
            dropout=dropout
        )

    def forward(
        self,
        raw_speaker_ids: list,
        symbols: th.Tensor,
        lin_specs: th.Tensor,
        mel_specs: th.Tensor,
        T_actual: th.Tensor,
        current_mel_pos: int = 0
    ):
        # raw_speaker_ids: list of speaker IDs (strings or ints)
        # symbols: (B, T_text) symbol indices
        # lin_specs: (B, linear_dim, T_max)
        # mel_specs: (B, mel_dim, T_max)
        # T_actual: (B,) actual length for each sample
        # current_mel_pos: starting mel position for positional encoding (int)

        # Map speaker IDs to embeddings
        speaker_ids = th.tensor([self.speaker_id_map[sid] for sid in raw_speaker_ids])
        speaker_embedding = self.speaker_embedder(speaker_ids)  # (B, speaker_embedding_dim)

        symbol_embedding = self.symbol_embedder(symbols)  # (B, T_text, symbol_embedding_dim)
        keys, values = self.encoder(symbol_embedding=symbol_embedding, speaker_embedding=speaker_embedding) # (B,T_text,H),(B,T_text,H)

        B, mel_dim, T_frames = mel_specs.shape
        T_dec = T_frames // self.reduction_factor
        initial_zeros = th.zeros(B, 1, mel_dim)
        mel_specs_dec = mel_specs.transpose(1, 2)[:, ::self.reduction_factor, ]
        mel_specs_dec = th.cat([initial_zeros, mel_specs_dec[:, :-1, :]], dim=1,) # we leave off the last frame for the input. 

        attn_out, mel_out, done_out = self.decoder(
            mel_specs=mel_specs_dec,
            encoder_keys=keys,
            encoder_values=values,
            speaker_embedding=speaker_embedding,
            current_mel_pos=current_mel_pos
        ) # attn_out: (B, T_dec, H), mel_out: (B, T_dec, mel_dim), done_out: (B, T_dec, 1)

        attn_out_expanded = attn_out.unsqueeze(2).expand(B, T_dec, self.reduction_factor, self.hidden_dim).contiguous()
        attn_out_expanded = attn_out_expanded.view(B, T_dec * self.reduction_factor, self.hidden_dim)

        lin_out = self.converter(
            decoder_out=attn_out_expanded,
            speaker_embedding=speaker_embedding
        ) # (B, T_dec, linear_dim)

        return {
            'mel_out': mel_out,
            'done_out': done_out,
            'lin_out': lin_out,
        }

    def inference(self, symbols: th.tensor, speaker_id: th.tensor, max_decoder_steps = 1000):
        assert symbols.size(0) == 1, "One utterance at a time"

        with th.no_grad():
            symbol_embedding = self.symbol_embedder(symbols)
            speaker_embedding = self.speaker_embedder(speaker_id)
            
            keys, values = self.encoder(
                symbol_embedding = symbol_embedding,
                speaker_embedding = speaker_embedding
            )

            generated_mels = [] 
            generated_attns = []
            done = False
            curr_mel_pos = 0
            curr_input = th.zeros((1, self.reduction_factor, self.mel_dim))
            step = 0

            while not done and step < max_decoder_steps:
                attn_out, mel_out, done_out = model.decoder(
                    mel_specs = curr_input,
                    encoder_keys = keys,
                    encoder_values = values,
                    speaker_embedding = speaker_embedding,
                    current_mel_pos = curr_mel_pos
                )

                predicted_frames = mel_out[:, -self.reduction_factor:, :]
                generated_mels.append(predicted_frames)
                generated_attns.append(attn_out)
                curr_input = predicted_frames[:, -1:, :]
                curr_mel_pos += self.reduction_factor
                step += 1

                if done_out[0, -1, 0] > .5:
                    done = True

            #full_mel = th.cat(generated_mels, dim=1)
            full_attn_out = th.cat(generated_attns, dim=1).unsqueeze(2).expand(-1, -1, self.reduction_factor, -1)
            full_attn_out = full_attn_out.contiguous().view(1, -1, self.hidden_dim)

            lin_out = self.converter(full_attn_out, speaker_embedding)
                
            return lin_out
    

if __name__ == "__main__":
    REDUCTION_FACTOR = 4
    FFT_N = 512
    MEL_DIM = 80
    
    tts_data = get_saved_data('VCTK_4096_512', with_lin=True, transpose_specs=False, max_load = 2000, reduction_factor = REDUCTION_FACTOR)
    dataset = TTSDataset(tts_data)
    speaker_id_map = get_speaker_id_map('./data/VCTK-Corpus/speaker-info.txt')
    symbol_id_map = phoneme.symbol_id_map

    model = DeepVoice3(
        speaker_id_map=speaker_id_map,
        n_symbols=phoneme.TOTAL_SYMBOLS,
        symbol_embedding_dim=128,
        hidden_dim=128,
        speaker_embedding_dim=64,
        encoder_layers=4,
        decoder_proj_layers=1,
        decoder_conv_layers=4,
        converter_layers=4,
        reduction_factor=REDUCTION_FACTOR,
        mel_dim=80,
        lin_dim=FFT_N // 2 + 1,
        dropout=0.1,
        sharpening_factor=1.4,
        kernel_size=5
    )

    dataloader = DataLoader(
        dataset, 
        batch_size = 32,
        shuffle = True,
        collate_fn = collate_fn
    )

    optimizer = th.optim.Adam(model.parameters(), lr=1e-4)
    i_epoch = 0

    while True:
        acc_loss = 0 
        for speaker_ids, texts, T_actual, lin_specs, mel_specs in dataloader:
            """
            speaker_ids: list[str]                  the embedding mapped speaker ids
            texts: tensor   (B, max_symbol_length)  padded embedding mapped symbols
            T_actual: tensor (B)                    actual lengths of unpadded mel_specs (incorporating R)
            lin_specs: tensor (B, lin_dim, lin_T)   
            mel_specs: tensor (B, mel_dim, mel_T)
            """
            B = len(speaker_ids)

            #TODO do we have to pad mel_spec here? 
            outputs = model(
                raw_speaker_ids=speaker_ids,
                symbols=texts,
                lin_specs=lin_specs,
                mel_specs=mel_specs,
                T_actual=T_actual
            )

            mel_out = outputs['mel_out']    # (B, T_dec, mel_dim)
            done_out = outputs['done_out']  # (B, T_dec, 1), probabilities in [0,1]
            lin_out = outputs['lin_out']    # (B, T_dec, linear_dim)

            # Create masks
            max_T = mel_specs.size(-1)
            mask = (th.arange(max_T).unsqueeze(0).to(mel_specs.device) < T_actual.unsqueeze(1)) # (B, T_dec)
            mel_mask = mask.unsqueeze(2) # (B, T_dec, 1)
            lin_mask = mask.unsqueeze(2) # (B, T_dec, 1)

            # Compute losses
            # Mel L1 loss
            #TODO: Validate whether this needs to be shifted by R since we are ideally predicting the next frame. 
            mel_loss = nn.functional.l1_loss(input=mel_out, target=mel_specs.transpose(1,2), reduction='none')
            mel_loss = (mel_mask * mel_loss).sum() / mel_mask.sum()

            # Linear L1 loss
            lin_loss = nn.functional.l1_loss(input=lin_out, target=lin_specs.transpose(1,2), reduction='none')
            lin_loss = (lin_mask * lin_loss).sum() / lin_mask.sum()

            # Done loss
            # Create target: done at last valid frame
            done_target = th.zeros_like(done_out)
            for i in range(B):
                final_step = (T_actual[i] - 1) // model.reduction_factor
                done_target[i, final_step, 0] = 1.0

            done_loss = nn.functional.binary_cross_entropy(
                input=done_out,
                target=done_target,
                reduction='none'
            )

            total_loss = (mel_loss + lin_loss + done_loss).mean()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            acc_loss += total_loss 

        i_epoch += 1
        if not i_epoch % 100:
            lin_out = model.inference(
                th.tensor(get_symbol_sequence(["I WENT TO THE STORE TO BUY SOME BREAD."])),
                th.tensor([0]), 
            ).squeeze(0)

            reconstruct_audio(
                lin_out, f'output/',
                hop_length= 64,
                win_length = 128,
                fft_n = FFT_N
            )

        print(f"EPOCH {i_epoch:10}\t{acc_loss:.4f}")