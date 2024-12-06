
from dataclasses import dataclass
from converter import Converter
from decoder import Decoder
from encoder import Encoder
import torch as th
import torch.nn as nn

from preprocessing import TTSDataItem

@dataclass
class HParams:
    speaker_embedding_dim: int
    hidden_dim: int
    n_vocab: int
    n_speakers: int
    dropout: float

class DeepVoice(nn.Module):
    def __init__(
        self,
        n_speakers,
        hidden_dim,
        speaker_embedding_dim,
        mel_frame_batch_size,
    ):
        super(DeepVoice, self).__init__()  

        self.mel_frame_batch_size = mel_frame_batch_size
        self.speaker_embedding = nn.Embedding(num_embeddings=n_speakers, embedding_dim=speaker_embedding_dim)  
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.converter = Converter()

    def forward(self, batch):
        speaker_ids: th.Tensor = batch['speaker_id']
        symbols: th.Tensor = batch['symbols']
        linear_spec: th.Tensor = batch['linear_spec']
        mel_spec: th.Tensor = batch['mel_spec']

        speaker_embeddings = self.speaker_embedding(speaker_ids)
        enc_keys, enc_values = self.encoder(symbols, speaker_embeddings)

        batched_mel_spec = mel_spec.view(-1, mel_spec.shape(1) // self.mel_frame_batch_size, self.mel_frame_batch_size)
        
        acc_attn = []
        acc_mel_out = []
        acc_done_out = []

        mel_frames = th.zeros_like(mel_spec[:, 0, :])
        for t in range(batched_mel_spec.shape(1)):
            attn_out, mel_frames, done_out = self.decoder(mel_frames, enc_keys, enc_values, speaker_embeddings)

            acc_attn.append(attn_out)
            acc_mel_out.append(mel_frames)
            acc_done_out.append(done_out)

        acc_attn = th.stack(acc_attn, dim=1)

        lin_spec_pred = self.converter(acc_attn)






