import os
import torch as th
import json
from test_encoder import SpeakerEncoder
from preprocessing import TTSDataItem

device = th.device(["cpu","cuda"][th.cuda.is_available()])

def get_custom_data(base_path, device):
    os.makedirs(base_path, exist_ok=True)
    batch = []

    for utt_folder in os.listdir(base_path):
        utt_path = os.path.join(base_path, utt_folder)
        data_path = os.path.join(utt_path, 'data.dat')
        lin_spec_path = os.path.join(utt_path, 'linear_spec')
        mel_spec_path = os.path.join(utt_path, 'mel_spec')

        with open(data_path, 'r') as fp:
            data = json.load(fp)
        
        lin_spec = th.load(lin_spec_path, weights_only=True).to(device)
        mel_spec = th.load(mel_spec_path, weights_only=True).to(device)
        
        data['mel_spec'] = mel_spec.T
        data['lin_spec'] = lin_spec

        batch.append(
            TTSDataItem(**data)
        )

    #batch = th.concat([it.mel_spec for it in batch], de23vice = device)
    padded_batch = th.zeros((1, len(batch), max(it.mel_spec.size(0) for it in batch), 80)).to(device)
    T_actual = th.zeros((len(batch))).to(device)

    for i, it in enumerate(batch):
        padded_batch[0, i, :it.T, :] = it.mel_spec
        T_actual[i] = it.T

    return padded_batch, T_actual


batch, T_actual  = get_custom_data('processed/custom_16384_4096', device = device)
model = th.load('output/savepoints/speaker_encoder_mode_savepoint.pth', map_location=device).to(device)
model.device = device
model.eval()

EMBEDDING_OUT_DIR = 'output/embeddings/'
os.makedirs(EMBEDDING_OUT_DIR, exist_ok=True)
with th.no_grad():
    output = model(batch, T_actual)
    th.save(output, os.path.join(EMBEDDING_OUT_DIR, 'Demetrius_large_3_2800__45.pth'))

