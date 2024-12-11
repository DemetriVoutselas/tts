"""
Runs preprocessing on the wav/text pairs from the source dataset

As of now, this just saves the preprocessed files to the hard drive. 
"""

from dataclasses import asdict, dataclass
import json
import os
import re
from typing import Optional
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch as th
from tqdm import tqdm

from phoneme import get_phoneme
import phoneme

# # PAPER VALUES FOR VCTK
# SR = 48000
# FFT_N = 4096
# FFT_WINDOW = 2400
# FFT_HOP = 600

SR = 4096
FFT_N = 512
FFT_WINDOW = 128
FFT_HOP = 64

R = 4
TOP_DB = 25

VCTK_ROOT_DIR = 'data/VCTK-Corpus'
VCTK_AUDIO_DIR = f'{VCTK_ROOT_DIR}/wav48'
VCTK_TXT_DIR = f'{VCTK_ROOT_DIR}/txt'
VCTK_SPEAKER_INFO_PATH = f"{VCTK_ROOT_DIR}/speaker-info.txt"

PROCESSED_SAVE_DIR = 'processed'

MEL_BANDS = 80
MEL_BASIS = librosa.filters.mel(sr = SR, n_fft= FFT_N, n_mels= MEL_BANDS)

UNKNOWN_PHONEME = "<UNK>"

def normalize_text(text: str):
    text = re.sub(r'[^\w\s.?!]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = text.upper()

    return text

def trim_silence(audio, top_db = TOP_DB):
    audio, _ = librosa.effects.trim(audio, top_db)
    return audio

def get_text(path):
    with open(path, 'r') as fp:
        text= fp.read().replace('\n', '')
    
    return normalize_text(text)

def get_symbol_sequence(text_or_phoneme) -> list[str]:
    symbol_seqs = []
    for i, it in enumerate(text_or_phoneme):
        symbol_seq = []

        if isinstance(it, str):
            for let in it:
                symbol_seq.append(phoneme.symbol_id_map[let])

        else:
            for ph_word in it:
                for ph in ph_word:
                    symbol_seq.append(phoneme.symbol_id_map[ph])
                symbol_seq.append(phoneme.symbol_id_map[' '])

        symbol_seqs.append(symbol_seq)

    return symbol_seqs

def get_audio(path, sr = SR):
    audio, sr = librosa.load(path, sr=sr)
    return audio

def get_linear_spec(audio, n_fft = FFT_N, hop_length = FFT_HOP, window_length = FFT_WINDOW) -> th.Tensor:
    linear_stft = librosa.stft(audio, n_fft= n_fft, hop_length= hop_length, win_length= window_length)
    linear_spec = np.abs(linear_stft)
    return linear_spec

def get_mel_spec(linear_spec, mel_basis = MEL_BASIS) -> th.Tensor:
    return np.dot(mel_basis, linear_spec)

def reconstruct_audio(linear_spec, save_path, n_iter = 1000, sr = SR, hop_length = FFT_HOP, win_length = FFT_WINDOW, fft_n = FFT_N, algo = 'griffin'):
    if algo == 'griffin':
        audio = librosa.griffinlim(linear_spec.T.numpy(), n_iter=n_iter, hop_length=hop_length, win_length=win_length, n_fft = FFT_N)
    else:
        raise NotImplementedError()
    
    sf.write(f'{save_path}/reconstructed.wav', audio, samplerate=sr)


def pad_mel_spec(mel_spec, target_len, r = R):
    assert not target_len % R, "target_len must be a multiple of R"

    curr_len = mel_spec.shape[1]
    if curr_len < target_len:
        np.pad(mel_spec, ((0,0), (0, target_len - curr_len)))

def log_spec(spec):
    return np.log(spec + 1E-10)

def get_vctk_audio(speaker_info_path = VCTK_SPEAKER_INFO_PATH, audio_path = VCTK_AUDIO_DIR, txt_path = VCTK_TXT_DIR):
    speakers = []
    with open(speaker_info_path, 'r') as fp:
        for line in fp.readlines()[1:]:
            line_speaker = line.split()
            line_speaker = line_speaker[:3] + [' '.join(line_speaker[3:])]

            speakers.append(line_speaker)

    tts_data_items = []    
    for speaker_id, *_ in tqdm(speakers):        
        speaker_dir = f'{speaker_id}'
        speaker_txt_dir = f'{txt_path}/{speaker_dir}'
        speaker_audio_dir = f'{audio_path}/{speaker_dir}'

        if not os.path.exists(speaker_txt_dir) or not os.path.exists(speaker_audio_dir):
            print(f'WARNING: path not found: {speaker_dir}')
            continue

        speaker_txt_files = os.listdir(speaker_txt_dir)        
        speaker_audio_files = [f'{fn[:-4]}_mic2.flac'  for fn in speaker_txt_files]
        
        for txt_file_path, audio_file_path in zip(speaker_txt_files, speaker_audio_files):
            utterance_id = txt_file_path[:-4]
            txt_file_path = f'{speaker_txt_dir}/{txt_file_path}'
            audio_file_path = f'{speaker_audio_dir}/{audio_file_path}'

            if not os.path.exists(txt_file_path) or not os.path.exists(audio_file_path):
                print(f"WARNING: failed to find txt/audio file for {speaker_id}/{utterance_id} ")
                continue

            tts_data_item = TTSDataItem.build(speaker_id = speaker_id, utterance_id = utterance_id, text_file=txt_file_path, audio_file=audio_file_path)
            #tts_data_item.plot_spec()
            #tts_data_items.append(tts_data_item)
            tts_data_item.save()


    return tts_data_items

def process_custom_audio(dir_path):
    for i, file in enumerate(os.listdir(dir_path)):
        file_path = os.path.join(dir_path, file)

        audio = get_audio(file_path)        
        linear_spec = get_linear_spec(audio)
        mel_spec = get_mel_spec(linear_spec)

        tts_item = TTSDataItem(
            speaker_id = 'demetri', 
            utterance_id = str(i), 
            type = 'custom',
            text = '',
            phoneme = None,
            lin_spec= th.from_numpy(linear_spec),
            mel_spec= th.from_numpy(mel_spec), 
            T = mel_spec.shape[1]
        )       

        tts_item.save() 


@dataclass(frozen=True)
class TTSDataItem:
    speaker_id: str					# the raw id of the speaker as defined in the data set
    utterance_id: str				# the id of the utterance
    type: str						# what dataset it came from
    text: str						# the actual text being spoken
    phoneme: Optional[list[str]]	# the equivalent phonemes 
    T: int							# the total time of the spectograms, preserved here due to padding
    #audio: np.ndarray

    lin_spec: th.Tensor			# a tensor of the lin spectrogram
    mel_spec: th.Tensor				# a tensor of the mel spectrogram

    @staticmethod
    def build(speaker_id: str, utterance_id: str, text_file: str, audio_file:str, type: str = 'VCTK' ) -> 'TTSDataItem':
        text = get_text(text_file)
        phoneme = get_phoneme(text)
        
        audio = get_audio(audio_file)        
        linear_spec = get_linear_spec(audio)
        mel_spec = get_mel_spec(linear_spec)

        return TTSDataItem(
            speaker_id= speaker_id,
            utterance_id = utterance_id,
            text = text,
            phoneme= phoneme,
            #audio = audio,
            lin_spec= th.from_numpy(linear_spec),
            mel_spec= th.from_numpy(mel_spec), 
            type = type,
            T = mel_spec.shape[1]
        )
    
    def plot_spec(self, sr = SR, hop_length = FFT_HOP):
        fig, ax = plt.subplots(1, 2, figsize=(20, 6))
        ax[0].set_title("Linear Spec")
        ax[1].set_title("Mel Spec")

        librosa.display.specshow(self.lin_spec, sr=sr, hop_length=hop_length, x_axis="time", y_axis="log", cmap="magma", ax=ax[0])        
        librosa.display.specshow(self.mel_spec, sr=sr, hop_length=hop_length, x_axis="time", y_axis="mel", cmap="magma", ax=ax[1])

        plt.show()

    def save(self, processed_path = PROCESSED_SAVE_DIR, reconstruct_audio_flag = False):
        save_path = f"{processed_path}/{self.type}_{SR}_{FFT_N}/{self.utterance_id}"
        os.makedirs(save_path, exist_ok = True)
        with open(f"{save_path}/data.dat", "w") as fp:
            fp.write(
                json.dumps(
                    dict(
                        speaker_id = self.speaker_id,
                        utterance_id = self.utterance_id,
                        type = self.type,
                        text = self.text,
                        phoneme = self.phoneme,
                        T = self.T
                    )
                )
            )
        th.save(self.lin_spec, f"{save_path}/linear_spec",)
        th.save(self.mel_spec, f"{save_path}/mel_spec", )
        if reconstruct_audio_flag:
            reconstruct_audio(linear_spec = self.lin_spec.numpy(), save_path= save_path)
    

def get_saved_data(set_path, device = th.device('cpu'), with_lin = False, transpose_specs = True, max_load = None, reduction_factor = 1):
    base_path = os.path.join("processed", set_path)
    tts_data = []

    for utt_folder in tqdm(os.listdir(base_path)[:max_load], "Loading TTS Data"):
        utt_path = os.path.join(base_path, utt_folder)
        data_path = os.path.join(utt_path, 'data.dat')
        lin_spec_path = os.path.join(utt_path, 'linear_spec')
        mel_spec_path = os.path.join(utt_path, 'mel_spec')

        with open(data_path, 'r') as fp:
            data = json.load(fp)

        def pad_spec(tensor):
            mod = tensor.size(-1) % reduction_factor
            if mod:
                return th.nn.functional.pad(tensor, (0, reduction_factor - mod), mode='constant', value=0.0)
            else:
                return tensor

        mel_spec = pad_spec(th.load(mel_spec_path, weights_only=True)).to(device)
        data['mel_spec'] = mel_spec

        if with_lin:
            lin_spec = pad_spec(th.load(lin_spec_path, weights_only=True)).to(device)
            data['lin_spec'] = lin_spec

        if transpose_specs:
            if data['mel_spec']: data['mel_spec'] = data['mel_spec'].T
            if data['lin_spec']: data['lin_spec'] = data['lin_spec'].T

        data['T'] = data['mel_spec'].size(-1) # to account for reduction factor

        tts_data.append(
            TTSDataItem(**data)
        )

    return tts_data

def pad_tts_data(data: list[TTSDataItem]):
    for it in data:
        yield it.copy()


def get_speaker_id_map(file_path = './data/VCTK-Corpus/speaker-info-old.txt'):
    speaker_id_map = {}
    with open(file_path, 'r') as fp:
        next(fp)
        for row in fp:
            tokens = row.split()
            id = tokens[0]
            if not id.startswith('p'):
                id = 'p' + id #add leading p to align it with newer version
            speaker_id_map[id] = len(speaker_id_map)

    return speaker_id_map

if __name__ == '__main__':
    get_vctk_audio()
    #process_custom_audio('DemetriVoice')