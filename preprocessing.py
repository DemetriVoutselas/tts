from dataclasses import dataclass
import os
import re
from typing import Optional
import librosa
import librosa.display
import matplotlib.pyplot as plt
import nltk
import numpy as np
import soundfile as sf
from tqdm import tqdm

#download phoneme dict
nltk.download('cmudict')
PHONEME_DICT = nltk.corpus.cmudict.dict()

SR = 48000
R = 4

TOP_DB = 25

FFT_N = 4096
FFT_WINDOW = 2400
FFT_HOP = 600

VCTK_ROOT_DIR = 'data/VCTK-Corpus'
VCTK_AUDIO_DIR = f'{VCTK_ROOT_DIR}/wav48/'
VCTK_TXT_DIR = f'{VCTK_ROOT_DIR}/txt'
VCTK_SPEAKER_INFO_PATH = f"{VCTK_ROOT_DIR}/speaker-info.txt"

MEL_BANDS = 80
MEL_BASIS = librosa.filters.mel(sr = SR, n_fft= FFT_N, n_mels= MEL_BANDS)

UNKNOWN_PHONEME = "<UNK>"

def get_phoneme(words: str, phoneme_dict = PHONEME_DICT):
    words = re.sub(r'[^\w\s]', '', words)
    words = re.sub(r'/s+', '', words)
    words = words.strip()

    def word_or_unknown(word):
        r = phoneme_dict.get(word.lower())
        if r:
            return r[0]
        else:
            #print(f"phoneme not found: {word}")
            return UNKNOWN_PHONEME
    
    return [word_or_unknown(word) for word in words.split() ]

def trim_silence(audio, top_db = TOP_DB):
    audio, _ = librosa.effects.trim(audio, top_db)
    return audio

def get_audio(path, sr = SR):
    audio, sr = librosa.load(path, sr=sr)
    return audio

def get_text(path):
    with open(path, 'r') as fp:
        return fp.read().replace('\n', '')

def get_linear_spec(audio, n_fft = FFT_N, hop_length = FFT_HOP, window_length = FFT_WINDOW):
    linear_stft = librosa.stft(audio, n_fft= n_fft, hop_length= hop_length, win_length= window_length)
    linear_spec = np.abs(linear_stft)
    return linear_spec

def get_mel_spec(linear_spec, mel_basis = MEL_BASIS):
    return np.dot(mel_basis, linear_spec)

def reconstruct_audio(linear_spec, save_path, n_iter = 1000, sr = SR, hop_length = FFT_HOP, win_length = FFT_WINDOW, algo = 'griffin'):
    if algo == 'griffin':
        audio = librosa.griffinlim(linear_spec, n_iter=n_iter, hop_length=hop_length, win_length=win_length)
    else:
        raise NotImplementedError()
    
    sf.write('reconstructed.wav', audio, samplerate=sr)


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
        speaker_dir = f'p{speaker_id}'
        speaker_txt_files = os.listdir(f'{txt_path}/{speaker_dir}')        
        speaker_audio_files = [f'{fn[:-4]}.wav'  for fn in speaker_txt_files]
        
        for txt_file, audio_file in zip(speaker_txt_files, speaker_audio_files):
            utterance_id = txt_file[:-4]
            txt_file = f'{txt_path}/{speaker_dir}/{txt_file}'
            audio_file = f'{audio_path}/{speaker_dir}/{audio_file}'
            tts_data_item = TTSDataItem.build(speaker_id = speaker_id, utterance_id = utterance_id, text_file=txt_file, audio_file=audio_file)
            #tts_data_item.plot_spec()
            tts_data_items.append(tts_data_item)

    return tts_data_items

@dataclass(frozen=True)
class TTSDataItem:
    speaker_id: str
    utterance_id: str
    type: str
    text: str
    phoneme: Optional[str]
    #audio: np.ndarray

    linear_spec: np.ndarray
    mel_spec: np.ndarray

    @staticmethod
    def build(speaker_id: str, utterance_id: str, text_file: str, audio_file:str, type: str = 'VCTK' ) -> 'TTSDataItem':
        text = get_text(text_file)
        audio = get_audio(audio_file)
         
        phoneme = get_phoneme(text)
        linear_spec = get_linear_spec(audio)
        mel_spec = get_mel_spec(linear_spec)

        return TTSDataItem(
            speaker_id= speaker_id,
            utterance_id = utterance_id,
            text = text,
            phoneme= phoneme,
            #audio = audio,
            linear_spec= linear_spec,
            mel_spec= mel_spec, 
            type = type
        )
    
    def plot_spec(self, sr = SR, hop_length = FFT_HOP):
        fig, ax = plt.subplots(1, 2, figsize=(20, 6))
        ax[0].set_title("Linear Spec")
        ax[1].set_title("Mel Spec")

        librosa.display.specshow(self.linear_spec, sr=sr, hop_length=hop_length, x_axis="time", y_axis="log", cmap="magma", ax=ax[0])        
        librosa.display.specshow(self.mel_spec, sr=sr, hop_length=hop_length, x_axis="time", y_axis="mel", cmap="magma", ax=ax[1])

        plt.show()

get_vctk_audio()






