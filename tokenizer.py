from typing import List, Dict, Optional
from phoneme import phoneme_single_letter_map, phoneme_symbols

class TTSTokenizer:
    def __init__(self):
        # Create vocabulary from phoneme symbols
        self.phoneme_to_id = {p: i for i, p in enumerate(sorted(phoneme_symbols))}

        # Add special tokens
        self.pad_token_id = len(self.phoneme_to_id)
        self.eos_token_id = len(self.phoneme_to_id) + 1
        self.phoneme_to_id['<PAD>'] = self.pad_token_id
        self.phoneme_to_id['<EOS>'] = self.eos_token_id

        self.id_to_phoneme = {v: k for k, v in self.phoneme_to_id.items()}

        self.letter_map = phoneme_single_letter_map

    def convert_phonemes_to_ids(self, phonemes: List[List[str]]) -> List[int]:
        ids = []
        for word_phonemes in phonemes:
            for p in word_phonemes:
                if p in self.phoneme_to_id:
                    ids.append(self.phoneme_to_id[p])
                else:
                    # Handle unknown phonemes
                    ids.append(self.phoneme_to_id['<UNK>'])
        return ids + [self.eos_token_id]  # Add EOS token

    def pad_sequence(self, sequence: List[int], max_length: int) -> List[int]:
        return sequence + [self.pad_token_id] * (max_length - len(sequence))

class SpeakerEncoder:
    def __init__(self):
        self.speaker_to_id: Dict[str, int] = {}
        self.id_to_speaker: Dict[int, str] = {}

    def add_speaker(self, speaker_id: str) -> None:
        if speaker_id not in self.speaker_to_id:
            idx = len(self.speaker_to_id)
            self.speaker_to_id[speaker_id] = idx
            self.id_to_speaker[idx] = speaker_id

    def encode(self, speaker_id: str) -> int:
        return self.speaker_to_id[speaker_id]

    @property
    def num_speakers(self) -> int:
        return len(self.speaker_to_id)