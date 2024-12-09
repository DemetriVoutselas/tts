import os
import json
import torch
from typing import List, Dict
from torch.utils.data import Dataset, DataLoader
import logging
from tokenizer import TTSTokenizer, SpeakerEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VCTKDataset(Dataset):
    def __init__(self, processed_dir: str):
        """
        Args:
            processed_dir: Path to the processed VCTK data directory
        """
        self.processed_dir = processed_dir
        self.utterances = []
        self.speaker_to_utterances = {}

        try:
            self._load_dataset()
        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            raise

    def _load_dataset(self):
        if not os.path.exists(self.processed_dir):
            raise FileNotFoundError(f"Processed data directory not found at {self.processed_dir}")

        # Iterate through utterance directories
        for utterance_id in os.listdir(self.processed_dir):
            utterance_path = os.path.join(self.processed_dir, utterance_id)

            try:
                metadata = self._read_metadata(utterance_path)
                self.utterances.append({
                    "utterance_id": utterance_id,
                    "metadata": metadata
                })

                # Group by speaker
                speaker_id = metadata['speaker_id']
                if speaker_id not in self.speaker_to_utterances:
                    self.speaker_to_utterances[speaker_id] = []
                self.speaker_to_utterances[speaker_id].append(utterance_id)

            except Exception as e:
                logger.warning(f"Skipping {utterance_id}: {str(e)}")
                continue

        logger.info(f"Loaded {len(self.utterances)} utterances from {len(self.speaker_to_utterances)} speakers")

    def _read_metadata(self, utterance_path: str) -> Dict:
        metadata_path = os.path.join(utterance_path, 'data.dat')
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in {metadata_path}")

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt_data = self.utterances[idx]
        utterance_path = os.path.join(self.processed_dir, utt_data['utterance_id'])

        try:
            linear_spec = self._load_tensor(utterance_path, 'linear_spec')
            mel_spec = self._load_tensor(utterance_path, 'mel_spec')

            if linear_spec is None or mel_spec is None:
                raise ValueError("Missing spectrogram data")

            return {
                'metadata': utt_data['metadata'],
                'linear_spec': linear_spec,
                'mel_spec': mel_spec
            }

        except Exception as e:
            logger.error(f"Error loading utterance {utt_data['utterance_id']}: {str(e)}")
            raise

    def _load_tensor(self, path: str, name: str) -> torch.Tensor:
        tensor_path = os.path.join(path, name)
        if not os.path.exists(tensor_path):
            raise FileNotFoundError(f"Missing {name} at {tensor_path}")
        return torch.load(tensor_path)


def create_dataloader(
        processed_dir: str,
        tokenizer: TTSTokenizer,
        speaker_encoder: SpeakerEncoder,
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle: bool = True
) -> DataLoader:
    """Creates a DataLoader for the VCTK dataset with proper batching"""

    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Custom collate function to create properly structured batches"""
        # Extract data
        linear_specs = [item['linear_spec'] for item in batch]
        mel_specs = [item['mel_spec'] for item in batch]

        # Process phonemes and create attention masks
        phoneme_sequences = []
        max_phoneme_len = 0
        for item in batch:
            phonemes = tokenizer.convert_phonemes_to_ids(item['metadata']['phoneme'])
            phoneme_sequences.append(phonemes)
            max_phoneme_len = max(max_phoneme_len, len(phonemes))

        # Pad phoneme sequences
        padded_phonemes = torch.tensor([
            tokenizer.pad_sequence(seq, max_phoneme_len)
            for seq in phoneme_sequences
        ])

        # Create attention mask (1 for tokens, 0 for padding)
        attention_mask = (padded_phonemes != tokenizer.pad_token_id).float()

        # Get max spectrogram length
        max_spec_len = max(spec.shape[1] for spec in linear_specs)

        # Pad spectrograms
        linear_specs_padded = torch.stack([
            torch.nn.functional.pad(spec, (0, max_spec_len - spec.shape[1]))
            for spec in linear_specs
        ])

        mel_specs_padded = torch.stack([
            torch.nn.functional.pad(spec, (0, max_spec_len - spec.shape[1]))
            for spec in mel_specs
        ])

        speaker_ids = torch.tensor([
            speaker_encoder.encode(item['metadata']['speaker_id'])
            for item in batch
        ])

        return {
            'speaker_ids': speaker_ids,
            'phonemes': padded_phonemes,
            'attention_mask': attention_mask,
            'linear_specs': linear_specs_padded,
            'mel_specs': mel_specs_padded
        }

    dataset = VCTKDataset(processed_dir)

    # Build speaker encoder
    for item in dataset:
        speaker_encoder.add_speaker(item['metadata']['speaker_id'])

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers
    )