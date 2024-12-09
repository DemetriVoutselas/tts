import os
import torch
from tokenizer import TTSTokenizer, SpeakerEncoder
from data_loader import create_dataloader
from deepvoice import DeepVoice


def test_pipeline(processed_dir: str):
    print("\n=== Testing TTS Pipeline ===\n")

    print("1. Initializing Components...")
    tokenizer = TTSTokenizer()
    speaker_encoder = SpeakerEncoder()

    print("\n2. Creating DataLoader...")
    loader = create_dataloader(
        processed_dir,
        tokenizer,
        speaker_encoder,
        batch_size=2,  # Small batch for testing
        num_workers=0
    )

    print(f"Found {speaker_encoder.num_speakers} unique speakers")

    print("\n3. Examining a Single Batch...")
    batch = next(iter(loader))

    print("\nBatch contents:")
    print(f"- Speaker IDs shape: {batch['speaker_ids'].shape}")
    print(f"- Phonemes shape: {batch['phonemes'].shape}")
    print(f"- Attention mask shape: {batch['attention_mask'].shape}")
    print(f"- Linear specs shape: {batch['linear_specs'].shape}")
    print(f"- Mel specs shape: {batch['mel_specs'].shape}")

    print("\n4. Creating Model...")
    model = DeepVoice(
        n_speakers=speaker_encoder.num_speakers,
        hidden_dim=512,
        speaker_embedding_dim=64,
        mel_frame_batch_size=80
    )

    print("\n5. Testing Model Forward Pass...")
    try:
        # Move both model and batch to same device if using GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        with torch.no_grad():
            output = model(batch)

        print("Forward pass successful!")
        print("\nModel outputs:")
        for k, v in output.items():
            if isinstance(v, torch.Tensor):
                print(f"- {k}: {v.shape}")

    except Exception as e:
        print(f"\nModel forward pass failed:")
        print(f"Error: {str(e)}")

        # Print model's expected input format
        print("\nModel expects input batch with:")
        print("- speaker_ids: (batch_size,)")
        print("- phonemes: (batch_size, seq_len)")
        print("- linear_specs: (batch_size, n_freq, time)")
        print("- mel_specs: (batch_size, n_mels, time)")

    print("\n=== Pipeline Test Complete ===")


if __name__ == "__main__":
    PROCESSED_DIR = "D:/CS Classes at GT/CS7643 - DL/Final Project/tts/data/processed/VCTK"
    test_pipeline(PROCESSED_DIR)