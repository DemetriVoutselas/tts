def load_embeddings():

    embedding_path = '/content/drive/MyDrive/Multi-speaker_DV3/deepvoice3_pytorch/speaker_embeddings.pth'
    embeddings = torch.load(embedding_path)
    return embeddings.weight.detach().numpy()


def create_speaker_map():

    embeddings = load_embeddings()
    speaker_info = {}

    # Load speaker info
    with open(VCTK_SPEAKER_INFO_PATH, 'r') as f:
        next(f)  # Skip header
        for idx, line in enumerate(f):
            if idx < len(embeddings):
                parts = line.strip().split()
                speaker_id = parts[0]
                speaker_info[speaker_id] = {
                    'embedding': embeddings[idx],
                    'info': {
                        'age': parts[1],
                        'gender': parts[2],
                        'accent': parts[3] if len(parts) > 3 else None
                    }
                }
    return speaker_info


# Test the mapping
mapping = create_speaker_map()
print(f"Total speakers mapped: {len(mapping)}")
for speaker_id in list(mapping.keys())[:3]:  
    print(f"\nSpeaker {speaker_id}:")
    print(f"Info: {mapping[speaker_id]['info']}")
    print(f"Embedding shape: {mapping[speaker_id]['embedding'].shape}")