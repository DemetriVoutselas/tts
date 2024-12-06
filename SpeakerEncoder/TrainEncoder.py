import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from SpeakerEncoder import SpeakerEncoder
from preprocessing_file import get_vctk_audio, TTSDataItem


# Test dataset class
class DummyMelSpectrogramDataset(Dataset):
    def __init__(self, num_samples=108, time=100, mel_channels=80):
        self.data = torch.randn(num_samples, 1, time, mel_channels)  # Random mel-spectrograms
        self.labels = torch.randn(num_samples, 16)  # Random embeddings

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# dataset class for preprocessed data
class MelSpectrogramDataset(Dataset):
    def __init__(self, tts_data_items, embedding_path):
        self.data_items = tts_data_items
        self.embeddings = torch.load(embedding_path)  # Load embeddings from .pth file

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, speaker_id):
        #Not sure how the indexes
        mel_spec = self.data_items[speaker_id].mel_spec #input to model is mel_spectrogram
        mel_spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        embedding = self.embeddings[speaker_id - 225] #Bc speaker ids start at 225 need to offset
        return mel_spec, embedding



def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.4f}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SpeakerEncoder().to(device)

#Dummy data
dummy_dataset = DummyMelSpectrogramDataset()
dummy_dataloader = DataLoader(dummy_dataset, batch_size=32, shuffle=True)

#Real data
# Haven't tried this not sure if it will work
# tts_data_items = get_vctk_audio()
# embedding_path = "../speaker_embeddings.pth"  # Path to the .pth file containing embeddings
# dataset = MelSpectrogramDataset(tts_data_items, embedding_path)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

#optimizer
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.0006)

# Train the model
train_model(model, dummy_dataloader, criterion, optimizer, num_epochs=10)
#train_model(model, dataloader, criterion, optimizer, num_epochs=10)




