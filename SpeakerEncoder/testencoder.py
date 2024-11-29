import torch
from SpeakerEncoder import SpeakerEncoder

batch_size = 16
num_samples = 8
frequency_channels = 128
time_steps = 512

input_data = torch.randn(batch_size, num_samples, time_steps, frequency_channels)

model = SpeakerEncoder()
embeddings = model(input_data)

print(embeddings.shape)
