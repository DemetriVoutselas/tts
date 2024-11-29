import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from  SpeakerEncoderBlocks import Prenet, TemporalProcessing, CloningSamplesAttention


class SpeakerEncoder(nn.Module):
    """
    Input into SpeakerEncoder forward pass is mel-spectrogram. Dimensitons: (Batch, N smaples, time, mel frequency channels)
    """

    def __init__(self,
                num_prenet_layers=4,
                prenet_input_dim=128, # This is the number of mel-frequency channels in mel-spectrogram
                prenet_hidden_dim=128, # This can be whatever
                prenet_output_dim=64, #Can also be whatever, this is also the input dimention to temporal processing block
                conv_kernel_size=3, #must be odd
                dropout=0.1,
                attention_dim=128, #can be whatever
                attention_heads=8,
                embedding_dim=64 #This should be size of actual speaker embeddings
                ):
        super().__init__()

        self.prenet = Prenet(num_layers=num_prenet_layers,
                input_dim=prenet_input_dim,
                hidden_dim=prenet_hidden_dim,
                output_dim=prenet_output_dim)

        #Output of temporal processing is same shape as its input
        self.temporal_processing = TemporalProcessing(num_layers=4,
                                                input_dim=prenet_output_dim,
                                                hidden_dim=2*prenet_output_dim, #Must be twice the size of input
                                                kernel_size=conv_kernel_size, 
                                                dropout=dropout) #dropout layer is not shown in architecture but might help model prevent overfitting with little data

        
        self.cloning_samples_attention = CloningSamplesAttention(input_size=prenet_output_dim,
                                                            attention_size=attention_dim, #can be whatever
                                                            num_heads=attention_heads, 
                                                            embed_dim=embedding_dim)

    def forward(self, x):
        
        prenet_x = self.prenet(x)
        temporal_x = self.temporal_processing(prenet_x)
        embeddings = self.cloning_samples_attention(temporal_x)

        return embeddings
