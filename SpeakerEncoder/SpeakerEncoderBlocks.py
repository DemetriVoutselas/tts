import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Prenet(nn.Module):

  def __init__(self,
               num_layers,
               input_dim,
               hidden_dim,
               output_dim,
               *args,
               **kwargs):
    super().__init__(*args, **kwargs)

    self.layer_in = nn.Linear(input_dim, hidden_dim)
    self.layer_out = nn.Linear(hidden_dim, output_dim)
    self.elu = nn.ELU()

    self.mid_layers = nn.ModuleList([
        nn.Linear(hidden_dim, hidden_dim) for i in range(num_layers)
        ])


  def forward(self, x):
    x = self.elu(self.layer_in(x))
    for layer in self.mid_layers:
      x = self.elu(layer(x))

    x = self.layer_out(x)
    return x

class TemporalProcessing(nn.Module):
  def __init__(self,
               num_layers,
               input_dim,
               hidden_dim,
               kernel_size,
               dropout,
               mask_prob=0.0,
               *args,
               **kwargs):
    super().__init__(*args, **kwargs)


    self.conv_layers = nn.ModuleList([
        nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=kernel_size // 2) for _ in range(num_layers)
        ])
    
    self.dropout = nn.Dropout(dropout)
    self.glu = nn.GLU(dim=1)
    self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
    self.mask_prob = mask_prob

  def forward(self, x):
    """
    Input
    ----------
    x: output of prenet; shape (B, N, T, F)

    Output
    ----------
    y: output of temporal processing; shape (B, N, T, F)
    """
    batch_size, num_samples, timeframes, input_dim = x.size()

    # Reshape for 1D convolution: (B * N, F, T)
    x = x.view(batch_size * num_samples, timeframes, input_dim).transpose(1, 2)

    residual = x
    for conv in self.conv_layers:
        conv_out = self.dropout(conv(x)) 
        gated = self.glu(conv_out)        
        x = gated + residual             
        residual = x * torch.sqrt(torch.tensor(0.5))                     

    mask = (torch.rand(x.size(0), 1, x.size(2)) > self.mask_prob).float()
    x = x * mask

    x = self.global_avg_pool(x)

    x = x.squeeze(-1).view(batch_size, num_samples, -1) 
    return x


class CloningSamplesAttention(nn.Module):

  def __init__(self,
               input_size,
               attention_size,
               num_heads,
               embed_dim,
               *args,
               **kwargs):
    super().__init__(*args, **kwargs)

    self.keys = nn.Linear(input_size, attention_size)
    self.queries = nn.Linear(input_size, attention_size)
    self.values = nn.Linear(input_size, attention_size)

    self.elu = nn.ELU()

    self.attention = nn.MultiheadAttention(attention_size, num_heads, batch_first=True)
    self.post_attention_block = nn.Sequential(
        nn.Linear(attention_size, 1),
        nn.Softsign()
    )

    self.fc_embedding = nn.Linear(input_size, embed_dim)

  def forward(self, x):

    keys = self.elu(self.keys(x))
    queries = self.elu(self.queries(x))
    values = self.elu(self.values(x))

    attention_layer, _ = self.attention(queries, keys, values)
    post_out = self.post_attention_block(attention_layer)
    pre_embed = self.fc_embedding(x)
    return torch.sum(pre_embed * post_out, dim=-2)
