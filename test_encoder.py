from collections import defaultdict
import json
import os
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from preprocessing import TTSDataItem

class SpeakerEncoder(nn.Module):
    def __init__(
        self,
        n_mel_bands=80,
        prenet_layers=2,
        prenet_dim=128,
        conv_layers=2,
        conv_kernel_size=5,
        F_mapped=128,       
        attention_dim=64,          
        embedding_dim=16,    
        dropout=0.1,
        attention_heads=2    
    ):
        super().__init__()
        
        self.n_mel_bands = n_mel_bands
        self.prenet_dim = prenet_dim
        self.F_mapped = F_mapped
        self.d_attn = attention_dim
        self.d_embedding = embedding_dim
        self.attention_heads = attention_heads

        # Prenet 
        # Input: (B, N, T, F) -> reshape to (B*N, T, F), apply prenet
        self.prenet = nn.ModuleList([nn.Linear(n_mel_bands, prenet_dim)])        
        for i in range(prenet_layers - 1):
            self.prenet.append(nn.Linear(prenet_dim, prenet_dim))
            
        # Temporal Processing
        self.conv_layers = nn.ModuleList()
        for _ in range(conv_layers):
            self.conv_layers.append(
                nn.Conv1d(in_channels=prenet_dim,
                          out_channels=prenet_dim*2,
                          kernel_size=conv_kernel_size,
                          padding=(conv_kernel_size- 1)//2)
            )
        self.dropout = nn.Dropout(dropout)

        assert prenet_dim == F_mapped, "For simplicity, ensure prenet_dim == F_mapped"

        # attn fc layers
        # (batch, N_samples, F_mapped) -> FC -> ELU -> (batch, N_samples, d_attn)
        self.heads = attention_heads
        total_attention_dim =  attention_dim * self.heads
        self.query_fc = nn.Linear(F_mapped, total_attention_dim)
        self.key_fc   = nn.Linear(F_mapped, total_attention_dim)
        self.value_fc = nn.Linear(F_mapped, total_attention_dim)

        self.post_mha_fc = nn.Linear(attention_dim * self.heads, 1)  # from (B,N,d_attn*heads)->(B,N)

        # (batch, N_samples, F_mapped) -> fc -> (batch, N_samples, d_embedding)
        self.skip_fc = nn.Linear(F_mapped, embedding_dim)

    def forward(self, mel_specs, T_actual):
        B, N, T, F = mel_specs.size()
        
        x = mel_specs.view(-1, T, F)

        # Prenet
        for fc in self.prenet:
            x = fc(x)
            x = nn.functional.elu(x, inplace=True)

        # Mid net
        x = x.transpose(1, 2)  # (B*N, prenet_dim, T)        
        for conv in self.conv_layers:
            out = conv(x)
            out = self.dropout(out)
            out = nn.functional.glu(out, dim=1)
            out = out + x  # residual
            x = out
        
        # Global mean pooling (B*N, prenet_dim, T)
        # a mask for each vector corresponding to the actual T
        mean_mask = torch.arange(x.size(2)).unsqueeze(0) < torch.tensor(T_actual).unsqueeze(1)
        mean_mask = mean_mask.unsqueeze(1).expand(-1, x.size(1), -1)

        # zero out the padded entries, and get the correct counts
        masked_sum = (x * mean_mask).sum(-1)
        masked_cnt = mean_mask.sum(dim=-1) 
        
        x = masked_sum/masked_cnt# (B*N, prenet_dim)
        x = x.view(B, N, self.prenet_dim)  # (B, N, F_mapped)        
        skip_repr = x  # (B, N, F_mapped)

        # Q, K, V: (B, N, F_mapped) -> FC -> ELU -> (B,N, heads*d_attn)
        Q = nn.functional.elu(self.query_fc(x), inplace=True)  # (B, N, heads*d_attn)
        K = nn.functional.elu(self.key_fc(x), inplace=True)
        V = nn.functional.elu(self.value_fc(x), inplace=True)

        # Multi-Head Attention:
        d_head = self.d_attn
        H = self.heads
        d_total = d_head * H  
        Q = Q.view(B, N, H, d_head)  # (B, N, H, d_head)
        K = K.view(B, N, H, d_head)
        V = V.view(B, N, H, d_head)

        # Permute to (B,H,N,d_head)
        Q = Q.permute(0, 2, 1, 3)  # (B,H,N,d_head)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)
        
        # scores: (B,H,N,N)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / (d_head**0.5)
        attn = torch.softmax(scores, dim=-1)  # (B,H,N,N)
        context = torch.matmul(attn, V)  # (B,H,N,d_head)

        #rearrange context to (B,N,H,d_head)
        context = context.permute(0, 2, 1, 3).contiguous() # (B,N,H,d_head)
        context = context.view(B, N, d_total)  # (B,N,heads*d_attn) = (B,N,d_attn*heads)

        attn_fc_out = self.post_mha_fc(context).squeeze(-1)  # (B,N)    
        # deviates from paper, but corresponds with attention. not sure why they use softsign
        attn_weights = torch.softmax(attn_fc_out, dim=-1).unsqueeze(-1) #(B,N)

        # # as depicted in the paper, but not standard? 
        # attn_fc_out = nn.functional.softsign(attn_fc_out)
        # norm = torch.norm(attn_fc_out, p=2, dim=-1, keepdim=True) + 1e-9
        # attn_fc_out = attn_fc_out / norm  # (B,N)        
        # attn_weights = attn_fc_out.unsqueeze(-1)  # (B,N,1)

        skip_emb = self.skip_fc(skip_repr) # (B, N, embedding_dim)
        weighted_sum = attn_weights * skip_emb  # (B, N, embedding_dim)

        # Sum over N samples:
        speaker_embedding = weighted_sum.sum(dim=1)  # (B, embedding_dim)

        return speaker_embedding

class VCTKDataset(Dataset):
    def __init__(self, processed_path, accept_ids = None, with_linear_spec = False, transpose_specs = True):
        """
        transpose_specs: swap the last two dimensions. Encoder architecture assumes T,F order while melspec is in F, T order
        """
        super().__init__()

        # indexed by speaker id
        self.data = []
        self.speaker_map = {}
        
        for folder in tqdm(sorted(os.listdir(processed_path,)), desc="Loading dataset"):
            speaker_id, utterance_num = folder.split('_')
            if (not speaker_id 
                or not speaker_id.startswith('p') 
                or (accept_ids and speaker_id not in accept_ids)
                or not utterance_num                
            ):
                continue

            tts_data_json = self._get_file_json(f'{processed_path}/{folder}/data.dat')             
            tts_data_json['linear_spec'] = torch.load(f'{folder}/linear_spec', weights_only=True) if with_linear_spec else None
            tts_data_json['mel_spec'] = torch.load(f'{processed_path}/{folder}/mel_spec', weights_only=True)
            if transpose_specs:
                tts_data_json['mel_spec'] = tts_data_json['mel_spec'].T
            tts_data_json['T'] = tts_data_json['mel_spec'].shape[0]
            tts_data_item = TTSDataItem( **tts_data_json )
        
            if speaker_id not in self.speaker_map:
                self.speaker_map[speaker_id] = len(self.data)
                self.data.append([])
            
            self.data[self.speaker_map[speaker_id]].append(tts_data_item)
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
    def _get_file_json(self, file_path):
        with open(file_path, 'r') as fp:
            return json.load(fp)

def collate_fn(batch, n_samples = 5):
    """
    batch: list of length `batch_size`, where each element is a list of TTSDataItems for that speaker.
    Returns: (batch, N_samples, T, F) tensor of mel spectrograms
    """

    batch_size = len(batch)

    selected_utterances = []
    for speaker_utterances in batch:
        if len(speaker_utterances) < n_samples:
            # If not enough utterances sample with replacement:
            chosen = [random.choice(speaker_utterances) for _ in range(n_samples)]
        else:
            chosen = random.sample(speaker_utterances, n_samples)
        selected_utterances.append(chosen)

    # selected_utterances shape (batch_size, N_samples)
    max_T = max(utt.mel_spec.size(0) for speaker_batch in selected_utterances for utt in speaker_batch)
    F = selected_utterances[0][0].mel_spec.size(1)
    
    #padding
    padded_mels = torch.zeros(batch_size, n_samples, max_T, F)
    T_actual = []
    for b_idx in range(batch_size):
        for n_idx in range(n_samples):
            mel = selected_utterances[b_idx][n_idx].mel_spec    
            T_actual.append(mel.size(0))
            padded_mels[b_idx, n_idx, :mel.size(0), :] = mel

    speaker_ids = [it[0].speaker_id for it in batch]    
    return speaker_ids, T_actual, padded_mels    

def get_speaker_id_map(file_path = './data/VCTK-Corpus/speaker-info-old.txt'):
    speaker_id_map = {}
    with open(file_path, 'r') as fp:
        next(fp)
        for row in fp:
            id = 'p' + row.split()[0] #add leading p to align it with newer version
            speaker_id_map[id] = len(speaker_id_map)

    return speaker_id_map

def get_speaker_embeddings(file_path = 'speaker_embeddings.pth'):
    # get old speaker info
    with open(file_path, 'rb') as fp:
       speaker_embeddings = torch.load(fp)

    return speaker_embeddings

# Example usage
if __name__ == "__main__":    
    speaker_id_map = get_speaker_id_map() 
    speaker_embeddings: torch.nn.Embedding = get_speaker_embeddings()

    dataset = VCTKDataset('./processed/VCTK_4096_512', accept_ids = speaker_id_map.keys())
    
    dataloader = DataLoader(
        dataset,
        batch_size = 2,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, 8)
    )

    model = SpeakerEncoder(
        n_mel_bands=80,
        prenet_layers=2,
        prenet_dim=64,
        F_mapped=64, 
        conv_layers=6,
        conv_kernel_size=5,
      
        attention_dim=32,          
        embedding_dim=16,    
        dropout=0.1,
        attention_heads=1
    )

    optimizer = torch.optim.Adam(model.parameters(), lr = 5e-4)

    SAVE_DIR = 'output/savepoints'
    os.makedirs(SAVE_DIR, exist_ok=True)

    epoch = 0
    max_epochs = 1000

    for epoch in range(max_epochs):
        epoch_loss = 0
        for speaker_ids, T_actual, batch in dataloader:            
            r = model.forward(batch, T_actual)
            true_embeddings = speaker_embeddings(torch.tensor([speaker_id_map[id] for id in speaker_ids]))

            loss = torch.nn.functional.l1_loss(r, true_embeddings)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch:20}\t{epoch_loss:20,.4f}")

        if epoch_loss < 1e-6:
            break

        if not epoch % 25:
            torch.save(model, f'{SAVE_DIR}/speaker_encoder_mode_{epoch}.pth')

    torch.save(model, f'{SAVE_DIR}/speaker_encoder_mode_FINAL.pth')
