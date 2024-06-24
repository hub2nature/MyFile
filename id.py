# -*- coding: utf-8 -*-
"""Id.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1dtPOnVkm3IFJWQoOk1KHJOnV8VK-HSe1
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import json

# Set the environment variable for CuBLAS to ensure determinism
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Seeds set and deterministic mode is enabled")

set_seed(42)

def load_data(file_path, sample_size=10, random_state=42):
    df = pd.read_csv(file_path)
    df_sampled = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
    return df_sampled[['SMILES', 'Target Sequence', 'Label']]

train_df = load_data('/mnt/research/Datta_Aniruddha/Students/Mondal_Madhurima/MolTrans/dataset/DAVIS/train.csv')
val_df = load_data('/mnt/research/Datta_Aniruddha/Students/Mondal_Madhurima/MolTrans/dataset/DAVIS/val.csv')
test_df = load_data('/mnt/research/Datta_Aniruddha/Students/Mondal_Madhurima/MolTrans/dataset/DAVIS/test.csv')

# Print samples for verification
print('Training sample:', train_df.head())
print('Validation sample:', val_df.head())
print('Test sample:', test_df.head())

# Tokenization: Extract unique characters from SMILES and protein sequences
df=train_df
df_val=val_df
df_test=test_df
smiles_strings = df['SMILES'].tolist()
protein_sequences = df['Target Sequence'].tolist()

smiles_strings_val = df_val['SMILES'].tolist()
protein_sequences_val = df_val['Target Sequence'].tolist()

smiles_strings_test = df_test['SMILES'].tolist()
protein_sequences_test = df_test['Target Sequence'].tolist()




# Combine all tokens from training, validation, and test sets
all_smiles_tokens = set(''.join(smiles_strings + smiles_strings_val + smiles_strings_test))
all_protein_tokens = set(''.join(protein_sequences + protein_sequences_val + protein_sequences_test))
all_tokens = list(all_smiles_tokens.union(all_protein_tokens))

# Convert tokens to numerical representations
token_to_idx = {token: idx for idx, token in enumerate(all_tokens)}

def tokenize_and_pad(sequences, token_to_idx, max_len):
    token_sequences = [[token_to_idx[token] for token in seq] for seq in sequences]
    padded_sequences = [seq + [0] * (max_len - len(seq)) for seq in token_sequences]
    return torch.tensor(padded_sequences, dtype=torch.long).to(device)

# Find max lengths
max_smiles_len = max(max(len(seq) for seq in smiles_strings),
                     max(len(seq) for seq in smiles_strings_val),
                     max(len(seq) for seq in smiles_strings_test))
max_protein_len = max(max(len(seq) for seq in protein_sequences),
                      max(len(seq) for seq in protein_sequences_val),
                      max(len(seq) for seq in protein_sequences_test))

# Tokenize and pad sequences
smiles_train = tokenize_and_pad(smiles_strings, token_to_idx, max_smiles_len)
protein_train = tokenize_and_pad(protein_sequences, token_to_idx, max_protein_len)
labels_train = torch.tensor(df['Label'].values, dtype=torch.float32).view(-1, 1).to(device)

smiles_val = tokenize_and_pad(smiles_strings_val, token_to_idx, max_smiles_len)
protein_val = tokenize_and_pad(protein_sequences_val, token_to_idx, max_protein_len)
labels_val = torch.tensor(df_val['Label'].values, dtype=torch.float32).view(-1, 1).to(device)

smiles_test = tokenize_and_pad(smiles_strings_test, token_to_idx, max_smiles_len)
protein_test = tokenize_and_pad(protein_sequences_test, token_to_idx, max_protein_len)
labels_test = torch.tensor(df_test['Label'].values, dtype=torch.float32).view(-1, 1).to(device)

from torch.utils.data import Dataset

class DTI_Dataset(Dataset):
    def __init__(self, smiles, proteins, labels):
        self.smiles = smiles
        self.proteins = proteins
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.smiles[idx], self.proteins[idx], self.labels[idx]


from torch.utils.data import DataLoader

batch_size = 10

train_dataset = DTI_Dataset(smiles_train, protein_train, labels_train)
val_dataset = DTI_Dataset(smiles_val, protein_val, labels_val)
test_dataset = DTI_Dataset(smiles_test, protein_test, labels_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=0)

# Define the Transformer model for SMILES sequences
class SMILESTransformer(nn.Module):
    def __init__(self, num_tokens, embed_dim, num_heads, num_encoder_layers, dropout=0.5):
        super(SMILESTransformer, self).__init__()
        self.embedding = nn.Embedding(num_tokens, embed_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True),
            num_layers=num_encoder_layers
        )

        self.fc = nn.Linear(embed_dim, embed_dim)
        self.print_initial_weights()
    def print_initial_weights(self):
        print("Initialized weights for SMILES Transformer:")
        print("Embedding weights:", self.embedding.weight.data)
        for idx, layer in enumerate(self.encoder.layers):
            print(f"Encoder Layer {idx} - Self Attention weights:", layer.self_attn.in_proj_weight.data)
            print(f"Encoder Layer {idx} - Self Attention bias:", layer.self_attn.in_proj_bias.data)

    def forward(self, smiles):
        smiles_emb = self.embedding(smiles)  # Transformer with batch_first=True expects input as (batch_size, seq_len, embed_dim)
        smiles_encoded = self.encoder(smiles_emb).mean(dim=1)  # Global average pooling
        return self.fc(smiles_encoded)

# Define the Transformer model for Protein sequences
class ProteinTransformer(nn.Module):
    def __init__(self, num_tokens, embed_dim, num_heads, num_encoder_layers, dropout=0.5):
        super(ProteinTransformer, self).__init__()
        self.embedding = nn.Embedding(num_tokens, embed_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True),
            num_layers=num_encoder_layers
        )
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.print_initial_weights()
    def print_initial_weights(self):

        print("Initialized weights for Protein Transformer:")
        print("Embedding weights:", self.embedding.weight.data)
        for idx, layer in enumerate(self.encoder.layers):
            print(f"Encoder Layer {idx} - Self Attention weights:", layer.self_attn.in_proj_weight.data)
            print(f"Encoder Layer {idx} - Self Attention bias:", layer.self_attn.in_proj_bias.data)
    def forward(self, proteins):
        proteins_emb = self.embedding(proteins)  # Transformer with batch_first=True expects input as (batch_size, seq_len, embed_dim)
        proteins_encoded = self.encoder(proteins_emb).mean(dim=1)  # Global average pooling
        return self.fc(proteins_encoded)

class DTIModel(nn.Module):
    def __init__(self, smiles_transformer, protein_transformer, embed_dim):
        super(DTIModel, self).__init__()
        self.smiles_transformer = smiles_transformer
        self.protein_transformer = protein_transformer
        torch.manual_seed(42)
        self.fc = nn.Linear(embed_dim * 2, 1)

    def forward(self, smiles, proteins):
        smiles_out = self.smiles_transformer(smiles)
        proteins_out = self.protein_transformer(proteins)
        combined = torch.cat((smiles_out, proteins_out), dim=1)
        return self.fc(combined)

# Model Initialization
from sklearn.metrics import roc_auc_score, roc_curve
from torch.optim.lr_scheduler import StepLR
num_tokens = len(token_to_idx)
embed_dim = 8
num_heads = 4
num_encoder_layers = 2

smiles_transformer = SMILESTransformer(num_tokens, embed_dim, num_heads, num_encoder_layers)
protein_transformer = ProteinTransformer(num_tokens, embed_dim, num_heads, num_encoder_layers)
dti_model = DTIModel(smiles_transformer, protein_transformer, embed_dim)
print(dti_model)

# Define a function to run the model
def run_model(epochs=100):
    # Define and initialize the model, optimizer, etc.
    model = DTIModel(smiles_transformer, protein_transformer,embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    initial_weights = model.state_dict()  # Save initial weights

    # Training loop
    for epoch in range(epochs):
        model.train()
        # Run your training process
        pass

    final_weights = model.state_dict()  # Save final weights
    return initial_weights, final_weights

# Function to compare the results of two runs
def compare_runs(run1_data, run2_data):
    for key in run1_data.keys():
        if torch.equal(run1_data[key], run2_data[key]):
            print(f'{key} weights are identical between runs.')
        else:
            print(f'{key} weights differ between runs.')

# Running the model twice
initial_weights_run1, final_weights_run1 = run_model()
initial_weights_run2, final_weights_run2 = run_model()

# Comparing initial and final states between two runs
print("Comparing initial weights between two runs:")
compare_runs(initial_weights_run1, initial_weights_run2)

print("Comparing final weights between two runs:")
compare_runs(final_weights_run1, final_weights_run2)