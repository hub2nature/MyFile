# -*- coding: utf-8 -*-
"""H- MolTrans.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/12WsUOZStIExf-l1dleAbNDzmmVd9aAc5
"""

from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/DTI_prediction/MolTrans/

# Commented out IPython magic to ensure Python compatibility.
# %cd MolTrans

!pip install torch

!pip install pandas

import os
import torch
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def set_seed(seed_value=42):
        os.environ['PYTHONHASHSEED'] = str(seed_value)
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False




set_seed(42)

"""# **DAVIS data on GCVAE**"""

!pip install torch_geometric

!pip install rdkit

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math
from torch_geometric.data import Data, DataLoader
from rdkit import Chem
from rdkit.Chem import rdmolops
from sklearn.metrics import roc_auc_score, roc_curve
from torch_geometric.nn import GCNConv, global_mean_pool

# Load and preprocess datasets
file_path = '/content/drive/MyDrive/DTI_prediction/MolTrans/MolTrans/dataset/DAVIS/train.csv'
df = pd.read_csv(file_path)
df = df.sample(n=1000, random_state=42).reset_index(drop=True)
df = df[['SMILES', 'Target Sequence', 'Label']]

df_val = pd.read_csv('/content/drive/MyDrive/DTI_prediction/MolTrans/MolTrans/dataset/DAVIS/val.csv')
df_val = df_val.sample(n=10, random_state=42).reset_index(drop=True)
df_val = df_val[['SMILES', 'Target Sequence', 'Label']]

df_test = pd.read_csv('/content/drive/MyDrive/DTI_prediction/MolTrans/MolTrans/dataset/DAVIS/test.csv')
df_test = df_test.sample(n=10, random_state=42).reset_index(drop=True)
df_test = df_test[['SMILES', 'Target Sequence', 'Label']]

# Function to convert SMILES to graph with padding and normalization
def smiles_to_graph(smiles, max_atoms):
    mol = Chem.MolFromSmiles(smiles)
    adj = rdmolops.GetAdjacencyMatrix(mol)
    features = np.zeros((max_atoms, 1))  # Only considering atomic number for simplicity
    for i, atom in enumerate(mol.GetAtoms()):
        features[i] = atom.GetAtomicNum() / 100.0  # Normalizing atomic number by dividing by 100
    adj_padded = np.zeros((max_atoms, max_atoms), dtype=int)
    adj_padded[:adj.shape[0], :adj.shape[1]] = adj
    return torch.tensor(features, dtype=torch.float), torch.tensor(adj_padded, dtype=torch.long)

# Function to convert protein sequences to token indices
def protein_to_indices(sequence, max_length, token_to_idx):
    indices = [token_to_idx[token] for token in sequence]
    indices += [0] * (max_length - len(indices))  # Pad to max_length
    return torch.tensor(indices, dtype=torch.long)

# Convert datasets
def create_graph_data(df, max_atoms, max_protein_len, token_to_idx):
    data_list = []
    for i, row in df.iterrows():
        x, adj = smiles_to_graph(row['SMILES'], max_atoms)
        edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        protein_indices = protein_to_indices(row['Target Sequence'], max_protein_len, token_to_idx)
        data = Data(x=x, edge_index=edge_index, proteins=protein_indices, y=torch.tensor([row['Label']], dtype=torch.float))
        data_list.append(data)
    return data_list

# Determine the maximum number of atoms and protein length
def get_max_atoms(df):
    max_atoms = 0
    for smiles in df['SMILES']:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            max_atoms = max(max_atoms, mol.GetNumAtoms())
    return max_atoms

def get_max_protein_len(df):
    return max(len(seq) for seq in df['Target Sequence'])

# Tokenization: Extract unique characters from protein sequences
protein_sequences = df['Target Sequence'].tolist()
protein_tokens = set(''.join(protein_sequences))
token_to_idx = {token: idx for idx, token in enumerate(protein_tokens, start=1)}

# Determine the maximum number of atoms and protein length in the training set
max_atoms = get_max_atoms(df)
max_protein_len = get_max_protein_len(df)

# Create graph data
train_data = create_graph_data(df, max_atoms, max_protein_len, token_to_idx)
val_data = create_graph_data(df_val, max_atoms, max_protein_len, token_to_idx)
test_data = create_graph_data(df_test, max_atoms, max_protein_len, token_to_idx)

# Create DataLoaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=10, shuffle=False)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)

print("Data loading complete.")

# Define the Hierarchical Transformer model for SMILES sequences
class HierarchicalSMILESTransformer(nn.Module):
    def __init__(self, num_tokens, embed_dim, num_heads, num_encoder_layers, dropout=0.5):
        super(HierarchicalSMILESTransformer, self).__init__()
        self.embedding = nn.Embedding(num_tokens, embed_dim)

        # Low-Level Transformer
        self.low_level_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True),
            num_layers=num_encoder_layers
        )

        # High-Level Transformer
        self.high_level_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True),
            num_layers=num_encoder_layers
        )

        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, smiles):
        # Embedding
        smiles_emb = self.embedding(smiles)

        # Low-Level Processing
        low_level_output = self.low_level_transformer(smiles_emb)

        # Pooling to reduce sequence length for high-level processing
        pooled_output = self.pooling(low_level_output)

        # High-Level Processing
        high_level_output = self.high_level_transformer(pooled_output)

        # Global average pooling
        smiles_encoded = high_level_output.mean(dim=1)

        return self.fc(smiles_encoded)

    def pooling(self, x):
        # Simple pooling mechanism to reduce sequence length, you can use more sophisticated pooling techniques
        return x[:, ::2, :]  # Example: take every second element, reducing sequence length by half

# Define the Hierarchical Transformer model for Protein sequences
class HierarchicalProteinTransformer(nn.Module):
    def __init__(self, num_tokens, embed_dim, num_heads, num_encoder_layers, dropout=0.5):
        super(HierarchicalProteinTransformer, self).__init__()
        self.embedding = nn.Embedding(num_tokens, embed_dim)

        # Low-Level Transformer
        self.low_level_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True),
            num_layers=num_encoder_layers
        )

        # High-Level Transformer
        self.high_level_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True),
            num_layers=num_encoder_layers
        )

        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, proteins):
        # Embedding
        proteins_emb = self.embedding(proteins)

        # Low-Level Processing
        low_level_output = self.low_level_transformer(proteins_emb)

        # Pooling to reduce sequence length for high-level processing
        pooled_output = self.pooling(low_level_output)

        # High-Level Processing
        high_level_output = self.high_level_transformer(pooled_output)

        # Global average pooling
        proteins_encoded = high_level_output.mean(dim=1)

        return self.fc(proteins_encoded)

    def pooling(self, x):
        # Simple pooling mechanism to reduce sequence length, you can use more sophisticated pooling techniques
        return x[:, ::2, :]  # Example: take every second element, reducing sequence length by half

# Define the combined DTI prediction model
class DTIModel(nn.Module):
    def __init__(self, smiles_transformer, protein_transformer, hidden_dim):
        super(DTIModel, self).__init__()
        self.smiles_transformer = smiles_transformer
        self.protein_transformer = protein_transformer
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, data, proteins):
        smiles_out = self.smiles_transformer(data.x.long())
        proteins_out = self.protein_transformer(proteins)
        combined = torch.cat((smiles_out, proteins_out), dim=1)
        x = torch.relu(self.fc1(combined))
        return torch.sigmoid(self.fc2(x))

# Initialize model, optimizer, and criterion
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_tokens = len(token_to_idx) + 1  # Plus one for padding token
embed_dim = 64
num_heads = 4
num_encoder_layers = 2

smiles_transformer = HierarchicalSMILESTransformer(num_tokens, embed_dim, num_heads, num_encoder_layers).to(device)
protein_transformer = HierarchicalProteinTransformer(num_tokens, embed_dim, num_heads, num_encoder_layers).to(device)
dti_model = DTIModel(smiles_transformer, protein_transformer, hidden_dim=embed_dim).to(device)

optimizer = optim.Adam(dti_model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Training and evaluation functions
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data, data.proteins)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data, data.proteins)
            loss = criterion(output, data.y)
            total_loss += loss.item() * data.num_graphs
            all_outputs.extend(output.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
    auc_score = roc_auc_score(all_labels, all_outputs)
    return total_loss / len(loader.dataset), auc_score

# Run training and evaluation
num_epochs = 50
train_losses = []
val_losses = []
val_auc_scores = []

for epoch in range(num_epochs):
    train_loss = train(dti_model, train_loader, optimizer, criterion, device)
    val_loss, val_auc = evaluate(dti_model, val_loader, criterion, device)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_auc_scores.append(val_auc)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}')

# Plotting the training and validation loss and AUC
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(val_auc_scores, label='Validation AUC')
plt.title('Validation AUC Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.grid(True)
plt.show()