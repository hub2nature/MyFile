
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import codecs
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import math
import time
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve, average_precision_score, confusion_matrix, recall_score, precision_score, f1_score
from torch.autograd import Variable
import sys
import os
import random
from subword_nmt.apply_bpe import BPE

# Set up for reproducibility and output redirection
seeds = [42, 123, 456, 782, 102]
if os.path.exists("their_code_curcumin_5seeds.txt"):
    os.remove("their_code_curcumin_5seeds.txt")
sys.stdout = open('their_code_curcumin_5seeds.txt', 'w')


# Initialize BPE for drug and protein sequences
vocab_path_protein = './ESPF/protein_codes_uniprot.txt'
bpe_codes_protein = codecs.open(vocab_path_protein)
pbpe = BPE(bpe_codes_protein, merges=-1, separator='')

sub_csv_protein = pd.read_csv('./ESPF/subword_units_map_uniprot.csv')
idx2word_p = sub_csv_protein['index'].values
words2idx_p = dict(zip(idx2word_p, range(0, len(idx2word_p))))

vocab_path_drug = './ESPF/drug_codes_chembl.txt'
bpe_codes_drug = codecs.open(vocab_path_drug)
dbpe = BPE(bpe_codes_drug, merges=-1, separator='')

sub_csv_drug = pd.read_csv('./ESPF/subword_units_map_chembl.csv')
idx2word_d = sub_csv_drug['index'].values
words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

def drug2emb_encoder(x, max_d):
    t1 = dbpe.process_line(x).split()
    try:
        i1 = np.asarray([words2idx_d.get(i, 0) for i in t1])
    except KeyError:
        i1 = np.array([0])

    l = len(i1)
    if l < max_d:
        i = np.pad(i1, (0, max_d - l), 'constant', constant_values=0)
        input_mask = ([1] * l) + ([0] * (max_d - l))
    else:
        i = i1[:max_d]
        input_mask = [1] * max_d

    return i, np.asarray(input_mask)

def protein2emb_encoder(x, max_p):
    t1 = pbpe.process_line(x).split()
    try:
        i1 = np.asarray([words2idx_p.get(i, 0) for i in t1])
    except KeyError:
        i1 = np.array([0])

    l = len(i1)
    if l < max_p:
        i = np.pad(i1, (0, max_p - l), 'constant', constant_values=0)
        input_mask = ([1] * l) + ([0] * (max_p - l))
    else:
        i = i1[:max_p]
        input_mask = [1] * max_p

    return i, np.asarray(input_mask)

# Negative sampling function
def negative_sampling(df, num_neg_samples=None):
    df_neg = df.copy()
    df_neg['Label'] = 0
    df_neg['SMILES'] = df_neg['SMILES'].sample(frac=1.0, random_state=42).reset_index(drop=True)
    df_neg['Target Sequence'] = df_neg['Target Sequence'].sample(frac=1.0, random_state=42).reset_index(drop=True)

    if num_neg_samples:
        df_neg = df_neg.sample(n=num_neg_samples, random_state=42).reset_index(drop=True)

    return df_neg

# Dataset class for DTI
class DTI_Dataset(Dataset):
    def __init__(self, df, labels, drug_dim, protein_dim):
        self.df = df
        self.labels = labels
        self.max_d = drug_dim
        self.max_p = protein_dim

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        d = self.df.iloc[index]['SMILES']
        p = self.df.iloc[index]['Target Sequence']
        y = self.labels[index]

        d_v, d_mask = drug2emb_encoder(d, self.max_d)
        p_v, p_mask = protein2emb_encoder(p, self.max_p)

        return torch.tensor(d_v, dtype=torch.long), torch.tensor(p_v, dtype=torch.long), torch.tensor(d_mask, dtype=torch.float32), torch.tensor(p_mask, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Embeddings class
class Embeddings(nn.Module):
    def __init__(self, vocab_size, emb_size, max_seq_len, dropout_rate):
        super(Embeddings, self).__init__()
        self.emb = nn.Embedding(vocab_size, emb_size)
        self.position_embeddings = nn.Embedding(max_seq_len, emb_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        word_embeddings = self.emb(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = word_embeddings + position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, emb_size, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.emb_size = emb_size

        self.query = nn.Linear(emb_size, emb_size)
        self.key = nn.Linear(emb_size, emb_size)
        self.value = nn.Linear(emb_size, emb_size)

        self.fc_out = nn.Linear(emb_size, emb_size)

    def forward(self, x):
        batch_size, seq_length, emb_size = x.shape
        assert emb_size == self.emb_size

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Split embedding into self.num_heads different pieces
        q = q.view(batch_size, seq_length, self.num_heads, self.emb_size // self.num_heads)
        k = k.view(batch_size, seq_length, self.num_heads, self.emb_size // self.num_heads)
        v = v.view(batch_size, seq_length, self.num_heads, self.emb_size // self.num_heads)

        # Transpose to get dimensions [batch_size, num_heads, seq_length, emb_size // num_heads]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        energy = torch.matmul(q, k.permute(0, 1, 3, 2)) / (self.emb_size ** (1 / 2))
        attention = torch.softmax(energy, dim=-1)

        out = torch.matmul(attention, v)

        # Reshape back to [batch_size, seq_length, emb_size]
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(batch_size, seq_length, emb_size)

        out = self.fc_out(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, emb_size):
        super(ResidualBlock, self).__init__()
        self.multi_head_attention = MultiHeadSelfAttention(emb_size, num_heads=8)
        self.layer_norm = nn.LayerNorm(emb_size)
        self.fc = nn.Sequential(
            nn.Linear(emb_size, emb_size * 4),
            nn.ReLU(),
            nn.Linear(emb_size * 4, emb_size)
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Apply multi-head self-attention and add the residual
        attention = self.multi_head_attention(x)
        x = self.layer_norm(attention + x)

        # Apply feed-forward network and add the residual
        fc_out = self.fc(x)
        x = self.layer_norm(fc_out + x)

        return x

class SpatialAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpatialAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, H, W = x.size()

        # Apply convolutions to get the query, key, and value tensors
        query = self.query_conv(x).view(batch_size, -1, W * H)  # B x C' x (H * W)
        key = self.key_conv(x).view(batch_size, -1, W * H)  # B x C' x (H * W)
        value = self.value_conv(x).view(batch_size, -1, W * H)  # B x C' x (H * W)

        # Transpose key tensor for matrix multiplication
        key = key.permute(0, 2, 1)  # B x (H * W) x C'

        # Compute the attention map
        attention = torch.bmm(query, key)  # B x C' x C'
        attention = F.softmax(attention, dim=-1)

        # Apply the attention map to the value tensor
        out = torch.bmm(attention, value)  # B x C' x (H * W)

        # Reshape the output tensor back to the original spatial dimensions
        out = out.view(batch_size, C, H, W)  # B x C' x H x W

        # Combine the attention output with the original input (skip connection)
        out = self.gamma * out + x

        return out

class BIN_Interaction_With_SpatialAttention(nn.Module):
    def __init__(self, **config):
        super(BIN_Interaction_With_SpatialAttention, self).__init__()
        self.max_d = config['max_drug_seq']
        self.max_p = config['max_protein_seq']
        self.emb_size = config['emb_size']
        self.dropout_rate = config['dropout_rate']
        self.num_heads = config['num_attention_heads']

        self.demb = Embeddings(config['input_dim_drug'], self.emb_size, self.max_d, self.dropout_rate)
        self.pemb = Embeddings(config['input_dim_target'], self.emb_size, self.max_p, self.dropout_rate)

        self.residual_block_drug = ResidualBlock(self.emb_size)
        self.residual_block_protein = ResidualBlock(self.emb_size)

        self.spatial_attention = SpatialAttention(in_channels=32, out_channels=32)

        self.decoder = nn.Sequential(
            nn.Linear(32*50*100, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512, 64),
            nn.ReLU(True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 1)
        )

    def forward(self, d, p, d_mask, p_mask):
        # print('shape of d',d.shape)
        batch_size = d.size(0)
        # print(f"batch_size: {batch_size}")

        d_emb = self.demb(d)
        p_emb = self.pemb(p)

        # Apply residual blocks
        d_emb = self.residual_block_drug(d_emb)
        p_emb = self.residual_block_protein(p_emb)

        d_emb = d_emb.transpose(0, 1)
        p_emb = p_emb.transpose(0, 1)

        d_aug = d_emb.unsqueeze(1).expand(-1, self.max_p, -1, -1)
        p_aug = p_emb.unsqueeze(0).expand(self.max_d, -1, -1, -1)

        pair_act = d_aug * p_aug

        # Transpose to get channels as the second dimension
        pair_act = pair_act.permute(2, 3, 0, 1)  # New shape should be [batch_size, channels, H, W]

        i_v = self.spatial_attention(pair_act)

        f = i_v.view(batch_size, -1)

        score = self.decoder(f)

        return score

# Updated configuration function
def BIN_config_DBPE():
    config = {}
    config['batch_size'] = 8
    config['input_dim_drug'] = 23532
    config['input_dim_target'] = 16693
    config['train_epoch'] = 50
    config['max_drug_seq'] = 50
    config['max_protein_seq'] = 100
    config['emb_size'] = 32
    config['dropout_rate'] = 0.2

    config['num_attention_heads'] = 8
    config['gating'] = False

    config['flat_dim'] = 2500
    return config
# Function to predict DTI probabilities for curcumin and return top 5 gene symbols (ID2)
def predict_curcumin_dti_probabilities(curcumin_smiles, model, df_mine, max_d, max_p):
    # Convert curcumin SMILES to embeddings
    curcumin_emb, curcumin_mask = drug2emb_encoder(curcumin_smiles, max_d)
    curcumin_emb_tensor = torch.tensor(curcumin_emb).unsqueeze(0).long().cuda()
    curcumin_mask_tensor = torch.tensor(curcumin_mask).unsqueeze(0).float().cuda()

    # Initialize list to store predictions
    protein_preds = []

    with torch.no_grad():
        for idx, row in df_mine.iterrows():
            protein = row['X2']  # Get the protein sequence (X2)
            gene_symbol = row['ID2']  # Get the gene symbol (ID2)

            # Convert each unique protein sequence to embeddings
            protein_emb, protein_mask = protein2emb_encoder(protein, max_p)
            protein_emb_tensor = torch.tensor(protein_emb).unsqueeze(0).long().cuda()
            protein_mask_tensor = torch.tensor(protein_mask).unsqueeze(0).float().cuda()

            # Predict DTI probability for curcumin against the current protein
            score = model(curcumin_emb_tensor, protein_emb_tensor, curcumin_mask_tensor, protein_mask_tensor)
            score = torch.sigmoid(score).item()  # Convert score to probability

            # Store the gene symbol (ID2) and the predicted score
            protein_preds.append((gene_symbol, score))

    # Sort predictions by probability and get the top 5
    top_5_preds = sorted(protein_preds, key=lambda x: x[1], reverse=True)[:5]

    print("\nTop 5 predicted gene symbols for Curcumin:")
    for gene_symbol, score in top_5_preds:
        print(f"Gene Symbol: {gene_symbol}, Predicted DTI Probability: {score:.4f}")

    return top_5_preds


# Function for testing the model with optimal threshold using ROC curve
def test_with_optimal_threshold(data_generator, model, pos_weight):
    y_pred = []
    y_label = []
    model.eval()
    loss_accumulate = 0.0
    count = 0.0

    with torch.no_grad():
        for d, p, d_mask, p_mask, label in data_generator:
            # print(f"drug test embedding shape: {d.shape}")
            # print(f"Protein test embedding shape: {p.shape}")

            score = model(d.long().cuda(), p.long().cuda(), d_mask.long().cuda(), p_mask.long().cuda())
            logits = torch.squeeze(score)

            if logits.size(0) > label.size(0):
                logits = logits[:label.size(0)]
            elif label.size(0) > logits.size(0):
                label = label[:logits.size(0)]

            loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).cuda())
            label = Variable(torch.from_numpy(np.array(label)).float()).cuda()

            loss = loss_fct(logits, label)

            loss_accumulate += loss
            count += 1

            logits = torch.sigmoid(logits).detach().cpu().numpy()
            label_ids = label.to('cpu').numpy()
            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + logits.flatten().tolist()

    loss = loss_accumulate / count

    # Use ROC curve to calculate optimal threshold
    fpr, tpr, thresholds_roc = roc_curve(y_label, y_pred)
    optimal_threshold_index = np.argmax(tpr - fpr)
    optimal_threshold = thresholds_roc[optimal_threshold_index]
    print("Optimal threshold (ROC curve): " + str(optimal_threshold))

    y_pred_s = [1 if i >= optimal_threshold else 0 for i in y_pred]

    auc_roc = roc_auc_score(y_label, y_pred)
    auc_pr = average_precision_score(y_label, y_pred)

    precision = precision_score(y_label, y_pred_s)
    recall = recall_score(y_label, y_pred_s)
    f1 = f1_score(y_label, y_pred_s)

    cm = confusion_matrix(y_label, y_pred_s)
    accuracy = (cm[0, 0] + cm[1, 1]) / np.sum(cm)
    sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

    print(f'AUC ROC: {auc_roc}')
    print(f'AUC PR: {auc_pr}')
    print(f'F1 Score: {f1}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'Accuracy: {accuracy}')
    print(f'Sensitivity: {sensitivity}')
    print(f'Specificity: {specificity}')
    print(f'Test Loss: {loss.item()}')

    return auc_roc, auc_pr, f1, loss.item(), precision, recall, sensitivity, specificity, accuracy, optimal_threshold


# Early stopping function
def early_stopping(val_auc_list, patience=10):
    if len(val_auc_list) > patience:
        if all(val_auc_list[-patience] >= val for val in val_auc_list[-patience:]):
            return True
    return False

def run_for_seeds(seeds, lr):
    all_test_roc_auc = []
    all_test_auprc = []
    all_test_f1 = []
    all_test_loss = []
    all_test_precision = []
    all_test_recall = []
    all_test_sensitivity = []
    all_test_specificity = []
    all_test_accuracy = []
    all_optimal_thresholds = []

    for seed in seeds:
        # Set seed for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        fold_n = seed  # Using seed as fold identifier
        s = time.time()

        model_max, loss_history, test_metrics = main(fold_n, lr)
        e = time.time()

        # Collect metrics for each seed
        test_roc_auc, test_auprc, test_f1, test_loss, precision, recall, sensitivity, specificity, accuracy, optimal_threshold = test_metrics
        # optimal_threshold = test_metrics[-1]  # Assuming the last return value is optimal threshold

        all_test_roc_auc.append(test_roc_auc)
        all_test_auprc.append(test_auprc)
        all_test_f1.append(test_f1)
        all_test_loss.append(test_loss)
        all_test_precision.append(precision)
        all_test_recall.append(recall)
        all_test_sensitivity.append(sensitivity)
        all_test_specificity.append(specificity)
        all_test_accuracy.append(accuracy)
        all_optimal_thresholds.append(optimal_threshold)

        print(f"Seed {seed} - Total time: {e - s} seconds")
        print(f"Test ROC AUC: {test_roc_auc}, Test AUPRC: {test_auprc}, Test F1: {test_f1}, Test Loss: {test_loss}\n")

    # Print mean and standard deviation of test metrics
    print(f"Mean Test ROC AUC: {np.mean(all_test_roc_auc):.4f}, Std: {np.std(all_test_roc_auc):.4f}")
    print(f"Mean Test AUPRC: {np.mean(all_test_auprc):.4f}, Std: {np.std(all_test_auprc):.4f}")
    print(f"Mean Test F1: {np.mean(all_test_f1):.4f}, Std: {np.std(all_test_f1):.4f}")
    print(f"Mean Test Loss: {np.mean(all_test_loss):.4f}, Std: {np.std(all_test_loss):.4f}")
    print(f"Mean Precision: {np.mean(all_test_precision):.4f}, Std: {np.std(all_test_precision):.4f}")
    print(f"Mean Recall: {np.mean(all_test_recall):.4f}, Std: {np.std(all_test_recall):.4f}")
    print(f"Mean Sensitivity: {np.mean(all_test_sensitivity):.4f}, Std: {np.std(all_test_sensitivity):.4f}")
    print(f"Mean Specificity: {np.mean(all_test_specificity):.4f}, Std: {np.std(all_test_specificity):.4f}")
    print(f"Mean Accuracy: {np.mean(all_test_accuracy):.4f}, Std: {np.std(all_test_accuracy):.4f}")
    print(f"Mean Optimal Threshold: {np.mean(all_optimal_thresholds):.4f}, Std: {np.std(all_optimal_thresholds):.4f}")

    return {
        'roc_auc_mean': np.mean(all_test_roc_auc),
        'roc_auc_std': np.std(all_test_roc_auc),
        'auprc_mean': np.mean(all_test_auprc),
        'auprc_std': np.std(all_test_auprc),
        'f1_mean': np.mean(all_test_f1),
        'f1_std': np.std(all_test_f1),
        'loss_mean': np.mean(all_test_loss),
        'loss_std': np.std(all_test_loss),
        'precision_mean': np.mean(all_test_precision),
        'precision_std': np.std(all_test_precision),
        'recall_mean': np.mean(all_test_recall),
        'recall_std': np.std(all_test_recall),
        'sensitivity_mean': np.mean(all_test_sensitivity),
        'sensitivity_std': np.std(all_test_sensitivity),
        'specificity_mean': np.mean(all_test_specificity),
        'specificity_std': np.std(all_test_specificity),
        'accuracy_mean': np.mean(all_test_accuracy),
        'accuracy_std': np.std(all_test_accuracy),
        'optimal_threshold_mean': np.mean(all_optimal_thresholds),
        'optimal_threshold_std': np.std(all_optimal_thresholds)
    }

def main(fold_n, lr):
    config = BIN_config_DBPE()
    loss_history = []
    val_auc_list = []
    patience = 10

    model = BIN_Interaction_With_SpatialAttention(**config)
    model = model.cuda()

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, dim=0)

    def weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    model.apply(weights_init)

    # Calculate weights for weighted loss
    dataFolder = './dataset/DAVIS'
    df_train_balanced = pd.read_csv(dataFolder + '/train.csv')
    pos_weight = df_train_balanced['Label'].value_counts()[0] / df_train_balanced['Label'].value_counts()[1]

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    df_val_balanced = pd.read_csv(dataFolder + '/val.csv')
    df_test_balanced = pd.read_csv(dataFolder + '/test.csv')

    # df_train_balanced=df_train_balanced.sample(frac=0.01,random_state=42)
    # df_val_balanced=df_val_balanced.sample(frac=0.01,random_state=42)
    # df_test_balanced=df_test_balanced.sample(frac=0.01,random_state=42)

    training_set = DTI_Dataset(df_train_balanced, df_train_balanced.Label.values, config['max_drug_seq'], config['max_protein_seq'])
    training_generator = DataLoader(training_set, batch_size=config['batch_size'], shuffle=True, num_workers=4, drop_last=True)

    validation_set = DTI_Dataset(df_val_balanced, df_val_balanced.Label.values, config['max_drug_seq'], config['max_protein_seq'])
    validation_generator = DataLoader(validation_set, batch_size=config['batch_size'], shuffle=False, num_workers=4, drop_last=True)

    testing_set = DTI_Dataset(df_test_balanced, df_test_balanced.Label.values, config['max_drug_seq'], config['max_protein_seq'])
    testing_generator = DataLoader(testing_set, batch_size=config['batch_size'], shuffle=False, num_workers=4, drop_last=True)

    max_auc = 0
    model_max = copy.deepcopy(model)

    for epo in range(150):  # Epochs set to 100
        model.train()
        for i, (d, p, d_mask, p_mask, label) in enumerate(training_generator):
            score = model(d.long().cuda(), p.long().cuda(), d_mask.long().cuda(), p_mask.long().cuda())
            label = label.float().cuda()
            score = score[:label.shape[0]]

            loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).cuda())
            n = torch.squeeze(score)
            loss = loss_fct(n, label)
            loss_history.append(loss.item())

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        avg_train_loss = loss / len(training_generator)

        # Validation step
        with torch.set_grad_enabled(False):
            roc_auc, auprc, f1, val_loss, precision, recall, sensitivity, specificity, accuracy, thresold = test_with_optimal_threshold(validation_generator, model, pos_weight)


            val_auc_list.append(roc_auc)
            if roc_auc > max_auc:
                model_max = copy.deepcopy(model)
                max_auc = roc_auc

        # Early stopping check
        if early_stopping(val_auc_list, patience):
            print(f"Early stopping at epoch {epo + 1}")
            break

    # Testing with best model
    with torch.set_grad_enabled(False):
        roc_auc, auprc, f1, test_loss, precision, recall, sensitivity, specificity, accuracy, optimal_threshold= test_with_optimal_threshold(testing_generator, model_max, pos_weight)

    print('fine')
    # Load the DAVIS data from the provided file path
    file_path = '/mnt/research/Datta_Aniruddha/Students/Mondal_Madhurima/MolTrans/MY_DAVIS_DATA/davis.tab'

    # Read the data as a tab-separated file
    df_davis = pd.read_csv(file_path, sep='\t')

    # Keep both 'Target ID' and 'Target Sequence' columns and drop duplicates to get unique targets
    df_mine = df_davis[['ID2', 'X2']].drop_duplicates()

    # Show the first few rows of the dataframe
    print(df_mine.head())

    # After testing, get unique protein sequences from the test dataset
    unique_proteins = df_mine #['Target Sequence'].unique().tolist()
    print(f"number of unique proteins: {len(unique_proteins)}")
    # Curcumin SMILES string
    curcumin_smiles = "COC1=C(C=CC(=C1)/C=C/C(=O)CC(=O)/C=C/C2=CC(=C(C=C2)O)OC)O"

    # Predict DTIs for curcumin against unique proteins
    predict_curcumin_dti_probabilities(curcumin_smiles, model_max, unique_proteins, config['max_drug_seq'], config['max_protein_seq'])

    return model_max, loss_history, (roc_auc, auprc, f1, test_loss, precision, recall, sensitivity, specificity, accuracy, optimal_threshold)


if __name__ == '__main__':
    lr = 1.704e-5
    run_for_seeds([42], lr)
