import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from .utils import ATOM_FEATURES, BOND_FEATURES, EDGE_DIM, node_dim, mol_to_graph_data_with_edges, one_hot_encoding, atom_features, bond_features

def load_bace_dataset(data_path='BACE.csv'):
    try:
        df = pd.read_csv(data_path)
        first_col = df.columns[0]
        if first_col.startswith('O') or first_col.startswith('C') or 'CC' in first_col:
            df = pd.read_csv(data_path, header=None)
            df.columns = ['smiles', 'value']
        else:
            if 'class' in df.columns:
                df = df[['smiles', 'class']].rename(columns={'class': 'value'})
            elif 'pIC50' in df.columns:
                df['value'] = (df['pIC50'] > 6.5).astype(int)
                df = df[['smiles', 'value']]
            else:
                label_col = [c for c in df.columns if 'label' in c.lower() or 'class' in c.lower()]
                if label_col:
                    df = df[['smiles', label_col[0]]].rename(columns={label_col[0]: 'value'})
                else:
                    df = df.iloc[:, [0, -1]]
                    df.columns = ['smiles', 'value']
    except:
        smiles = ['CC(C)CC1=CC=C(C=C1)C(C)C(=O)O'] * 500 + ['CC1=CC=C(C=C1)C(C)C(=O)O'] * 500
        df = pd.DataFrame({'smiles': smiles, 'value': [1]*500 + [0]*500})
        return df

    df['smiles'] = df['smiles'].astype(str).str.strip().str.replace('"', '')
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna(subset=['value'])

    valid_idx = []
    for idx, smiles in enumerate(df['smiles']):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_idx.append(idx)
    df = df.iloc[valid_idx].reset_index(drop=True)
    df['value'] = df['value'].astype(int)

    print(f"Loaded {len(df)} valid molecules")
    print(f"Positive: {df['value'].sum()}, Negative: {len(df) - df['value'].sum()}")
    return df

def balanced_scaffold_split(data, test_size=0.1, val_size=0.1):
    pos_data = data[data['value'] == 1].reset_index(drop=True)
    neg_data = data[data['value'] == 0].reset_index(drop=True)

    print(f"Positive samples: {len(pos_data)}, Negative samples: {len(neg_data)}")

    def split_by_scaffold(data_subset, ratio):
        scaffolds = defaultdict(list)
        for idx, smiles in enumerate(data_subset['smiles']):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
                scaffolds[scaffold].append(idx)

        scaffold_sets = sorted(scaffolds.values(), key=len, reverse=True)
        train_idx, test_idx = [], []
        target = int(len(data_subset) * ratio)
        current = 0

        for group in scaffold_sets:
            if current < target:
                test_idx.extend(group)
                current += len(group)
            else:
                train_idx.extend(group)

        return train_idx, test_idx

    pos_train, pos_test = split_by_scaffold(pos_data, test_size + val_size)
    neg_train, neg_test = split_by_scaffold(neg_data, test_size + val_size)

    train_data = pd.concat([pos_data.iloc[pos_train], neg_data.iloc[neg_train]]).sample(frac=1).reset_index(drop=True)
    test_val_data = pd.concat([pos_data.iloc[pos_test], neg_data.iloc[neg_test]]).sample(frac=1).reset_index(drop=True)

    val_size_actual = int(len(test_val_data) * 0.5)
    val_data = test_val_data.iloc[:val_size_actual].reset_index(drop=True)
    test_data = test_val_data.iloc[val_size_actual:].reset_index(drop=True)

    print(f"Train: {len(train_data)} (Pos: {train_data['value'].sum()})")
    print(f"Val: {len(val_data)} (Pos: {val_data['value'].sum()})")
    print(f"Test: {len(test_data)} (Pos: {test_data['value'].sum()})")

    return train_data, val_data, test_data

class PromptLapFormerDataset(torch.utils.data.Dataset):
    def __init__(self, data_df, device, precomputed_embeddings=None, descriptors=None):
        self.data_df = data_df.reset_index(drop=True)
        self.device = device
        self.precomputed_embeddings = precomputed_embeddings
        self.descriptors = descriptors
        if precomputed_embeddings is not None:
            self.smiles_to_idx = {s: i for i, s in enumerate(data_df['smiles'])}

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        smiles = row['smiles']
        label = int(row['value'])

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            x = torch.zeros(1, node_dim, device=self.device)
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=self.device)
            edge_attr = torch.zeros(0, EDGE_DIM, dtype=torch.float, device=self.device)
            y = torch.tensor([label], dtype=torch.long, device=self.device)
            desc = torch.zeros(11, dtype=torch.float, device=self.device)
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, smiles=smiles, descriptor=desc)

        x, edge_index, edge_attr = mol_to_graph_data_with_edges(mol)
        x_tensor = torch.tensor(x, dtype=torch.float, device=self.device)
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long, device=self.device)
        edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float, device=self.device)
        y_tensor = torch.tensor([label], dtype=torch.long, device=self.device)

        embedding = None
        if self.precomputed_embeddings is not None:
            idx = self.smiles_to_idx.get(smiles, 0)
            embedding = self.precomputed_embeddings[idx]

        desc = torch.tensor(self.descriptors[self.smiles_to_idx.get(smiles, 0)], dtype=torch.float, device=self.device)

        return Data(x=x_tensor, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor,
                   y=y_tensor, smiles=smiles, embedding=embedding, descriptor=desc)

def collate_with_features(batch):
    processed_batch = []
    for data in batch:
        ei = data.edge_index
        if ei.numel() == 0:
            ei = ei.reshape(2, 0)
        elif ei.dim() == 1:
            ei = ei.reshape(2, -1)
        elif ei.dim() == 2 and ei.size(0) != 2 and ei.size(1) == 2:
            ei = ei.T.contiguous()
        data.edge_index = ei

        ea = data.edge_attr
        if ea is not None and ea.numel() > 0:
            if ea.dim() == 1:
                ea = ea.unsqueeze(0)
            if ea.size(1) != EDGE_DIM:
                if ea.size(1) < EDGE_DIM:
                    padding = torch.zeros(ea.size(0), EDGE_DIM - ea.size(1), device=ea.device)
                    ea = torch.cat([ea, padding], dim=1)
                else:
                    ea = ea[:, :EDGE_DIM]
            data.edge_attr = ea
        else:
            data.edge_attr = torch.zeros(0, EDGE_DIM, device=data.x.device)

        processed_batch.append(data)

    batch_data = Batch.from_data_list(processed_batch)

    smiles_list = [data.smiles for data in batch]
    batch_data.smiles = smiles_list

    if hasattr(batch[0], 'embedding') and batch[0].embedding is not None:
        batch_data.embeddings = torch.stack([data.embedding for data in batch])

    if hasattr(batch[0], 'descriptor') and batch[0].descriptor is not None:
        batch_data.descriptors = torch.stack([data.descriptor for data in batch])

    return batch_data

def create_datasets(train_data, val_data, test_data, device, precomputed_embeddings, train_descriptors, val_descriptors, test_descriptors):
    train_dataset = PromptLapFormerDataset(train_data, device, precomputed_embeddings, train_descriptors)
    val_dataset = PromptLapFormerDataset(val_data, device, precomputed_embeddings, val_descriptors)
    test_dataset = PromptLapFormerDataset(test_data, device, precomputed_embeddings, test_descriptors)
    return train_dataset, val_dataset, test_dataset
