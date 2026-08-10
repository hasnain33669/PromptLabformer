import random
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from transformers import AutoTokenizer, AutoModel

ATOM_FEATURES = {
    'atomic_num': list(range(1, 119)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
    'is_aromatic': [0, 1]
}

BOND_FEATURES = {
    'bond_type': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'is_conjugated': [0, 1],
    'is_in_ring': [0, 1],
    'stereo': [0, 1, 2, 3]
}

EDGE_DIM = 12
node_dim = 145

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def one_hot_encoding(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def atom_features(atom):
    features = []
    features += one_hot_encoding(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num'])
    features += one_hot_encoding(atom.GetTotalDegree(), ATOM_FEATURES['degree'])
    features += one_hot_encoding(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge'])
    features += one_hot_encoding(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag'])
    features += one_hot_encoding(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs'])
    features += one_hot_encoding(int(atom.GetHybridization()), ATOM_FEATURES['hybridization'])
    features += one_hot_encoding(int(atom.GetIsAromatic()), ATOM_FEATURES['is_aromatic'])
    return np.array(features, dtype=np.float32)

def bond_features(bond):
    features = []
    features += one_hot_encoding(bond.GetBondType(), BOND_FEATURES['bond_type'])
    features += one_hot_encoding(int(bond.GetIsConjugated()), BOND_FEATURES['is_conjugated'])
    features += one_hot_encoding(int(bond.IsInRing()), BOND_FEATURES['is_in_ring'])
    features += one_hot_encoding(int(bond.GetStereo()), BOND_FEATURES['stereo'])
    assert len(features) == EDGE_DIM, f"Edge features should be {EDGE_DIM}, got {len(features)}"
    return np.array(features, dtype=np.float32)

def mol_to_graph_data_with_edges(mol):
    if mol is None:
        return np.zeros((1, 145), dtype=np.float32), np.zeros((2, 0), dtype=np.int64), np.zeros((0, EDGE_DIM), dtype=np.float32)

    atom_features_list = [atom_features(atom) for atom in mol.GetAtoms()]
    x = np.array(atom_features_list, dtype=np.float32)

    edge_index = []
    edge_attr = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        edge_index.append([i, j])
        edge_index.append([j, i])

        bond_feat = bond_features(bond)
        edge_attr.append(bond_feat)
        edge_attr.append(bond_feat)

    edge_index = np.array(edge_index, dtype=np.int64).T if edge_index else np.empty((2, 0), dtype=np.int64)
    edge_attr = np.array(edge_attr, dtype=np.float32) if edge_attr else np.empty((0, EDGE_DIM), dtype=np.float32)

    return x, edge_index, edge_attr

def calculate_rdkit_descriptors(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        descriptors = []

        descriptors.append(Descriptors.MolWt(mol))
        descriptors.append(Descriptors.MolLogP(mol))
        descriptors.append(Descriptors.TPSA(mol))
        descriptors.append(Lipinski.NumHDonors(mol))
        descriptors.append(Lipinski.NumHAcceptors(mol))
        descriptors.append(Descriptors.NumRotatableBonds(mol))
        descriptors.append(Descriptors.RingCount(mol))
        descriptors.append(Descriptors.NumAromaticRings(mol))
        descriptors.append(Descriptors.HeavyAtomCount(mol))
        descriptors.append(Descriptors.FractionCsp3(mol))
        descriptors.append(Descriptors.BalabanJ(mol))
        descriptors.append(Descriptors.BertzCT(mol))

        return np.array(descriptors, dtype=np.float32)
    except:
        return None

def precompute_descriptors(data_df):
    descriptors_list = []
    for smiles in data_df['smiles']:
        desc = calculate_rdkit_descriptors(smiles)
        if desc is not None:
            descriptors_list.append(desc)
        else:
            descriptors_list.append(np.zeros(12, dtype=np.float32))
    return np.array(descriptors_list, dtype=np.float32)

class LanguageEncoder:
    def __init__(self, hidden_size=256, device='cpu'):
        self.device = device
        self.hidden_size = hidden_size
        self.model_loaded = False

        try:
            print("Loading ChemBERTa...")
            self.tokenizer = AutoTokenizer.from_pretrained('DeepChem/ChemBERTa-77M-MTR')
            self.model = AutoModel.from_pretrained('DeepChem/ChemBERTa-77M-MTR')
            for param in self.model.parameters():
                param.requires_grad = False
            self.model = self.model.to(device)
            self.model.eval()
            self.model_loaded = True
            self.actual_hidden_size = self.model.config.hidden_size
            print(f"Loaded ChemBERTa (hidden_size: {self.actual_hidden_size})")

            if self.actual_hidden_size != self.hidden_size:
                self.token_projection = nn.Linear(self.actual_hidden_size, self.hidden_size).to(device)
        except Exception as e:
            print(f"Could not load ChemBERTa: {e}")
            self.model_loaded = False

    def precompute_embeddings(self, smiles_list, batch_size=32):
        if not self.model_loaded:
            return torch.randn(len(smiles_list), self.hidden_size, device=self.device)

        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(smiles_list), batch_size):
                batch = smiles_list[i:i+batch_size]
                try:
                    tokenized = self.tokenizer(batch, padding=True, return_tensors='pt', max_length=512, truncation=True)
                    input_ids = tokenized['input_ids'].to(self.device)
                    attention_mask = tokenized['attention_mask'].to(self.device)

                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    token_embeddings = outputs.last_hidden_state

                    mask = attention_mask.unsqueeze(-1).float()
                    token_embeddings = token_embeddings * mask
                    token_mean = token_embeddings.sum(dim=1) / mask.sum(dim=1)

                    if hasattr(self, 'token_projection'):
                        token_mean = self.token_projection(token_mean)

                    all_embeddings.append(token_mean.cpu())
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    all_embeddings.append(torch.randn(len(batch), self.hidden_size))

        return torch.cat(all_embeddings, dim=0).to(self.device)
