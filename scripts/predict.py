import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
import numpy as np
from rdkit import Chem
from torch_geometric.data import Data, DataLoader

from src.model import EnhancedPromptLapFormer
from src.data_loader import collate_with_features
from src.utils import set_seed, mol_to_graph_data_with_edges, node_dim, EDGE_DIM

class Predictor:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.model = EnhancedPromptLapFormer(
            node_dim=node_dim,
            hidden_dim=256,
            num_heads=8,
            num_layers=3,
            num_prompts=5,
            prompt_dim=256,
            dropout=0.1,
            edge_dim=EDGE_DIM,
            descriptor_dim=11
        ).to(self.device)

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"Loaded model from {model_path}")
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")

    def predict_single(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None

        x, edge_index, edge_attr = mol_to_graph_data_with_edges(mol)

        data = Data(
            x=torch.tensor(x, dtype=torch.float, device=self.device),
            edge_index=torch.tensor(edge_index, dtype=torch.long, device=self.device),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float, device=self.device),
            descriptor=torch.zeros(11, dtype=torch.float, device=self.device),
            smiles=smiles
        )

        with torch.no_grad():
            logits = self.model(data, return_all_losses=False)
            prob = torch.softmax(logits, dim=1)[0, 1].item()
            pred = int(logits.argmax(dim=1).item())

        return pred, prob

    def predict_batch(self, smiles_list, batch_size=64):
        data_list = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                x = torch.zeros(1, node_dim, device=self.device)
                edge_index = torch.zeros(2, 0, dtype=torch.long, device=self.device)
                edge_attr = torch.zeros(0, EDGE_DIM, dtype=torch.float, device=self.device)
                desc = torch.zeros(11, dtype=torch.float, device=self.device)
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, descriptor=desc, smiles=smiles)
            else:
                x, edge_index, edge_attr = mol_to_graph_data_with_edges(mol)
                data = Data(
                    x=torch.tensor(x, dtype=torch.float, device=self.device),
                    edge_index=torch.tensor(edge_index, dtype=torch.long, device=self.device),
                    edge_attr=torch.tensor(edge_attr, dtype=torch.float, device=self.device),
                    descriptor=torch.zeros(11, dtype=torch.float, device=self.device),
                    smiles=smiles
                )
            data_list.append(data)

        loader = DataLoader(data_list, batch_size=batch_size, collate_fn=collate_with_features)

        predictions = []
        probabilities = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                logits = self.model(batch, return_all_losses=False)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds = logits.argmax(dim=1).cpu().numpy()

                predictions.extend(preds)
                probabilities.extend(probs)

        return predictions, probabilities

def main():
    set_seed(42)

    predictor = Predictor('best_model_enhanced_bace.pt')

    smiles_list = [
        'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',
        'CC1=CC=C(C=C1)C(C)C(=O)O',
        'CCC1=CC=CC=C1',
    ]

    for smiles in smiles_list:
        pred, prob = predictor.predict_single(smiles)
        print(f"SMILES: {smiles}")
        print(f"  Prediction: {'Active' if pred == 1 else 'Inactive'}")
        print(f"  Probability: {prob:.4f}")
        print()

    results_df = pd.DataFrame({
        'smiles': smiles_list,
        'prediction': predictions,
        'probability': probabilities
    })
    results_df.to_csv('predictions.csv', index=False)
    print("Predictions saved to 'predictions.csv'")

if __name__ == "__main__":
    main()
