import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from torch_geometric.data import DataLoader

from src.data_loader import load_bace_dataset, balanced_scaffold_split, create_datasets, collate_with_features
from src.model import EnhancedPromptLapFormer
from src.trainer import EnhancedPromptLapFormerTester
from src.utils import set_seed, precompute_descriptors, LanguageEncoder, node_dim, EDGE_DIM

def main():
    set_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading BACE Dataset")
    bace_data = load_bace_dataset('BACE.csv')

    print("Performing Balanced Scaffold Split")
    train_data, val_data, test_data = balanced_scaffold_split(
        bace_data, test_size=0.1, val_size=0.1
    )

    print("Pre-computing RDKit descriptors...")
    train_descriptors = precompute_descriptors(train_data)
    val_descriptors = precompute_descriptors(val_data)
    test_descriptors = precompute_descriptors(test_data)
    descriptor_dim = train_descriptors.shape[1]

    all_smiles = list(train_data['smiles']) + list(val_data['smiles']) + list(test_data['smiles'])
    language_encoder = LanguageEncoder(hidden_size=256, device=device)
    precomputed_embeddings = language_encoder.precompute_embeddings(all_smiles)

    train_dataset, val_dataset, test_dataset = create_datasets(
        train_data, val_data, test_data, device,
        precomputed_embeddings, train_descriptors, val_descriptors, test_descriptors
    )

    model = EnhancedPromptLapFormer(
        node_dim=node_dim,
        hidden_dim=256,
        num_heads=8,
        num_layers=3,
        num_prompts=5,
        prompt_dim=256,
        dropout=0.1,
        edge_dim=EDGE_DIM,
        descriptor_dim=descriptor_dim
    ).to(device)

    if os.path.exists('best_model_enhanced_bace.pt'):
        model.load_state_dict(torch.load('best_model_enhanced_bace.pt', map_location=device))
        print("Loaded best model")
    else:
        print("Best model not found")

    tester = EnhancedPromptLapFormerTester(model)
    test_metrics = tester.test(test_dataset)

    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    print(f"Test Loss: {test_metrics['loss']:.6f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.6f}")
    print(f"Test AUC-ROC: {test_metrics['auc_roc']:.6f}")
    print(f"Test Precision: {test_metrics['precision']:.6f}")
    print(f"Test Recall: {test_metrics['recall']:.6f}")
    print(f"Test F1: {test_metrics['f1']:.6f}")

    cm = confusion_matrix(test_metrics['y_true'], test_metrics['y_pred'])
    print("\nConfusion Matrix:")
    print(cm)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix - BACE')
    plt.tight_layout()
    plt.savefig('confusion_matrix_bace.png', dpi=300)
    print("Saved confusion matrix to 'confusion_matrix_bace.png'")

    fpr, tpr, _ = roc_curve(test_metrics['y_true'], test_metrics['y_probs'])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - BACE')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('roc_curve_bace.png', dpi=300)
    print("Saved ROC curve to 'roc_curve_bace.png'")

if __name__ == "__main__":
    main()
