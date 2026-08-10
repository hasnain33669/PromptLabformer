import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from src.data_loader import load_bace_dataset, balanced_scaffold_split, create_datasets
from src.model import EnhancedPromptLapFormer
from src.trainer import EnhancedPromptLapFormerTrainer
from src.utils import set_seed, precompute_descriptors, LanguageEncoder, node_dim, EDGE_DIM

def main():
    config_path = 'configs/default_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    set_seed(config['seed'])

    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading BACE Dataset")
    bace_data = load_bace_dataset('BACE.csv')

    print("Performing Balanced Scaffold Split")
    train_data, val_data, test_data = balanced_scaffold_split(
        bace_data,
        test_size=config['data']['test_size'],
        val_size=config['data']['val_size']
    )

    print("Pre-computing RDKit descriptors...")
    train_descriptors = precompute_descriptors(train_data)
    val_descriptors = precompute_descriptors(val_data)
    test_descriptors = precompute_descriptors(test_data)
    descriptor_dim = train_descriptors.shape[1]

    all_smiles = list(train_data['smiles']) + list(val_data['smiles']) + list(test_data['smiles'])
    print(f"Pre-computing embeddings for {len(all_smiles)} molecules...")
    language_encoder = LanguageEncoder(hidden_size=256, device=device)
    precomputed_embeddings = language_encoder.precompute_embeddings(all_smiles)
    torch.save(precomputed_embeddings, 'embeddings_bace.pt')

    train_dataset, val_dataset, test_dataset = create_datasets(
        train_data, val_data, test_data, device,
        precomputed_embeddings, train_descriptors, val_descriptors, test_descriptors
    )

    model = EnhancedPromptLapFormer(
        node_dim=node_dim,
        hidden_dim=config['model']['hidden_dim'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        num_prompts=config['model']['num_prompts'],
        prompt_dim=config['model']['prompt_dim'],
        dropout=config['model']['dropout'],
        edge_dim=EDGE_DIM,
        descriptor_dim=descriptor_dim
    ).to(device)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = EnhancedPromptLapFormerTrainer(
        model,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    results = trainer.train(
        train_dataset, val_dataset, test_dataset,
        epochs=config['training']['epochs']
    )

if __name__ == "__main__":
    main()
