# PromptLapFormer

PromptLapFormer is a prompt-guided molecular graph learning framework for molecular property prediction. The framework combines molecular graph representations with semantic information from a pretrained molecular language model, adaptive weighted Laplacian graph refinement, edge-aware Graph Transformer encoding, and cross-modal feature fusion.

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.3+
- RDKit
- PyTorch Geometric
- Transformers
- scikit-learn
- NumPy
- Pandas

### Create Conda Environment

```bash
conda create -n promptlapformer python=3.9
conda activate promptlapformer


Model Configuration
model = EnhancedPromptLapFormer(
    node_dim=145,
    hidden_dim=256,
    num_heads=8,
    num_layers=3,
    num_prompts=5,
    prompt_dim=256,
    dropout=0.1,
    edge_dim=12,
    descriptor_dim=11
)
Structure
PromptLapFormer/
├── README.md
├── configs/
│   └── default_config.yaml
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── model.py
│   ├── trainer.py
│   └── utils.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
└── tests/
    ├── test_model.py
    └── test_data_loader.py
