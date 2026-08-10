import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import EnhancedPromptLapFormer
from src.utils import node_dim, EDGE_DIM

def test_model_initialization():
    model = EnhancedPromptLapFormer(
        node_dim=node_dim,
        hidden_dim=256,
        num_heads=8,
        num_layers=3,
        num_prompts=5,
        prompt_dim=256,
        dropout=0.1,
        edge_dim=EDGE_DIM,
        descriptor_dim=11
    )

    assert model is not None
    assert hasattr(model, 'node_projection')
    assert hasattr(model, 'graph_transformer')
    assert hasattr(model, 'prediction_head')

    print("Model initialization test passed")

def test_forward_pass():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = EnhancedPromptLapFormer(
        node_dim=node_dim,
        hidden_dim=256,
        num_heads=8,
        num_layers=3,
        num_prompts=5,
        prompt_dim=256,
        dropout=0.1,
        edge_dim=EDGE_DIM,
        descriptor_dim=11
    ).to(device)

    x = torch.randn(10, node_dim, device=device)
    edge_index = torch.randint(0, 10, (2, 20), device=device)
    edge_attr = torch.randn(20, EDGE_DIM, device=device)
    descriptors = torch.randn(1, 11, device=device)

    from torch_geometric.data import Data
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, descriptor=descriptors)

    with torch.no_grad():
        logits = model(data, return_all_losses=False)

    assert logits.shape == (1, 2)
    print("Forward pass test passed")

if __name__ == "__main__":
    test_model_initialization()
    test_forward_pass()
