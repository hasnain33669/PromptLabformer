import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_bace_dataset, balanced_scaffold_split
from src.utils import set_seed

def test_data_loading():
    set_seed(42)

    data = load_bace_dataset('BACE.csv')
    assert data is not None
    assert 'smiles' in data.columns
    assert 'value' in data.columns
    assert len(data) > 0

    print(f"Data loading test passed: {len(data)} samples")

def test_scaffold_split():
    set_seed(42)

    data = load_bace_dataset('BACE.csv')
    train, val, test = balanced_scaffold_split(data, test_size=0.1, val_size=0.1)

    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0
    assert len(train) + len(val) + len(test) == len(data)

    print("Scaffold split test passed")

if __name__ == "__main__":
    test_data_loading()
    test_scaffold_split()
