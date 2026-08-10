# PromptLabformer

A novel framework combining prompt learning and lab-based transformers for molecular property prediction.

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Performance](#performance)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

## Overview

PromptLabformer is an advanced deep learning framework designed for molecular property prediction, specifically optimized for BACE (Beta-Secretase) inhibitor activity prediction. The model leverages prompt learning techniques combined with transformer architectures to achieve state-of-the-art performance.

## Key Features

- **Prompt Learning Integration**: Utilizes learnable prompts for improved molecular representation
- **Transformer Architecture**: Advanced self-attention mechanisms for molecular graphs
- **Multi-Task Learning**: Supports multiple prediction tasks simultaneously
- **Efficient Training**: Optimized for GPU acceleration with mixed precision support
- **Comprehensive Metrics**: Includes accuracy, AUC, F1, precision, and recall

## Performance

Our model achieves the following performance on the BACE benchmark dataset:

| Metric | Value |
|--------|-------|
| Test Loss | 0.4655 |
| Test Accuracy | 76.10% |
| Test AUC | 91.04% |
| Test Precision | 90.20% |
| Test Recall | 58.23% |
| Test F1 Score | 70.77% |

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.10+
- CUDA 11.3+ (optional, for GPU acceleration)

### Install from source

```bash
git clone https://github.com/yourusername/PromptLabformer.git
cd PromptLabformer
pip install -e .
