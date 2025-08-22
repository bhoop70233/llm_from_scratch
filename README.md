# LLM From Scratch - Chapter 3: Attention Mechanisms

This repository contains the implementation of attention mechanisms from scratch, as part of the "LLM From Scratch" learning series.

## Overview

Chapter 3 focuses on implementing various attention mechanisms:

1. **Basic Attention**: Simple dot-product attention without trainable weights
2. **Self-Attention**: Attention with trainable query, key, and value weight matrices
3. **Causal Attention**: Self-attention with causal masking for autoregressive models
4. **Multi-Head Attention**: Attention with multiple heads for parallel processing

## Files

- `ch03.ipynb`: Main Jupyter notebook containing all implementations
- `setup_github.bat`: Script to help set up GitHub repository

## Key Implementations

### 1. Basic Attention
- Manual computation of attention scores using dot products
- Softmax normalization of attention weights
- Context vector computation

### 2. Self-Attention Classes
- `SelfAttention_v1`: Using `nn.Parameter` for weight matrices
- `SelfAttention_v2`: Using `nn.Linear` layers for better integration

### 3. Causal Attention
- `CausalAttention`: Implements causal masking to prevent looking at future tokens
- Uses upper triangular mask to enforce autoregressive property

### 4. Multi-Head Attention
- `MultiHeadAttentionWrapper`: Simple wrapper using multiple attention heads
- `MultiHeadAttention`: Efficient implementation with weight splitting
- Proper tensor reshaping and transposition for multi-head processing

## Requirements

- Python 3.x
- PyTorch 2.8.0+
- Jupyter Notebook

## Setup Instructions

1. Install Git (if not already installed):
   ```bash
   winget install --id Git.Git -e --source winget
   ```

2. Run the setup script:
   ```bash
   setup_github.bat
   ```

3. Follow the prompts to provide your GitHub repository URL

## Key Concepts Covered

- **Attention Scores**: Computing similarity between queries and keys
- **Attention Weights**: Softmax normalization of attention scores
- **Context Vectors**: Weighted combination of values using attention weights
- **Causal Masking**: Preventing attention to future tokens in autoregressive models
- **Multi-Head Processing**: Parallel attention computation with different weight matrices
- **Tensor Operations**: Proper reshaping and transposition for multi-head attention

## Recent Fixes

- Fixed `d_out` parameter to be divisible by `num_heads` in MultiHeadAttention
- Added missing `keys.transpose(1,2)` operation in MultiHeadAttention
- Corrected tensor shape handling for multi-head attention
- Fixed indentation issues in class implementations

## Usage

Open `ch03.ipynb` in Jupyter Notebook and run the cells sequentially to understand each attention mechanism implementation.

## License

This project is for educational purposes as part of the "LLM From Scratch" learning series.
