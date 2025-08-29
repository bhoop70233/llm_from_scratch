#!/usr/bin/env python3
"""
Test script to verify that MultiHeadAttention can be imported correctly.
"""

import torch
import torch.nn as nn

# Try importing from the local ch03.py file
try:
    from ch03 import MultiHeadAttention
    print("✓ Successfully imported MultiHeadAttention from ch03.py")

    # Test creating an instance
    config = {
        "emb_dim": 768,
        "context_length": 1024,
        "drop_rate": 0.1,
        "n_heads": 12,
        "qkv_bias": False
    }

    mha = MultiHeadAttention(
        d_in=config["emb_dim"],
        d_out=config["emb_dim"],  # d_out should be the same as d_in for proper dimensions
        context_length=config["context_length"],
        dropout=config["drop_rate"],
        num_heads=config["n_heads"],
        qkv_bias=config["qkv_bias"]
    )

    print("✓ Successfully created MultiHeadAttention instance")

    # Test forward pass
    batch_size, seq_len, emb_dim = 2, 4, 768
    x = torch.randn(batch_size, seq_len, emb_dim)
    output = mha(x)
    print(f"✓ Forward pass successful. Input shape: {x.shape}, Output shape: {output.shape}")

except ImportError as e:
    print(f"✗ Import failed: {e}")
except Exception as e:
    print(f"✗ Error: {e}")

print("\nYou can now use: from ch03 import MultiHeadAttention")
