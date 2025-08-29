import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.emb_dim = cfg["emb_dim"]
        self.n_heads = cfg["n_heads"]
        self.head_dim = self.emb_dim // self.n_heads

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(self.emb_dim, self.emb_dim, bias=cfg["qkv_bias"])
        self.k_proj = nn.Linear(self.emb_dim, self.emb_dim, bias=cfg["qkv_bias"])
        self.v_proj = nn.Linear(self.emb_dim, self.emb_dim, bias=cfg["qkv_bias"])

        # Output projection
        self.out_proj = nn.Linear(self.emb_dim, self.emb_dim)

        # Dropout
        self.dropout = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, mask=None):
        batch_size, seq_len, emb_dim = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)  # (batch_size, seq_len, emb_dim)
        k = self.k_proj(x)  # (batch_size, seq_len, emb_dim)
        v = self.v_proj(x)  # (batch_size, seq_len, emb_dim)

        # Reshape to separate heads
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Apply softmax
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, emb_dim
        )

        # Final projection
        output = self.out_proj(attn_output)

        return output
