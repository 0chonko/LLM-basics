# This is a core component of the transformer architecture. It computes the 
# attention scores as the dot product of the query and key vectors, scaled by 
# the square root of the key vectors' dimensionality. This scaling helps to 
# stabilize the gradients during training.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Scaled Dot-Product Attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, query, key, value, mask=None):
        # Query, Key, Value dimensions: (batch_size, seq_length, embed_size)
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention = F.softmax(scores, dim=-1)
        
        # Compute weighted sum of values
        output = torch.matmul(attention, value)
        return output, attention

# Example usage
if __name__ == "__main__":
    # Hyperparameters
    embed_size = 64
    seq_length = 10
    batch_size = 2

    # Randomly initialize query, key, and value tensors
    query = torch.randn(batch_size, seq_length, embed_size)
    key = torch.randn(batch_size, seq_length, embed_size)
    value = torch.randn(batch_size, seq_length, embed_size)

    # Initialize attention mechanism
    attention = ScaledDotProductAttention()

    # Forward pass
    output, attention_weights = attention(query, key, value)

    print("Output shape:", output.shape)
    print("Attention weights shape:", attention_weights.shape)