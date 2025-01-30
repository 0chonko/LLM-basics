# Toy Dataset: We create a simple dataset with random sequences of a fixed length and embedding size.
# DataLoader: We use PyTorch's DataLoader to create batches of data.
# Self-Attention Module: We apply the self-attention module to the input sequences and print the output shape.

import torch
import torch.nn as nn
import torch.nn.functional as F

# Toy dataset
class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, seq_length, embed_size, num_samples):
        self.seq_length = seq_length
        self.embed_size = embed_size
        self.num_samples = num_samples
        self.data = torch.randn(num_samples, seq_length, embed_size)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

# Create dataset and dataloader
seq_length = 10
embed_size = 64
num_samples = 100
batch_size = 10

dataset = ToyDataset(seq_length, embed_size, num_samples)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize self-attention module
self_attention = SelfAttention(embed_size, heads=4)

# Example forward pass with toy dataset
for batch in dataloader:
    out = self_attention(batch, batch, batch, None)
    print("Output shape:", out.shape)
    break