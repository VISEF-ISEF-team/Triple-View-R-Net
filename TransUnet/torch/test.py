import ml_collections
import torch
from torch.nn.modules.utils import _pair
import torch.nn as nn

# Create a tensor of shape (2, 3, 4)
x = torch.randn(1, 196, 768)
print(x)

# Apply Layer Normalization along the last dimension (feature dimension)
layer_norm = nn.LayerNorm(768)
normalized_x = layer_norm(x)

num_attention_heads = 12
attention_head_size = 768 / num_attention_heads
new_x_shape = x.size()[:-1] + (num_attention_heads, int(attention_head_size))
x = x.view(*new_x_shape)

print(x.size())  # Output: torch.Size([2, 3, 4])