import torch

def compute_rope_parameters(head_dim, theta_base=10_000, context_len=4096, dtype=torch.float32):
    """
    Compute RoPE (Rotary Positional Embedding) parameters.

    Args:
        head_dim (int): The dimension of each attention head (must be even).
        theta_base (float, optional): The base for computing inverse frequency. Default: 10_000.
        context_len (int, optional): The maximum sequence length (number of positions). Default: 4096.
        dtype (torch.dtype, optional): Data type for the computation. Default: torch.float32.

    Returns:
        cos (torch.Tensor): Cosine values for RoPE, shape (context_len, head_dim)
        sin (torch.Tensor): Sine values for RoPE, shape (context_len, head_dim)
    """

    assert head_dim % 2 == 0, "head_dim must be divisible by 2"

    # Compute the inverse frequency
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim)) # inv_freq: (head_dim // 2,)

    # Generate the position indices
    position_ids = torch.arange(context_len, dtype=dtype) # position_ids: (context_len,)

    # Compute the angles
    angles = position_ids[:, None] * inv_freq[None, :] # angles: (context_len, head_dim // 2)

    # Expand the angles to match the head_dim
    angles = torch.cat([angles, angles], dim=-1) # angles: (context_len, head_dim)

    # Precompute the sine and cosine values
    sin = torch.sin(angles) # sin: (context_len, head_dim)
    cos = torch.cos(angles) # cos: (context_len, head_dim)

    return cos, sin

def apply_rope(input_tensor, cos, sin):
    """
    Apply RoPE (Rotary Positional Embedding) to the input tensor.

    Args:
        input_tensor (torch.Tensor): The input tensor of shape (batch_size, seq_len, head_dim)
        cos (torch.Tensor): The cosine values for RoPE, shape (context_len, head_dim)
        sin (torch.Tensor): The sine values for RoPE, shape (context_len, head_dim)
    """
    
    batch_size, seq_len, head_dim = input_tensor.shape

    assert head_dim % 2 == 0, "head_dim must be divisible by 2"

    # Split input_tensor into two halves
    x1, x2 = input_tensor.split(head_dim // 2, dim = -1) # x1: (batch_size, seq_len, head_dim // 2), x2: (batch_size, seq_len, head_dim // 2)

    # Adjust cos and sin for the actual sequence length
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0) # cos: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0) # sin: (1, 1, seq_len, head_dim)

    # Apply RoPE to the input tensor
    rotated = torch.cat((-x2, x1), dim = -1)
    input_rotated = input_tensor * cos + rotated * sin

    return input_rotated.to(input_tensor.dtype)