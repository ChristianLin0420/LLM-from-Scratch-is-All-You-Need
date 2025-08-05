import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.positional_embedding import apply_rope


class MultiHeadAttention(nn.Module):
    """
    MultiHeadAttention implements standard multi-head attention mechanism (MHA).
    This is simpler than GroupQueryAttention and is used for Qwen1.

    Args:
        config (dict): Configuration dictionary with the following keys:
            - "num_heads" (int): Number of attention heads.
            - "d_input" (int): Input feature dimension.
            - "head_dim" (int or None): Dimension per head. If None, computed as d_input // num_heads.
            - "dtype" (torch.dtype): Data type for the layers.
            - "q_proj_use_bias" (bool): Whether to use bias in query projection.
            - "k_proj_use_bias" (bool): Whether to use bias in key projection.
            - "v_proj_use_bias" (bool): Whether to use bias in value projection.
            - "o_proj_use_bias" (bool): Whether to use bias in output projection.

    Input shape:
        input_tensor: (batch_size, seq_len, d_input)
        attention_mask: (batch_size, 1, 1, seq_len) or broadcastable mask for attention
        cos: (context_len, head_dim) - RoPE cosine values
        sin: (context_len, head_dim) - RoPE sine values

    Output shape:
        output: (batch_size, seq_len, d_input)
    """
    def __init__(self, config):
        super().__init__()

        self.num_heads = config["num_heads"]
        
        if config["head_dim"] is None:
            assert config["d_input"] % self.num_heads == 0, "d_input must be divisible by num_heads"
            self.head_dim = config["d_input"] // self.num_heads
        else:
            self.head_dim = config["head_dim"]

        self.d_output = self.num_heads * self.head_dim

        # Standard MHA: same number of heads for Q, K, V
        self.q_proj = nn.Linear(config["d_input"], self.d_output, dtype=config["dtype"], bias=config["q_proj_use_bias"])
        self.k_proj = nn.Linear(config["d_input"], self.d_output, dtype=config["dtype"], bias=config["k_proj_use_bias"])
        self.v_proj = nn.Linear(config["d_input"], self.d_output, dtype=config["dtype"], bias=config["v_proj_use_bias"])

        self.output_proj = nn.Linear(self.d_output, config["d_input"], dtype=config["dtype"], bias=config["o_proj_use_bias"])

    def forward(self, input_tensor, attention_mask, cos, sin):
        """
        Args:
            input_tensor (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_input)
            attention_mask (torch.Tensor): Attention mask, shape (batch_size, 1, 1, seq_len) or broadcastable
            cos (torch.Tensor): RoPE cosine values, shape (context_len, head_dim)
            sin (torch.Tensor): RoPE sine values, shape (context_len, head_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_input)
        """
        batch_size, seq_len, d_input = input_tensor.shape

        # Project to Q, K, V
        q = self.q_proj(input_tensor)  # (batch_size, seq_len, d_output)
        k = self.k_proj(input_tensor)  # (batch_size, seq_len, d_output)
        v = self.v_proj(input_tensor)  # (batch_size, seq_len, d_output)

        # Reshape for multi-head attention (but don't transpose yet for RoPE)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)  # (batch_size, seq_len, num_heads, head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)  # (batch_size, seq_len, num_heads, head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)  # (batch_size, seq_len, num_heads, head_dim)

        # Apply RoPE (works on last dimension, handles multi-head case)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        
        # Now transpose for attention computation
        q = q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim) 
        v = v.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)

        # Compute attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply attention mask
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask, float('-inf'))

        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply attention to values
        attention_output = torch.matmul(attention_weights, v)  # (batch_size, num_heads, seq_len, head_dim)

        # Reshape and concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_output)

        # Final projection
        output = self.output_proj(attention_output)

        return output