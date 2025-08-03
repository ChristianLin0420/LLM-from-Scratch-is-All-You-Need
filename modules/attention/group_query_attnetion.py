import torch
import torch.nn as nn

from ..common.layer_norm import RMSNorm
from ..common.positional_embedding import apply_rope

class GroupQueryAttention(nn.Module):
    """
    GroupQueryAttention implements multi-head attention with grouped key-value heads (GQA).

    Args:
        config (dict): Configuration dictionary with the following keys:
            - "num_heads" (int): Number of attention heads.
            - "num_kv_groups" (int): Number of key-value groups.
            - "d_input" (int): Input feature dimension.
            - "head_dim" (int or None): Dimension per head. If None, computed as d_input // num_heads.
            - "dtype" (torch.dtype): Data type for the layers.
            - "q_proj_use_bias" (bool): Whether to use bias in query projection.
            - "k_proj_use_bias" (bool): Whether to use bias in key projection.
            - "v_proj_use_bias" (bool): Whether to use bias in value projection.
            - "o_proj_use_bias" (bool): Whether to use bias in output projection.
            - "use_qk_norm" (bool): Whether to use RMSNorm on queries and keys.
            - (Other RMSNorm config keys if use_qk_norm is True.)

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

        assert config["num_heads"] % config["num_kv_groups"] == 0, "num_heads must be divisible by num_kv_groups"

        self.num_heads = config["num_heads"]
        self.num_kv_groups = config["num_kv_groups"]
        self.group_size = self.num_heads // self.num_kv_groups

        if config["head_dim"] is None:
            assert config["d_input"] % self.num_heads == 0, "d_input must be divisible by num_heads"
            self.head_dim = config["d_input"] // self.num_heads
        else:
            self.head_dim = config["head_dim"]

        self.d_output = self.num_heads * self.head_dim

        self.q_proj = nn.Linear(config["d_input"], self.d_output, dtype = config["dtype"], bias = config["q_proj_use_bias"])
        self.k_proj = nn.Linear(config["d_input"], self.d_output, dtype = config["dtype"], bias = config["k_proj_use_bias"])
        self.v_proj = nn.Linear(config["d_input"], self.d_output, dtype = config["dtype"], bias = config["v_proj_use_bias"])

        self.output_proj = nn.Linear(self.d_output, config["d_input"], dtype = config["dtype"], bias = config["o_proj_use_bias"])

        if config["use_qk_norm"]:
            self.q_norm = RMSNorm(config)
            self.k_norm = RMSNorm(config)
        else:
            self.q_norm = None
            self.k_norm = None

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

        # Apply attention projection
        queries = self.q_proj(input_tensor)
        keys = self.k_proj(input_tensor)
        values = self.v_proj(input_tensor)

        # Reshape queries, keys, and values for group query attention
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim)
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Apply layer normalization if enabled
        if self.q_norm:
            queries = self.q_norm(queries)

        if self.k_norm:
            keys = self.k_norm(keys)

        # Apply RoPE
        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)

        # Expand K and V for group query attention
        keys = keys.repeat_interleave(self.num_kv_groups, dim = 1)
        values = values.repeat_interleave(self.num_kv_groups, dim = 1)

        # Attention Calculation
        attention_scores = queries @ keys.transpose(-2, -1) # attention_scores: (batch_size, seq_len, num_heads, num_kv_groups, seq_len)
        attention_scores = attention_scores.masked_fill(attention_mask, -torch.inf)
        attention_weights = torch.softmax(attention_scores / self.head_dim ** 0.5, dim = -1)

        context_states = (attention_weights @ values).transpose(1, 2).reshape(batch_size, seq_len, self.d_output)

        # Apply output projection
        output = self.output_proj(context_states)

        return output
        
            
        