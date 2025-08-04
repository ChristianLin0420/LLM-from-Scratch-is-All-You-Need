import torch
import torch.nn as nn

from ..attention.group_query_attnetion import GroupQueryAttention
from ..common.feed_forward_net import FeedForwardNet
from ..common.layer_norm import RMSNorm

class Qwen3Block(nn.Module):
    def __init__(self, config):
        """
        Qwen3Block implements a single block of the Qwen3 model.

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
        """
        super().__init__()

        self.attention = GroupQueryAttention(config)
        self.ffn = FeedForwardNet(config)
        self.norm1 = RMSNorm(config, is_layer_norm = True)
        self.norm2 = RMSNorm(config, is_layer_norm = True)

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
        # First Normalization
        hidden_states = self.norm1(input_tensor)

        # Attention
        attention_output = self.attention(hidden_states, attention_mask, cos, sin)

        # Add & Residual
        hidden_states = input_tensor + attention_output

        # Second Normalization
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)

        # Feed-Forward Network
        ffn_output = self.ffn(hidden_states)

        # Add & Residual
        hidden_states = residual + ffn_output

        return hidden_states