import torch
import torch.nn as nn

from ..attention.multi_head_attention import MultiHeadAttention
from ..common.feed_forward_net import FeedForwardNet
from ..common.layer_norm import RMSNorm

class Qwen1Block(nn.Module):
    def __init__(self, config):
        """
        Qwen1Block implements a single block of the Qwen1 model.
        This is simpler than Qwen3Block with standard multi-head attention.

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
        """
        super().__init__()

        self.attention = MultiHeadAttention(config)
        self.ffn = FeedForwardNet(config)
        self.norm1 = RMSNorm(config, is_layer_norm=True)
        self.norm2 = RMSNorm(config, is_layer_norm=True)

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