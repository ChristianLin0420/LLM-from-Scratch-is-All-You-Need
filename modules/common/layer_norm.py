import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    RMSNorm implements Root Mean Square Layer Normalization.

    Args:
        config (dict): Configuration dictionary with the following keys:
            - "embedding_dim" (int): The dimension of the input features.
            - "epsilon" (float): A small value to avoid division by zero.
            - "qwen3_compatiable" (bool): Whether to use Qwen3 compatibility mode (casts to float32).
            - "rms_norm_use_bias" (bool): Whether to use a bias (shift) parameter.

    Input shape:
        hidden_states: (batch_size, seq_len, embedding_dim) or (seq_len, embedding_dim)

    Output shape:
        norm_hidden_states: Same shape as input (batch_size, seq_len, embedding_dim) or (seq_len, embedding_dim)
    """
    def __init__(self, config, is_layer_norm = False):
        super().__init__()
        self.epsilon = config["epsilon"]
        self.qwen3_compatiable = config["qwen3_compatiable"]

        if is_layer_norm:
            self.scale = nn.Parameter(torch.ones(config["embedding_dim"]))
            self.shift = nn.Parameter(torch.zeros(config["embedding_dim"])) if config["rms_norm_use_bias"] else None
        else:
            self.scale = nn.Parameter(torch.ones(config["head_dim"]))
            self.shift = nn.Parameter(torch.zeros(config["head_dim"])) if config["rms_norm_use_bias"] else None

    def forward(self, hidden_states):
        """
        Args:
            hidden_states (torch.Tensor): Input tensor of shape (batch_size, seq_len, embedding_dim) or (seq_len, embedding_dim)

        Returns:
            torch.Tensor: Normalized tensor of the same shape as input.
        """
        input_type = hidden_states.dtype

        if self.qwen3_compatiable:
            hidden_states = hidden_states.to(torch.float32)

        variance = hidden_states.pow(2).mean(dim = -1, keepdim = True)
        norm_hidden_states = hidden_states * torch.rsqrt(variance + self.epsilon)
        norm_hidden_states = norm_hidden_states * self.scale

        if self.shift is not None:
            norm_hidden_states = norm_hidden_states + self.shift

        return norm_hidden_states.to(input_type)