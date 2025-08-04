import torch
import torch.nn as nn

from ..block.qwen3_block import Qwen3Block
from ..common.layer_norm import RMSNorm
from ..common.positional_embedding import compute_rope_parameters


class Qwen3(nn.Module):
    """
    Args:
        config (dict): Configuration dictionary with the following keys:
            - "vocab_size" (int): Size of the vocabulary.
            - "embedding_dim" (int): Dimension of the token embeddings.
            - "dtype" (torch.dtype): Data type for the layers.
            - "num_layers" (int): Number of transformer blocks.
            - "num_heads" (int): Number of attention heads.
            - "head_dim" (int or None): Dimension per head. If None, computed as embedding_dim // num_heads.
            - (Other keys required by Qwen3Block and RMSNorm.)

    Input:
        input_ids (torch.Tensor): Input tensor of shape (batch_size, seq_len), containing token ids.

    Output:
        logits (torch.Tensor): Output tensor of shape (batch_size, seq_len, vocab_size), containing the predicted logits for each token position.
    """
    def __init__(self, config):
        super().__init__()

        self.token_embedding = nn.Embedding(config["vocab_size"], config["embedding_dim"], dtype = config["dtype"])

        self.blocks = nn.ModuleList([Qwen3Block(config) for _ in range(config["num_layers"])])

        self.norm = RMSNorm(config, is_layer_norm = True)

        self.lm_head = nn.Linear(config["embedding_dim"], config["vocab_size"], dtype = config["dtype"])

        # Reusable Utilities
        if config["head_dim"] is None:
            self.head_dim = config["embedding_dim"] // config["num_heads"]
        else:
            self.head_dim = config["head_dim"]

        self.cos, self.sin = compute_rope_parameters(self.head_dim, dtype = config["dtype"])

        if not hasattr(self, "cos"):
            self.register_buffer("cos", self.cos, persistent = False)
        if not hasattr(self, "sin"):
            self.register_buffer("sin", self.sin, persistent = False)

    def forward(self, input_ids):
        """
        Forward pass of the Qwen3 model.

        Args:
            input_ids (torch.Tensor): Input tensor of shape (batch_size, seq_len), containing token ids.

        Returns:
            torch.Tensor: Logits of shape (batch_size, seq_len, vocab_size).
        """
        batch_size, seq_len = input_ids.shape

        # Token Embedding
        x = self.token_embedding(input_ids)

        # Create Attention Mask
        attention_mask = torch.triu(torch.ones(seq_len, seq_len, dtype = torch.bool, device = input_ids.device), diagonal = 1)

        # Block Forward Pass
        for block in self.blocks:
            x = block(x, attention_mask, self.cos, self.sin)

        # Final Layer Normalization
        x = self.norm(x)

        # Language Model Head
        logits = self.lm_head(x)

        return logits
        
        
        