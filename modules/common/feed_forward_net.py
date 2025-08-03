import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardNet(nn.Module):
    """
    FeedForwardNet implements a feed-forward neural network block with two parallel linear layers
    followed by a gated activation and a final linear projection.

    Args:
        config (dict): Configuration dictionary with the following keys:
            - "embedding_dim" (int): Input and output feature dimension.
            - "hidden_dim" (int): Hidden layer dimension.
            - "dtype" (torch.dtype): Data type for the layers.
            - "fc_use_bias" (bool): Whether to use bias in the linear layers.

    Input shape:
        input_tensor: (batch_size, seq_len, embedding_dim)

    Output shape:
        output: (batch_size, seq_len, embedding_dim)
    """
    def __init__(self, config):
        super().__init__()

        self.fc1 = nn.Linear(config["embedding_dim"], config["hidden_dim"], dtype=config["dtype"], bias=config["fc_use_bias"])
        self.fc2 = nn.Linear(config["embedding_dim"], config["hidden_dim"], dtype=config["dtype"], bias=config["fc_use_bias"])
        self.fc3 = nn.Linear(config["hidden_dim"], config["embedding_dim"], dtype=config["dtype"], bias=config["fc_use_bias"])

    def forward(self, input_tensor):
        """
        Args:
            input_tensor (torch.Tensor): Input tensor of shape (batch_size, seq_len, embedding_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embedding_dim)
        """
        fc1_output = self.fc1(input_tensor)
        fc2_output = self.fc2(input_tensor)
        output = F.silu(fc1_output) * fc2_output
        return self.fc3(output)