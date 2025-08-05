# Adding New Models

## Overview

This guide explains how to add support for new model architectures to the LLM from Scratch framework.

## Step-by-Step Process

### 1. Create Model Configuration

Create a YAML configuration file in `configs/[model_name]/`:

```yaml
# configs/new_model/new_model_7B.yaml
vocab_size: 50000
context_length: 4096
embedding_dim: 4096
num_heads: 32
num_layers: 32
hidden_dim: 11008

# Model-specific parameters
custom_param1: value1
custom_param2: value2

# Standard parameters
dtype: bfloat16
```

### 2. Implement Attention Mechanism

Create attention implementation in `modules/attention/`:

```python
# modules/attention/new_model_attention.py
import torch
import torch.nn as nn

class NewModelAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Initialize attention components
        pass
    
    def forward(self, input_tensor, attention_mask, cos, sin):
        # Implement attention forward pass
        pass
```

### 3. Create Model Block

Implement the transformer block in `modules/block/`:

```python
# modules/block/new_model_block.py
import torch
import torch.nn as nn

from ..attention.new_model_attention import NewModelAttention
from ..common.feed_forward_net import FeedForwardNet
from ..common.layer_norm import RMSNorm

class NewModelBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.attention = NewModelAttention(config)
        self.ffn = FeedForwardNet(config)
        self.norm1 = RMSNorm(config, is_layer_norm=True)
        self.norm2 = RMSNorm(config, is_layer_norm=True)

    def forward(self, input_tensor, attention_mask, cos, sin):
        # Pre-norm architecture
        hidden_states = self.norm1(input_tensor)
        attention_output = self.attention(hidden_states, attention_mask, cos, sin)
        hidden_states = input_tensor + attention_output

        # Feed-forward
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        ffn_output = self.ffn(hidden_states)
        hidden_states = residual + ffn_output

        return hidden_states
```

### 4. Implement Main Model Class

Create the main model in `modules/llm/`:

```python
# modules/llm/new_model.py
import torch
import torch.nn as nn

from ..block.new_model_block import NewModelBlock
from ..common.layer_norm import RMSNorm
from ..common.positional_embedding import compute_rope_parameters

class NewModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.token_embedding = nn.Embedding(
            config["vocab_size"], 
            config["embedding_dim"], 
            dtype=config["dtype"]
        )

        self.blocks = nn.ModuleList([
            NewModelBlock(config) for _ in range(config["num_layers"])
        ])

        self.norm = RMSNorm(config, is_layer_norm=True)
        self.lm_head = nn.Linear(
            config["embedding_dim"], 
            config["vocab_size"], 
            dtype=config["dtype"]
        )

        # RoPE parameters
        if config["head_dim"] is None:
            self.head_dim = config["embedding_dim"] // config["num_heads"]
        else:
            self.head_dim = config["head_dim"]

        self.cos, self.sin = compute_rope_parameters(
            self.head_dim, 
            dtype=config["dtype"]
        )

        self.register_buffer("cos", self.cos, persistent=False)
        self.register_buffer("sin", self.sin, persistent=False)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape

        # Token embedding
        x = self.token_embedding(input_ids)

        # Attention mask
        attention_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device), 
            diagonal=1
        )

        # Transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask, self.cos, self.sin)

        # Final normalization and output
        x = self.norm(x)
        logits = self.lm_head(x)

        return logits
```

### 5. Create Weight Mapping Function

Add weight mapping in `modules/load_model.py`:

```python
def load_weights_into_new_model(model, config, params):
    """
    Load pretrained weights into NewModel with proper mapping.
    """
    def assign(left, right, tensor_name="unknown"):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}")
        return torch.nn.Parameter(right.clone().detach())
    
    # Token embedding
    model.token_embedding.weight = assign(
        model.token_embedding.weight, 
        params["model.embed_tokens.weight"], 
        "model.embed_tokens.weight"
    )
    
    # Process each transformer block
    for l in range(config["num_layers"]):
        block = model.blocks[l]
        
        # Attention weights
        block.attention.q_proj.weight = assign(
            block.attention.q_proj.weight,
            params[f"model.layers.{l}.self_attn.q_proj.weight"],
            f"model.layers.{l}.self_attn.q_proj.weight"
        )
        # ... (continue for k_proj, v_proj, output_proj)
        
        # Layer norms
        block.norm1.scale = assign(
            block.norm1.scale,
            params[f"model.layers.{l}.input_layernorm.weight"],
            f"model.layers.{l}.input_layernorm.weight"
        )
        # ... (continue for norm2)
        
        # Feed-forward weights
        # ... (map FFN weights)
    
    # Final layer norm and output head
    model.norm.scale = assign(
        model.norm.scale, 
        params["model.norm.weight"], 
        "model.norm.weight"
    )
    
    if "lm_head.weight" in params:
        model.lm_head.weight = assign(
            model.lm_head.weight, 
            params["lm_head.weight"], 
            "lm_head.weight"
        )
    else:
        # Weight tying
        model.lm_head.weight = assign(
            model.lm_head.weight, 
            params["model.embed_tokens.weight"], 
            "model.embed_tokens.weight"
        )
```

### 6. Register in ModelRegistry

Update `ModelRegistry` in `modules/load_model.py`:

```python
# Add to imports
from .llm import Qwen3, Qwen1, NewModel

# Add to SUPPORTED_MODELS
SUPPORTED_MODELS = {
    "qwen3": {
        "class": Qwen3,
        "weight_mapping": "qwen3_mapping",
        "repo_patterns": {
            "base": "Qwen/Qwen3-{size}-Base",
            "instruct": "Qwen/Qwen3-{size}",
            "reasoning": "Qwen/Qwen3-{size}"
        }
    },
    "qwen1": {
        "class": Qwen1,
        "weight_mapping": "qwen1_mapping",
        "repo_patterns": {
            "base": "Qwen/Qwen-{size}-Chat",
            "instruct": "Qwen/Qwen-{size}-Chat",
            "chat": "Qwen/Qwen-{size}-Chat"
        }
    },
    "new_model": {
        "class": NewModel,
        "weight_mapping": "new_model_mapping",
        "repo_patterns": {
            "base": "Organization/NewModel-{size}-Base",
            "instruct": "Organization/NewModel-{size}-Instruct"
        }
    }
}

# Add to WEIGHT_MAPPINGS
WEIGHT_MAPPINGS = {
    "qwen3_mapping": load_weights_into_qwen3,
    "qwen1_mapping": load_weights_into_qwen1,
    "new_model_mapping": load_weights_into_new_model
}

# Add convenience function
def load_new_model(size: str = "7B", variant: str = "base", **kwargs):
    """Convenience function to load NewModel."""
    return load_pretrained_model(model_type="new_model", size=size, variant=variant, **kwargs)
```

### 7. Update Module Exports

Update `modules/llm/__init__.py`:

```python
from .qwen3 import Qwen3
from .qwen1 import Qwen1
from .new_model import NewModel
```

### 8. Create Documentation

Create documentation file `docs/models/NewModel.md` using the template.

### 9. Testing

Create a test script:

```python
# test_new_model.py
import torch
from modules.load_model import load_config, load_model

def test_new_model():
    config_path = "configs/new_model/new_model_7B.yaml"
    config = load_config(config_path)
    
    model = load_model(config, "new_model")
    
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
    
    model.eval()
    with torch.no_grad():
        logits = model(input_ids)
    
    expected_shape = (batch_size, seq_len, config["vocab_size"])
    assert logits.shape == expected_shape, f"Expected {expected_shape}, got {logits.shape}"
    
    print("✅ NewModel test passed!")

if __name__ == "__main__":
    test_new_model()
```

## Best Practices

### 1. Code Organization
- Follow the existing module structure
- Use descriptive naming conventions
- Document all classes and methods

### 2. Configuration Management
- Use YAML for configurations
- Include all necessary parameters
- Provide sensible defaults

### 3. Weight Mapping
- Handle shape mismatches gracefully
- Support both tied and untied embeddings
- Include detailed error messages

### 4. Testing
- Test with dummy inputs first
- Verify output shapes
- Check memory usage
- Compare with reference implementations

### 5. Documentation
- Use the template provided
- Include architecture diagrams
- Provide usage examples
- Document any special requirements

## Common Patterns

### Attention Variations
```python
# Standard attention
scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

# Grouped query attention
k_expanded = k.repeat_interleave(group_size, dim=1)
v_expanded = v.repeat_interleave(group_size, dim=1)

# Multi-query attention
k_expanded = k.expand(-1, num_heads, -1, -1)
v_expanded = v.expand(-1, num_heads, -1, -1)
```

### Layer Normalization
```python
# Pre-norm (recommended)
normalized = self.norm(x)
output = self.layer(normalized)
x = x + output

# Post-norm (traditional)
output = self.layer(x)
x = self.norm(x + output)
```

### RoPE Application
```python
# Standard RoPE
q_rotated = apply_rope(q, cos, sin)
k_rotated = apply_rope(k, cos, sin)

# Extended RoPE for longer contexts
cos, sin = compute_rope_parameters(
    head_dim, 
    theta_base=1000000.0,  # Higher base for longer sequences
    context_len=131072
)
```

## Troubleshooting

### Common Issues

1. **Shape Mismatches**: Check tensor dimensions carefully
2. **Memory Errors**: Verify gradient accumulation and batch sizes
3. **NaN Values**: Check for proper initialization and normalization
4. **Slow Performance**: Profile attention computations and memory access

### Debugging Tips

1. **Use Small Models**: Start with minimal configurations
2. **Check Gradients**: Verify gradient flow through the model
3. **Compare Outputs**: Test against reference implementations
4. **Monitor Memory**: Track GPU memory usage during training/inference