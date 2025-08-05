# Qwen3 Model Documentation

## 🎯 Overview

Qwen3 represents the latest generation of the Qwen model family, featuring advanced architectural improvements including Grouped Query Attention (GQA), QK normalization, and extended context lengths. This model is designed for high performance and efficiency in production environments.

## 🏗️ Architecture

### High-Level Architecture

```
🏗️ Qwen3 Model Architecture (Advanced Transformer)
│
Input IDs
    ↓
Token Embedding
    ↓
┌─────────────────────────────────┐
│ Qwen3 Blocks (x N layers)       │
│                                 │
│ ┌─────────────────────────────┐ │
│ │ Single Qwen3 Block          │ │
│ │                             │ │
│ │ Input                       │ │
│ │   ↓                         │ │
│ │ RMSNorm                     │ │
│ │   ↓                         │ │
│ │ Grouped Query Attention     │ │
│ │   ↓                         │ │
│ │ Residual Connection         │ │
│ │   ↓                         │ │
│ │ RMSNorm                     │ │
│ │   ↓                         │ │
│ │ Feed Forward Network        │ │
│ │   ↓                         │ │
│ │ Residual Connection         │ │
│ └─────────────────────────────┘ │
└─────────────────────────────────┘
    ↓
Final RMSNorm
    ↓
Language Model Head
    ↓
Output Logits

📋 Grouped Query Attention Details:
┌─────────────────────────────────┐
│ Input → Q, K, V Projections     │
│         ↓   ↓   ↓               │
│    QK Norm QK Norm V            │
│         ↓   ↓   ↓               │
│      RoPE RoPE  V               │
│         ↓   ↓   ↓               │
│    Group Expansion              │
│         ↓                       │
│    Attention Computation        │
│         ↓                       │
│    Output Projection            │
└─────────────────────────────────┘
```

### Key Components

#### 1. Grouped Query Attention (GQA)
```
📊 Grouped Query Attention Flow
├─ Input: [batch_size, seq_len, d_model]
│
├─ Linear Projections:
│  ├─ Q Projection → [batch_size, seq_len, num_heads, head_dim]
│  ├─ K Projection → [batch_size, seq_len, num_kv_groups, head_dim]
│  └─ V Projection → [batch_size, seq_len, num_kv_groups, head_dim]
│
├─ QK Normalization (unique to Qwen3):
│  ├─ Apply RMSNorm to Q
│  └─ Apply RMSNorm to K
│
├─ Positional Encoding:
│  ├─ Apply RoPE to Q
│  └─ Apply RoPE to K
│
├─ Group Expansion:
│  ├─ Expand K: repeat_interleave(group_size) → num_heads
│  └─ Expand V: repeat_interleave(group_size) → num_heads
│
├─ Attention Computation:
│  ├─ Compute attention scores: Q @ K^T / √head_dim
│  ├─ Apply causal mask
│  ├─ Softmax normalization
│  └─ Apply to values: attention_weights @ V
│
├─ Reshape & Project:
│  ├─ Concatenate heads → [batch_size, seq_len, d_model]
│  └─ Output projection
│
└─ Output: [batch_size, seq_len, d_model]
```

**GQA Benefits:**
- **Memory Efficiency**: Reduces KV cache size by sharing key-value heads
- **Inference Speed**: Faster attention computation with fewer KV heads
- **Quality Retention**: Maintains model quality with reduced parameters

#### 2. Feed Forward Network (SwiGLU)
```
🔀 SwiGLU Feed Forward Network
│
Input
├─ Linear 1 (Gate) ──┐
│                    ├─ Element-wise Multiply ─┐
├─ Linear 2 (Up) ────┤                         │
   │                 │                         │
   └─ SiLU Activation─┘                        │
                                               │
                                               ↓
                                         Linear 3 (Down)
                                               │
                                               ↓
                                            Output

🔍 Component Details:
├─ Gate: d_model → hidden_dim
├─ Up:   d_model → hidden_dim  
├─ SiLU: Sigmoid Linear Unit activation
├─ Multiply: Gate(x) * SiLU(Up(x))
└─ Down: hidden_dim → d_model
```

#### 3. Positional Encoding (Extended RoPE)
```
🔄 Extended Rotary Position Embedding (RoPE)
│
Query/Key Tensors [batch, seq_len, heads, head_dim]
    ↓
Split head_dim into pairs: [(d0,d1), (d2,d3), ...]
    ↓
Position Calculation (Extended for Long Context):
├─ Position indices: [0, 1, 2, ..., seq_len-1]
├─ Extended frequency: 1000000^(-2i/head_dim)  # Higher base
├─ Angles: position × frequency
└─ Sin/Cos values: sin(angles), cos(angles)
    ↓
Apply Rotation:
├─ For each pair (x, y):
│  ├─ x_new = x * cos - y * sin
│  └─ y_new = x * sin + y * cos
└─ Concatenate rotated pairs
    ↓
Rotated Query/Key Tensors

🎯 Extended RoPE Benefits:
├─ Support for very long contexts (128K+ tokens)
├─ Better extrapolation beyond training length
├─ Stable attention patterns at extended lengths
└─ Maintains relative position information
```

## 📊 Model Specifications

### Available Configurations

| Model | Parameters | Embedding Dim | Layers | Heads | KV Groups | Context Length |
|-------|------------|---------------|--------|-------|-----------|----------------|
| **Qwen3-0.6B** | 0.6B | 1024 | 24 | 16 | 8 | 32K |
| **Qwen3-1.7B** | 1.7B | 2048 | 28 | 16 | 8 | 40K |
| **Qwen3-4B** | 4B | 2560 | 36 | 32 | 16 | 40K |
| **Qwen3-8B** | 8B | 4096 | 36 | 32 | 16 | 128K |
| **Qwen3-14B** | 14B | 5120 | 40 | 40 | 20 | 128K |
| **Qwen3-32B** | 32B | 5120 | 64 | 64 | 32 | 128K |
| **Qwen3-30B-A3B** | 30B (3B active) | 2048 | 48 | 32 | 16 | 262K |

### Configuration Parameters

```yaml
# Example: Qwen3 8B Configuration
vocab_size: 151936
context_length: 128000
embedding_dim: 4096
num_heads: 32
num_kv_groups: 16  # Key difference from standard MHA
num_layers: 36
hidden_dim: 22016
head_dim: 128

# Advanced features
use_qk_norm: true      # QK normalization for stability
rope_base: 1000000.0   # Extended RoPE for long contexts
qwen3_compatiable: true
```

## 🚀 Usage Examples

### Basic Usage

```python
from modules.load_model import load_qwen3

# Load Qwen3 model
model, tokenizer, config = load_qwen3(
    size="8B",
    variant="instruct",
    device="cuda"
)

# Generate text
from modules.text_generation import TextGenerator
generator = TextGenerator(model, tokenizer, device="cuda")

response = generator.chat("Explain the benefits of Grouped Query Attention")
print(response)
```

### Advanced Configuration

```python
from modules.load_model import load_pretrained_model

# Load with custom config
model, tokenizer, config = load_pretrained_model(
    model_type="qwen3",
    size="14B",
    variant="reasoning",
    config_path="configs/qwen3/custom_qwen3_14B.yaml",
    device="cuda"
)

# Streaming generation with advanced sampling
for token in generator.generate(
    "Write a technical explanation of transformer architecture:",
    max_new_tokens=500,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    stream=True
):
    print(token, end='', flush=True)
```

### Command Line Usage

```bash
# Available sizes: 0.6B, 1.7B, 4B, 8B, 14B, 32B, 30B_A3B
python main.py --model qwen3 --size 8B --prompt "Explain quantum computing"

# Interactive chat with reasoning mode
python main.py --model qwen3 --size 14B --variant reasoning --chat --stream

# Benchmark performance
python main.py --model qwen3 --size 32B --benchmark

# MoE model usage
python main.py --model qwen3 --size 30B_A3B --interactive
```

## 🔧 Technical Details

### Grouped Query Attention Implementation

```python
class GroupQueryAttention(nn.Module):
    def __init__(self, config):
        self.num_heads = config["num_heads"]
        self.num_kv_groups = config["num_kv_groups"]
        self.group_size = self.num_heads // self.num_kv_groups
        
        # Projections
        self.q_proj = nn.Linear(d_input, num_heads * head_dim)
        self.k_proj = nn.Linear(d_input, num_kv_groups * head_dim)  # Fewer KV heads
        self.v_proj = nn.Linear(d_input, num_kv_groups * head_dim)
        
        # QK Normalization
        if config["use_qk_norm"]:
            self.q_norm = RMSNorm(config)
            self.k_norm = RMSNorm(config)
    
    def forward(self, x, attention_mask, cos, sin):
        # Project and reshape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_groups, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_groups, self.head_dim)
        
        # Apply QK normalization
        if self.q_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        
        # Apply RoPE
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        
        # Expand K,V to match Q heads (group query attention)
        k = k.repeat_interleave(self.group_size, dim=2)
        v = v.repeat_interleave(self.group_size, dim=2)
        
        # Standard attention computation
        return attention_output
```

### Weight Mapping

Qwen3 weights are mapped from HuggingFace format:

```python
# Attention weights
model.blocks[i].attention.q_proj.weight ← params[f"model.layers.{i}.self_attn.q_proj.weight"]
model.blocks[i].attention.k_proj.weight ← params[f"model.layers.{i}.self_attn.k_proj.weight"]
model.blocks[i].attention.v_proj.weight ← params[f"model.layers.{i}.self_attn.v_proj.weight"]
model.blocks[i].attention.output_proj.weight ← params[f"model.layers.{i}.self_attn.o_proj.weight"]

# QK Normalization (if enabled)
model.blocks[i].attention.q_norm.scale ← params[f"model.layers.{i}.self_attn.q_norm.weight"]
model.blocks[i].attention.k_norm.scale ← params[f"model.layers.{i}.self_attn.k_norm.weight"]

# Feed Forward weights
model.blocks[i].ffn.fc1.weight ← params[f"model.layers.{i}.mlp.gate_proj.weight"]
model.blocks[i].ffn.fc2.weight ← params[f"model.layers.{i}.mlp.up_proj.weight"]
model.blocks[i].ffn.fc3.weight ← params[f"model.layers.{i}.mlp.down_proj.weight"]
```

## 📈 Performance Characteristics

### Memory Usage
- **GQA Reduction**: ~30-50% reduction in KV cache memory compared to MHA
- **Context Scaling**: Linear memory scaling with extended context lengths
- **MoE Efficiency**: 30B-A3B uses only 3B active parameters during inference

### Speed Benchmarks
| Model Size | Device | Throughput (tok/s) | Memory (GB) | Context Window |
|------------|--------|-------------------|-------------|----------------|
| 0.6B | GPU | 150-300 | 2-4 | 32K |
| 1.7B | GPU | 80-150 | 6-10 | 40K |
| 8B | GPU | 20-40 | 24-32 | 128K |
| 32B | GPU | 5-15 | 80-120 | 128K |

## 🔍 Architecture Advantages

### 1. Grouped Query Attention
- **Memory Efficiency**: Reduces KV cache by factor of `num_heads/num_kv_groups`
- **Inference Speed**: Faster attention computation
- **Quality**: Maintains performance with reduced parameters

### 2. QK Normalization
- **Training Stability**: Prevents attention distribution collapse
- **Gradient Flow**: Better gradient propagation in deep networks
- **Long Context**: Improved handling of extended sequences

### 3. Extended Context
- **Long Documents**: Handle up to 128K-262K tokens
- **Complex Tasks**: Better performance on multi-turn conversations
- **Retrieval**: Enhanced in-context learning capabilities

## 🛠️ Development Notes

### Repository Patterns
```python
# HuggingFace repository patterns
"base": "Qwen/Qwen3-{size}-Base"
"instruct": "Qwen/Qwen3-{size}"
"reasoning": "Qwen/Qwen3-{size}"
```

### Configuration Files
- `configs/qwen3/qwen3_0.6B.yaml`
- `configs/qwen3/qwen3_1.7B.yaml`
- `configs/qwen3/qwen3_4B.yaml`
- `configs/qwen3/qwen3_8B.yaml`
- `configs/qwen3/qwen3_14B.yaml`
- `configs/qwen3/qwen3_32B.yaml`
- `configs/qwen3/qwen3_30B_A3B.yaml`

## 📚 Related Documentation

- [Qwen1 Model](Qwen1.md) - Comparison with standard MHA architecture
- [Architecture Overview](../architecture/attention.md) - Deep dive into attention mechanisms
- [Configuration Guide](../development/configuration.md) - How to customize model configs
- [Performance Tuning](../development/performance.md) - Optimization strategies

---

*For implementation details, see the source code in `modules/llm/qwen3.py` and `modules/block/qwen3_block.py`* 