# Qwen1 Model Documentation

## 🎯 Overview

Qwen1 implements the classic transformer architecture with standard Multi-Head Attention (MHA). This model provides a simpler, more traditional approach compared to Qwen3, making it ideal for research, experimentation, and understanding fundamental transformer concepts.

## 🏗️ Architecture

### High-Level Architecture

```
🏗️ Qwen1 Model Architecture
│
Input IDs
    ↓
Token Embedding
    ↓
┌─────────────────────────────────┐
│ Qwen1 Blocks (x N layers)       │
│                                 │
│ ┌─────────────────────────────┐ │
│ │ Single Qwen1 Block          │ │
│ │                             │ │
│ │ Input                       │ │
│ │   ↓                         │ │
│ │ RMSNorm                     │ │
│ │   ↓                         │ │
│ │ Multi-Head Attention        │ │
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

📋 Multi-Head Attention Details:
┌─────────────────────────────────┐
│ Input → Q, K, V Projections     │
│           ↓    ↓    ↓           │
│         RoPE RoPE   V           │
│           ↓    ↓    ↓           │
│      Standard Attention         │
│           ↓                     │
│      Output Projection          │
└─────────────────────────────────┘
```

### Key Components

#### 1. Standard Multi-Head Attention (MHA)
```
📊 Multi-Head Attention Flow
├─ Input: [batch_size, seq_len, d_model]
│
├─ Linear Projections:
│  ├─ Q Projection → [batch_size, seq_len, num_heads, head_dim]
│  ├─ K Projection → [batch_size, seq_len, num_heads, head_dim]
│  └─ V Projection → [batch_size, seq_len, num_heads, head_dim]
│
├─ Positional Encoding:
│  ├─ Apply RoPE to Q
│  └─ Apply RoPE to K
│
├─ Attention Computation:
│  ├─ Transpose to [batch_size, num_heads, seq_len, head_dim]
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

**MHA Characteristics:**
- **Simplicity**: Equal number of heads for Q, K, V
- **Standard Implementation**: Classic transformer attention
- **Research Friendly**: Well-understood and documented

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

#### 3. Positional Encoding (Standard RoPE)
```
🔄 Rotary Position Embedding (RoPE)
│
Query/Key Tensors [batch, seq_len, heads, head_dim]
    ↓
Split head_dim into pairs: [(d0,d1), (d2,d3), ...]
    ↓
Position Calculation:
├─ Position indices: [0, 1, 2, ..., seq_len-1]
├─ Frequency calculation: 10000^(-2i/head_dim)
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

🎯 RoPE Benefits:
├─ Relative position encoding
├─ Extrapolation to longer sequences
└─ Preserves attention patterns
```

## 📊 Model Specifications

### Available Configurations

| Model | Parameters | Embedding Dim | Layers | Heads | Head Dim | Context Length |
|-------|------------|---------------|--------|-------|----------|----------------|
| **Qwen1-1.8B** | 1.8B | 2048 | 24 | 16 | 128 | 8K |
| **Qwen1-7B** | 7B | 4096 | 32 | 32 | 128 | 8K |
| **Qwen1-14B** | 14B | 5120 | 40 | 40 | 128 | 8K |
| **Qwen1-32B** | 32B | 5120 | 64 | 40 | 128 | 8K |

### Configuration Parameters

```yaml
# Example: Qwen1 7B Configuration
vocab_size: 151936
context_length: 8192          # Shorter than Qwen3
embedding_dim: 4096
num_heads: 32                 # Same for Q, K, V (no grouping)
num_layers: 32
hidden_dim: 11008
head_dim: 128

# Simplified features
use_qk_norm: false            # No QK normalization
rope_base: 10000.0            # Standard RoPE frequency
qwen3_compatiable: false      # Different architecture
```

## 🚀 Usage Examples

### Basic Usage

```python
from modules.load_model import load_qwen1

# Load Qwen1 model
model, tokenizer, config = load_qwen1(
    size="7B",
    variant="chat",
    device="cuda"
)

# Generate text
from modules.text_generation import TextGenerator
generator = TextGenerator(model, tokenizer, device="cuda")

response = generator.chat("Explain standard multi-head attention")
print(response)
```

### Research & Experimentation

```python
from modules.load_model import load_pretrained_model

# Load smaller model for quick experiments
model, tokenizer, config = load_pretrained_model(
    model_type="qwen1",
    size="1.8B",
    variant="chat",
    device="cuda"
)

# Test different attention patterns
def analyze_attention_patterns(model, text):
    # Your research code here
    pass
```

### Command Line Usage

```bash
# Available sizes: 1.8B, 7B, 14B, 32B
python main.py --model qwen1 --size 1.8B --prompt "Explain transformers"

# Interactive mode for experiments
python main.py --model qwen1 --size 7B --interactive

# Compare with Qwen3
python main.py --model qwen1 --size 14B --benchmark
python main.py --model qwen3 --size 14B --benchmark
```

## 🔧 Technical Details

### Multi-Head Attention Implementation

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        self.num_heads = config["num_heads"]
        self.head_dim = config["head_dim"]
        self.d_output = self.num_heads * self.head_dim
        
        # Standard MHA: same number of heads for Q, K, V
        self.q_proj = nn.Linear(d_input, self.d_output)
        self.k_proj = nn.Linear(d_input, self.d_output)  # Same size as Q
        self.v_proj = nn.Linear(d_input, self.d_output)  # Same size as Q
        
        self.output_proj = nn.Linear(self.d_output, d_input)
    
    def forward(self, x, attention_mask, cos, sin):
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Apply RoPE
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch, heads, seq, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Standard attention computation
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(attention_mask, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.d_output)
        
        return self.output_proj(context)
```

### Weight Mapping

Qwen1 weights are mapped from HuggingFace format (similar to Qwen3 but without QK norm):

```python
# Attention weights (same structure as Qwen3)
model.blocks[i].attention.q_proj.weight ← params[f"model.layers.{i}.self_attn.q_proj.weight"]
model.blocks[i].attention.k_proj.weight ← params[f"model.layers.{i}.self_attn.k_proj.weight"]
model.blocks[i].attention.v_proj.weight ← params[f"model.layers.{i}.self_attn.v_proj.weight"]
model.blocks[i].attention.output_proj.weight ← params[f"model.layers.{i}.self_attn.o_proj.weight"]

# No QK normalization weights (key difference from Qwen3)

# Feed Forward weights (same as Qwen3)
model.blocks[i].ffn.fc1.weight ← params[f"model.layers.{i}.mlp.gate_proj.weight"]
model.blocks[i].ffn.fc2.weight ← params[f"model.layers.{i}.mlp.up_proj.weight"]
model.blocks[i].ffn.fc3.weight ← params[f"model.layers.{i}.mlp.down_proj.weight"]
```

## 📈 Performance Characteristics

### Memory Usage
- **Standard MHA**: Full KV cache (no grouping)
- **Predictable Memory**: Linear scaling with sequence length
- **Research Friendly**: Easier to analyze and modify

### Speed Benchmarks
| Model Size | Device | Throughput (tok/s) | Memory (GB) | Context Window |
|------------|--------|-------------------|-------------|----------------|
| 1.8B | GPU | 100-200 | 4-8 | 8K |
| 7B | GPU | 30-60 | 16-24 | 8K |
| 14B | GPU | 15-30 | 32-48 | 8K |
| 32B | GPU | 5-15 | 80-120 | 8K |

## 🔍 Architecture Comparison: Qwen1 vs Qwen3

### Key Differences

| Feature | Qwen1 | Qwen3 | Impact |
|---------|-------|--------|--------|
| **Attention** | Multi-Head Attention | Grouped Query Attention | Memory usage, Speed |
| **QK Normalization** | ❌ No | ✅ Yes | Training stability |
| **Context Length** | 8K tokens | 32K-262K tokens | Long document handling |
| **RoPE Base** | 10,000 | 1,000,000 | Position encoding range |
| **Complexity** | Simple | Advanced | Implementation difficulty |

### When to Choose Qwen1

**Research & Education:**
- Understanding transformer fundamentals
- Implementing custom attention mechanisms
- Debugging and analysis
- Quick prototyping

**Resource Constraints:**
- Limited memory environments
- Simpler deployment requirements
- CPU-only inference

**Specific Use Cases:**
- Short-context applications
- Traditional NLP tasks
- Model comparison studies

### Architecture Visualization: Side-by-Side

```
📊 Qwen1 vs Qwen3 Attention Comparison

┌─────────────────────────────┐  ┌─────────────────────────────┐
│    Qwen1 (Standard MHA)     │  │   Qwen3 (Grouped QA)       │
├─────────────────────────────┤  ├─────────────────────────────┤
│ Input                       │  │ Input                       │
│   ↓                         │  │   ↓                         │
│ Q/K/V Projections           │  │ Q/K/V Projections           │
│ ├─ Q: num_heads             │  │ ├─ Q: num_heads             │
│ ├─ K: num_heads             │  │ ├─ K: num_kv_groups         │
│ └─ V: num_heads             │  │ └─ V: num_kv_groups         │
│   ↓                         │  │   ↓                         │
│ RoPE (Q, K)                 │  │ QK Normalization + RoPE     │
│   ↓                         │  │   ↓                         │
│ Standard Attention          │  │ Grouped Attention           │
│ (All heads independent)     │  │ (Shared K/V across groups)  │
│   ↓                         │  │   ↓                         │
│ Output                      │  │ Output                      │
└─────────────────────────────┘  └─────────────────────────────┘

🔍 Key Differences:
├─ Qwen1: Equal heads for Q, K, V
├─ Qwen3: Fewer K/V heads (grouped)
├─ Qwen1: No QK normalization
├─ Qwen3: QK normalization for stability
├─ Qwen1: Higher memory usage
└─ Qwen3: Reduced memory, faster inference
```

## 🛠️ Development Notes

### Repository Patterns
```python
# HuggingFace repository patterns
"base": "Qwen/Qwen-{size}-Chat"
"instruct": "Qwen/Qwen-{size}-Chat"
"chat": "Qwen/Qwen-{size}-Chat"
```

### Configuration Files
- `configs/qwen1/qwen1_1.8B.yaml`
- `configs/qwen1/qwen1_7B.yaml`
- `configs/qwen1/qwen1_14B.yaml`
- `configs/qwen1/qwen1_32B.yaml`

### Implementation Files
- `modules/llm/qwen1.py` - Main model class
- `modules/block/qwen1_block.py` - Transformer block
- `modules/attention/multi_head_attention.py` - MHA implementation

## 🔧 Customization Examples

### Research Modifications

```python
# Example: Custom attention analysis
class AnalyzableQwen1Block(Qwen1Block):
    def forward(self, x, attention_mask, cos, sin):
        # Store attention weights for analysis
        hidden_states = self.norm1(x)
        attention_output, attention_weights = self.attention(
            hidden_states, attention_mask, cos, sin, return_attention=True
        )
        
        # Log attention patterns
        self.log_attention_patterns(attention_weights)
        
        # Continue normal forward pass
        x = x + attention_output
        residual = x
        x = self.norm2(x)
        ffn_output = self.ffn(x)
        x = residual + ffn_output
        
        return x
```

### Configuration Customization

```yaml
# Custom Qwen1 configuration for experiments
vocab_size: 151936
context_length: 4096          # Shorter for faster experiments
embedding_dim: 2048           # Smaller for resource constraints
num_heads: 16                 # Reduced complexity
num_layers: 20                # Fewer layers
hidden_dim: 5504

# Experimental features
use_qk_norm: false
rope_base: 10000.0
dtype: float32                # Full precision for research
```

## 📚 Related Documentation

- [Qwen3 Model](Qwen3.md) - Advanced architecture comparison
- [Attention Mechanisms](../architecture/attention.md) - MHA vs GQA deep dive
- [Model Development](../development/adding-models.md) - How Qwen1 was implemented
- [Performance Comparison](../architecture/performance.md) - Qwen1 vs Qwen3 benchmarks

---

*For implementation details, see the source code in `modules/llm/qwen1.py` and `modules/block/qwen1_block.py`*