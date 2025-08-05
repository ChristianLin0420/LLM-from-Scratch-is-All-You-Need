# Attention Mechanisms

## Overview

This document compares the different attention mechanisms implemented in the LLM from Scratch framework.

## Multi-Head Attention (MHA) vs Grouped Query Attention (GQA)

### Multi-Head Attention (Qwen1)

```
📊 Standard Multi-Head Attention Flow
│
Input: [batch_size, seq_len, d_model]
    ↓
┌─────────────────────────────────┐
│ Linear Projections (Equal Size) │
├─ Q: d_model → num_heads × head_dim
├─ K: d_model → num_heads × head_dim
└─ V: d_model → num_heads × head_dim
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│ Positional Encoding             │
├─ Apply RoPE to Q
└─ Apply RoPE to K
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│ Attention Computation           │
├─ Q @ K^T / √head_dim
├─ Apply causal mask
├─ Softmax
└─ Apply to V
└─────────────────────────────────┘
    ↓
Output Projection
    ↓
Output: [batch_size, seq_len, d_model]
```

**Characteristics:**
- Equal number of heads for Q, K, V
- Full KV cache requirements
- Standard transformer attention

### Grouped Query Attention (Qwen3)

```
📊 Grouped Query Attention Flow
│
Input: [batch_size, seq_len, d_model]
    ↓
┌─────────────────────────────────┐
│ Linear Projections (Grouped)    │
├─ Q: d_model → num_heads × head_dim
├─ K: d_model → num_kv_groups × head_dim  ⭐ Fewer heads
└─ V: d_model → num_kv_groups × head_dim  ⭐ Fewer heads
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│ QK Normalization + RoPE         │
├─ Apply RMSNorm to Q ⭐ Unique
├─ Apply RMSNorm to K ⭐ Unique
├─ Apply RoPE to Q
└─ Apply RoPE to K
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│ Group Expansion                 │
├─ Expand K: kv_groups → num_heads
└─ Expand V: kv_groups → num_heads
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│ Attention Computation           │
├─ Q @ K^T / √head_dim
├─ Apply causal mask
├─ Softmax
└─ Apply to V
└─────────────────────────────────┘
    ↓
Output Projection
    ↓
Output: [batch_size, seq_len, d_model]
```

**Characteristics:**
- Fewer KV heads than Q heads
- Reduced memory usage
- Shared KV heads across query groups

## Memory Comparison

| Model | Q Heads | K Heads | V Heads | KV Cache Reduction |
|-------|---------|---------|---------|-------------------|
| **Qwen1-7B** | 32 | 32 | 32 | 1× (baseline) |
| **Qwen3-8B** | 32 | 16 | 16 | 2× reduction |
| **Qwen3-32B** | 64 | 32 | 32 | 2× reduction |

## Implementation Details

### Standard MHA Implementation

```python
def multi_head_attention(q, k, v, mask):
    # q, k, v: [batch, num_heads, seq_len, head_dim]
    scores = torch.matmul(q, k.transpose(-2, -1))
    scores = scores / math.sqrt(head_dim)
    scores = scores.masked_fill(mask, float('-inf'))
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)
    return output
```

### Grouped Query Attention Implementation

```python
def grouped_query_attention(q, k, v, mask, num_kv_groups):
    # q: [batch, num_heads, seq_len, head_dim]
    # k, v: [batch, num_kv_groups, seq_len, head_dim]
    
    group_size = num_heads // num_kv_groups
    
    # Expand k, v to match q
    k = k.repeat_interleave(group_size, dim=1)
    v = v.repeat_interleave(group_size, dim=1)
    
    # Standard attention computation
    scores = torch.matmul(q, k.transpose(-2, -1))
    scores = scores / math.sqrt(head_dim)
    scores = scores.masked_fill(mask, float('-inf'))
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)
    return output
```

## Performance Analysis

### Memory Usage

```
📊 Memory Usage Comparison

┌─────────────────────────────────┐  ┌─────────────────────────────────┐
│    MHA Memory Usage (Qwen1)     │  │    GQA Memory Usage (Qwen3)     │
├─────────────────────────────────┤  ├─────────────────────────────────┤
│                                 │  │                                 │
│ Q Cache: num_heads × seq × head │  │ Q Cache: num_heads × seq × head │
│ K Cache: num_heads × seq × head │  │ K Cache: kv_groups × seq × head │
│ V Cache: num_heads × seq × head │  │ V Cache: kv_groups × seq × head │
│                                 │  │                                 │
│ Total KV: 2 × num_heads × ...   │  │ Total KV: 2 × kv_groups × ...   │
│                                 │  │                                 │
│ Example (32 heads):             │  │ Example (32 Q, 16 KV):          │
│ ├─ Q: 32 × seq × 128            │  │ ├─ Q: 32 × seq × 128            │
│ ├─ K: 32 × seq × 128            │  │ ├─ K: 16 × seq × 128 ⭐         │
│ └─ V: 32 × seq × 128            │  │ └─ V: 16 × seq × 128 ⭐         │
│                                 │  │                                 │
│ KV Memory: 100% (baseline)      │  │ KV Memory: 50% (2× reduction)   │
└─────────────────────────────────┘  └─────────────────────────────────┘
```

### Speed Comparison

| Operation | MHA | GQA | Speedup |
|-----------|-----|-----|---------|
| **Forward Pass** | 1.0× | 1.1×-1.3× | Modest |
| **KV Cache** | 1.0× | 2.0×-4.0× | Significant |
| **Memory Bandwidth** | 1.0× | 1.5×-2.5× | Good |

## QK Normalization

Only available in Qwen3 (GQA):

```python
# Apply normalization to queries and keys
if self.q_norm:
    q = self.q_norm(q)  # RMSNorm on queries
if self.k_norm:
    k = self.k_norm(k)  # RMSNorm on keys
```

**Benefits:**
- Improved training stability
- Better gradient flow
- Enhanced long-context performance