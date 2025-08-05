# LLM from Scratch Documentation

This directory contains detailed documentation for all supported models in the LLM from Scratch framework.

## 📋 Available Models

### Production Ready
- **[Qwen1](models/Qwen1.md)** - Classic architecture with standard Multi-Head Attention
- **[Qwen3](models/Qwen3.md)** - Advanced architecture with Grouped Query Attention and extended features

### Model Comparison

| Model | Architecture | Attention | Context Length | QK Norm | Sizes Available |
|-------|-------------|-----------|----------------|---------|----------------|
| **Qwen1** | Standard Transformer | Multi-Head Attention (MHA) | 8K tokens | ❌ No | 1.8B, 7B, 14B, 32B |
| **Qwen3** | Advanced Transformer | Grouped Query Attention (GQA) | 40K-128K tokens | ✅ Yes | 0.6B, 1.7B, 4B, 8B, 14B, 32B, 30B-A3B |

## 📚 Documentation Structure

```
docs/
├── README.md                 # This overview
├── models/                   # Individual model documentation
│   ├── Qwen1.md             # Qwen1 detailed documentation
│   ├── Qwen3.md             # Qwen3 detailed documentation
│   └── template.md          # Template for new models
├── architecture/             # Architecture guides
│   ├── attention.md         # Attention mechanisms
│   ├── positional-encoding.md # RoPE and position encoding
│   └── weight-mapping.md    # Weight loading patterns
└── development/             # Development guides
    ├── adding-models.md     # How to add new models
    ├── configuration.md     # Configuration system
    └── testing.md           # Testing guidelines
```

## 🚀 Quick Navigation

### By Use Case
- **Research & Experimentation**: Start with [Qwen1](models/Qwen1.md) for simplicity
- **Production & Efficiency**: Use [Qwen3](models/Qwen3.md) for advanced features
- **Understanding Architecture**: Read [Architecture Guides](architecture/)
- **Contributing**: Check [Development Guides](development/)

### By Model Size
- **Small Models (< 2B)**: Qwen1 1.8B, Qwen3 0.6B-1.7B
- **Medium Models (2-10B)**: Qwen1 7B, Qwen3 4B-8B
- **Large Models (10B+)**: Qwen1 14B-32B, Qwen3 14B-32B
- **MoE Models**: Qwen3 30B-A3B (3B active parameters)

## 🔧 Technical References

### Architecture Components
- [Attention Mechanisms](architecture/attention.md) - MHA vs GQA comparison
- [Positional Encoding](architecture/positional-encoding.md) - RoPE implementation
- [Feed Forward Networks](architecture/ffn.md) - SwiGLU and gated activations

### Implementation Details
- [Weight Mapping](architecture/weight-mapping.md) - HuggingFace compatibility
- [Model Registry](development/model-registry.md) - Adding new architectures
- [Configuration System](development/configuration.md) - YAML-based configs

## 📖 Getting Started

1. **Choose a Model**: Review the comparison table above
2. **Read Model Docs**: Check the detailed documentation for your chosen model
3. **Follow Examples**: Each model doc includes usage examples
4. **Explore Architecture**: Dive into technical details as needed

## 🤝 Contributing Documentation

When adding a new model:
1. Copy `models/template.md` to `models/YourModel.md`
2. Update this README with the new model entry
3. Add architecture diagrams using Mermaid syntax
4. Include configuration examples and usage patterns

---

*For the main project documentation, see the [root README.md](../README.md)*