# LLM from Scratch - Complete Framework

A flexible, extensible framework for loading and running large language models from scratch. Supporting multiple model architectures with easy extensibility for adding new models.

```
🏗️ LLM from Scratch Framework
│
├── 🤖 Models
│   ├── Qwen1 (Standard Transformer)
│   │   ├── 1.8B - 32B Models
│   │   ├── Standard Multi-Head Attention
│   │   └── Research Friendly
│   │
│   ├── Qwen3 (Advanced Transformer)
│   │   ├── 0.6B - 32B Dense Models
│   │   ├── 30B-A3B MoE Model
│   │   └── GQA + QK Normalization
│   │
│   └── Future Models (Extensible Design)
│
├── 📚 Documentation
│   ├── Individual Model Docs
│   ├── Architecture Guides
│   └── Development Guides
│
└── 💡 Examples
    ├── Quick Start
    ├── Advanced Usage
    └── Custom Models
```

## 🚀 Features

- **Multiple Model Support**: Qwen1, Qwen3 (various sizes from 0.6B to 32B + MoE)
- **Flexible Generation**: Streaming, batch, chat, and completion modes
- **Advanced Sampling**: Temperature, top-k, top-p, repetition penalty
- **Easy Configuration**: YAML-based model configurations
- **Extensible Architecture**: Simple framework for adding new models
- **Performance Optimized**: Efficient weight loading and text generation
- **Multiple Interfaces**: CLI, interactive modes, and programmatic API

## 📋 Supported Models

| Model | Architecture | Sizes Available | Key Features |
|-------|-------------|----------------|--------------|
| **[Qwen1](docs/models/Qwen1.md)** | Standard Transformer + MHA | 1.8B, 7B, 14B, 32B | Multi-Head Attention, Research-Friendly |
| **[Qwen3](docs/models/Qwen3.md)** | Advanced Transformer + GQA | 0.6B, 1.7B, 4B, 8B, 14B, 32B, 30B-A3B | Grouped Query Attention, QK Norm, Extended Context |

> 📚 **Detailed Documentation**: See [docs/](docs/) for comprehensive model documentation and architecture guides.

## 📦 Installation

### Prerequisites
```bash
pip install torch transformers tokenizers safetensors huggingface-hub pyyaml
```

### Clone Repository
```bash
git clone <repository-url>
cd LLM-from-Scratch-is-All-You-Need
```

## 🎯 Quick Start

### 1. Command Line Interface

```bash

# Qwen1 models (standard transformer)
python main.py --model qwen1 --size 7B --prompt "Explain transformers"
python main.py --model qwen1 --size 1.8B --interactive

# Qwen3 models (advanced features)
python main.py --model qwen3 --size 8B --prompt "Explain quantum computing"
python main.py --model qwen3 --size 14B --chat --stream

# Run benchmarks
python main.py --model qwen3 --size 32B --benchmark
python main.py --model qwen1 --size 32B --benchmark
```

### 2. Programmatic Usage

```python
from modules.load_model import load_qwen3, load_qwen1
from modules.text_generation import TextGenerator

# Load Qwen3 model (advanced features)
model, tokenizer, config = load_qwen3(
    size="8B",
    variant="instruct",  # or "base", "reasoning"
    device="cuda"
)

# Or load Qwen1 model (standard transformer)
# model, tokenizer, config = load_qwen1(
#     size="7B",
#     variant="chat",
#     device="cuda"
# )

# Create generator
generator = TextGenerator(model, tokenizer, device="cuda")

# Generate text
response = generator.chat("What is artificial intelligence?")
print(response)
```

### 3. Quick Start Example

```bash
python examples/quick_start.py
```

## 🛠️ Advanced Usage

### Configuration Files

Model configurations are stored in:
- **Qwen3**: `configs/qwen3/` - Advanced transformer configurations
- **Qwen1**: `configs/qwen1/` - Standard transformer configurations

> 📋 See [Model Documentation](docs/models/) for detailed configuration options

### Custom Configuration

```python
from modules.load_model import load_config, load_pretrained_model

# Load any model with custom config
model, tokenizer, config = load_pretrained_model(
    model_type="qwen3",  # or "qwen1"
    size="8B",
    config_path="path/to/custom_config.yaml"
)
```

### Streaming Generation

```python
# Streaming text generation
for token in generator.generate(
    "Write a story about",
    max_new_tokens=100,
    temperature=0.8,
    stream=True
):
    print(token, end='', flush=True)
```

### Advanced Sampling

```python
# Fine-tuned generation parameters
result = generator.generate(
    prompt="Creative writing prompt:",
    max_new_tokens=200,
    temperature=0.8,          # Creativity level
    top_k=40,                 # Limit to top 40 tokens
    top_p=0.9,                # Nucleus sampling
    repetition_penalty=1.1,   # Reduce repetition
    num_return_sequences=3    # Generate 3 variations
)
```

## 📊 Quick Model Comparison

| Model Family | Attention Type | Context Length | Best For |
|--------------|---------------|----------------|----------|
| **Qwen1** | Multi-Head Attention | 8K tokens | Research, Education |
| **Qwen3** | Grouped Query Attention | 32K-262K tokens | Production, Long contexts |

> 📋 **Detailed Specifications**: See [Qwen1 Models](docs/models/Qwen1.md) and [Qwen3 Models](docs/models/Qwen3.md)

## 🎮 Command Line Options

### Model Selection
- `--model`: Model type (`qwen1`, `qwen3`)
- `--size`: Model size (varies by model - see docs)
- `--variant`: Model variant (`base`, `instruct`, `reasoning`, `chat`)
- `--device`: Compute device (`auto`, `cpu`, `cuda`)

### Generation Parameters
- `--max-tokens`: Maximum tokens to generate
- `--temperature`: Sampling temperature (0.0-2.0)
- `--top-k`: Top-k sampling parameter
- `--top-p`: Top-p (nucleus) sampling parameter
- `--repetition-penalty`: Repetition penalty (1.0 = no penalty)

### Operation Modes
- `--prompt`: Single prompt generation
- `--chat`: Interactive chat mode
- `--interactive`: Interactive completion mode
- `--benchmark`: Run performance benchmarks
- `--stream`: Enable streaming output

## 🏗️ Architecture

### Core Components

1. **Model Loader** (`modules/load_model.py`)
   - Flexible model registration system
   - YAML configuration loading
   - Automatic weight mapping
   - HuggingFace integration

2. **Text Generator** (`modules/text_generation.py`)
   - Multiple sampling strategies
   - Streaming support
   - Batch generation
   - Performance optimizations

3. **Model Registry** (`modules/load_model.py`)
   - Extensible model support
   - Weight mapping functions
   - Repository patterns

### Adding New Models

The framework is designed for easy extensibility. To add a new model:

1. **Follow the Model Template**: Use the structure from existing models
2. **Implement Core Components**: Attention, blocks, and main model class
3. **Create Configurations**: YAML files for different sizes
4. **Add Weight Mapping**: HuggingFace compatibility layer
5. **Register in ModelRegistry**: Make it available through the API

> 📖 **Detailed Guide**: See [Adding Models Documentation](docs/development/adding-models.md)

## 🔧 Development

### Project Structure
```
├── configs/                 # Model configurations
│   ├── qwen3/              # Qwen3 configs
│   └── qwen1/              # Qwen1 configs
├── docs/                   # Documentation
│   ├── models/             # Individual model docs
│   ├── architecture/       # Technical guides
│   └── development/        # Development guides
├── modules/                # Core modules
│   ├── llm/               # Model implementations
│   ├── attention/         # Attention mechanisms
│   ├── block/             # Model blocks
│   └── common/            # Shared components
├── examples/              # Usage examples
└── main.py               # Main CLI interface
```

### Running Examples
```bash
# Basic usage examples
python examples/quick_start.py

# Advanced features demonstration
python examples/advanced_usage.py
```

## 📈 Performance

Example performance characteristics:

| Model Family | Size | Device | Speed (tok/s) | Memory (GB) |
|--------------|------|--------|---------------|-------------|
| **Qwen1** | 7B | GPU | 30-60 | 16-24 |
| **Qwen3** | 1.7B | GPU | 80-150 | 6-10 |
| **Qwen3** | 8B | GPU | 20-40 | 24-32 |

> 📊 **Detailed Benchmarks**: See individual model documentation for comprehensive performance analysis

*Performance varies based on hardware specifications and generation parameters.*

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and examples
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Qwen team for the model architectures
- HuggingFace for tokenizers and model hub
- PyTorch team for the deep learning framework

---

## 📚 Documentation

- **[📋 Model Overview](docs/)** - Compare all supported models
- **[🔧 Qwen1 Documentation](docs/models/Qwen1.md)** - Standard transformer with MHA
- **[🏗️ Qwen3 Documentation](docs/models/Qwen3.md)** - Advanced transformer with GQA
- **[⚙️ Architecture Guides](docs/architecture/)** - Technical deep dives
- **[🛠️ Development Guides](docs/development/)** - Adding new models

**Happy generating! 🎉**