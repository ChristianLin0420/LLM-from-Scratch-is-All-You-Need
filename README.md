# LLM from Scratch - Complete Framework

A flexible, extensible framework for loading and running large language models from scratch. Currently supports Qwen3 models with easy extensibility for adding new model architectures.

## 🚀 Features

- **Multiple Model Support**: Qwen3 (1.7B to 32B parameters + MoE)
- **Flexible Generation**: Streaming, batch, chat, and completion modes
- **Advanced Sampling**: Temperature, top-k, top-p, repetition penalty
- **Easy Configuration**: YAML-based model configurations
- **Extensible Architecture**: Simple framework for adding new models
- **Performance Optimized**: Efficient weight loading and text generation
- **Multiple Interfaces**: CLI, interactive modes, and programmatic API

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
# Single prompt generation
python main.py --model qwen3 --size 1.7B --prompt "Explain quantum computing"

# Interactive chat mode
python main.py --model qwen3 --size 1.7B --chat --stream

# Text completion mode
python main.py --model qwen3 --size 1.7B --interactive

# Run benchmarks
python main.py --model qwen3 --size 1.7B --benchmark
```

### 2. Programmatic Usage

```python
from modules import load_qwen3, TextGenerator

# Load model
model, tokenizer, config = load_qwen3(
    size="1.7B",
    variant="base",  # or "instruct", "reasoning"
    device="cuda"    # or "cpu", "auto"
)

# Create generator
generator = TextGenerator(model, tokenizer, device="cuda")

# Generate text
response = generator.chat("What is artificial intelligence?")
print(response)

# Text completion
completion = generator.complete("The future of technology")
print(completion)
```

### 3. Quick Start Example

```bash
python examples/quick_start.py
```

## 🛠️ Advanced Usage

### Configuration Files

Model configurations are stored in `configs/qwen/`:
- `qwen3_1.7B.yaml` - 1.7B parameter model
- `qwen3_4B.yaml` - 4B parameter model
- `qwen3_8B.yaml` - 8B parameter model
- `qwen3_14B.yaml` - 14B parameter model
- `qwen3_32B.yaml` - 32B parameter model
- `qwen3_30B_A3B.yaml` - 30B MoE model

### Custom Configuration

```python
from modules.load_model import load_config, load_pretrained_model

# Load custom config
config = load_config("path/to/custom_config.yaml")

# Load model with custom config
model, tokenizer, config = load_pretrained_model(
    model_type="qwen3",
    size="1.7B",
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

## 📊 Model Specifications

| Model | Parameters | Embedding Dim | Layers | Heads | Context Length |
|-------|------------|---------------|--------|-------|----------------|
| 1.7B  | 1.7B       | 2048         | 28     | 16    | 40,960         |
| 4B    | 4B         | 2560         | 36     | 32    | 40,960         |
| 8B    | 8B         | 4096         | 36     | 32    | 40,960         |
| 14B   | 14B        | 5120         | 40     | 40    | 40,960         |
| 32B   | 32B        | 5120         | 64     | 64    | 40,960         |
| 30B-A3B | 30B      | 2048         | 48     | 32    | 262,144        |

## 🎮 Command Line Options

### Model Selection
- `--model`: Model type (qwen3)
- `--size`: Model size (1.7B, 4B, 8B, 14B, 32B, 30B_A3B)
- `--variant`: Model variant (base, instruct, reasoning)
- `--device`: Compute device (auto, cpu, cuda)

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

To add support for a new model type:

1. **Implement the model class** in `modules/llm/`
2. **Create configuration templates** in `configs/`
3. **Add weight mapping function**
4. **Register in ModelRegistry**

Example:
```python
# Add to ModelRegistry.SUPPORTED_MODELS
"new_model": {
    "class": NewModel,
    "weight_mapping": "new_model_mapping",
    "repo_patterns": {
        "base": "Organization/NewModel-{size}-Base"
    }
}

# Implement weight mapping
def load_weights_into_new_model(model, config, params):
    # Weight mapping logic
    pass

# Register mapping
WEIGHT_MAPPINGS["new_model_mapping"] = load_weights_into_new_model
```

## 🔧 Development

### Project Structure
```
├── configs/                 # Model configurations
│   └── qwen/               # Qwen3 configs
├── modules/                # Core modules
│   ├── llm/               # Model implementations
│   ├── tokenizer/         # Tokenizer implementations
│   ├── common/            # Shared components
│   ├── attention/         # Attention mechanisms
│   └── block/             # Model blocks
├── examples/              # Usage examples
├── main.py               # Main CLI interface
└── README.md
```

### Running Examples
```bash
# Basic usage examples
python examples/quick_start.py

# Advanced features demonstration
python examples/advanced_usage.py
```

## 📈 Performance

Typical performance on different hardware:

| Model | Device | Speed (tok/s) | Memory (GB) |
|-------|--------|---------------|-------------|
| 1.7B  | CPU    | 3-8          | 4-6         |
| 1.7B  | GPU    | 30-60        | 8-12        |
| 4B    | GPU    | 15-30        | 16-24       |

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

**Happy generating! 🎉**

For more examples and advanced usage, check the `examples/` directory.