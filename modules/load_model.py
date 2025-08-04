"""
Flexible model loading system for LLM from Scratch project.
Supports loading models, configurations, tokenizers, and pretrained weights.
"""
import json
import os
import yaml
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Union
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download, snapshot_download

from .llm import Qwen3
from .tokenizer import get_tokenizer


class ModelRegistry:
    """Registry for supported models and their configurations."""
    
    SUPPORTED_MODELS = {
        "qwen3": {
            "class": Qwen3,
            "weight_mapping": "qwen3_mapping",
            "repo_patterns": {
                "base": "Qwen/Qwen3-{size}-Base",
                "instruct": "Qwen/Qwen3-{size}",
                "reasoning": "Qwen/Qwen3-{size}"
            }
        }
    }
    
    @classmethod
    def get_model_class(cls, model_type: str):
        """Get the model class for a given model type."""
        if model_type not in cls.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type: {model_type}. Supported: {list(cls.SUPPORTED_MODELS.keys())}")
        return cls.SUPPORTED_MODELS[model_type]["class"]
    
    @classmethod
    def get_weight_mapping(cls, model_type: str):
        """Get the weight mapping function for a given model type."""
        if model_type not in cls.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type: {model_type}")
        return cls.SUPPORTED_MODELS[model_type]["weight_mapping"]
    
    @classmethod
    def get_repo_id(cls, model_type: str, size: str, variant: str = "base"):
        """Generate repository ID for a model."""
        if model_type not in cls.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        pattern = cls.SUPPORTED_MODELS[model_type]["repo_patterns"].get(variant)
        if not pattern:
            raise ValueError(f"Unsupported variant: {variant}")
        
        return pattern.format(size=size)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load model configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert dtype string to torch dtype
    if 'dtype' in config and isinstance(config['dtype'], str):
        dtype_map = {
            'float32': torch.float32,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
            'int8': torch.int8,
            'int32': torch.int32,
        }
        config['dtype'] = dtype_map.get(config['dtype'], torch.float32)
    
    return config


def load_model(config: Dict[str, Any], model_type: str = "qwen3") -> torch.nn.Module:
    """
    Load a model based on configuration.
    
    Args:
        config: Model configuration dictionary
        model_type: Type of model to load
        
    Returns:
        Initialized model
    """
    model_class = ModelRegistry.get_model_class(model_type)
    model = model_class(config)
    return model


def load_tokenizer(repo_id: str, local_dir: Optional[str] = None, **kwargs):
    """
    Load tokenizer for a given model repository.
    
    Args:
        repo_id: HuggingFace repository ID
        local_dir: Local directory where model is stored
        **kwargs: Additional arguments for tokenizer
        
    Returns:
        Initialized tokenizer
    """
    if local_dir:
        tokenizer_path = Path(local_dir) / "tokenizer.json"
    else:
        # Download tokenizer file
        tokenizer_path = hf_hub_download(
            repo_id=repo_id,
            filename="tokenizer.json",
            local_dir=repo_id.split('/')[-1] if not local_dir else local_dir
        )
    
    return get_tokenizer(repo_id.lower(), tokenizer_path, **kwargs)


def load_weights(repo_id: str, model_size: str, local_dir: Optional[str] = None) -> Dict[str, torch.Tensor]:
    """
    Load pretrained weights from HuggingFace repository.
    
    Args:
        repo_id: HuggingFace repository ID
        model_size: Size of the model (e.g., "1.7B", "4B")
        local_dir: Optional local directory to save/load from
        
    Returns:
        Dictionary of model weights
    """
    if not local_dir:
        local_dir = Path(repo_id).parts[-1]
    else:
        local_dir = Path(local_dir)
    
    # Load model weights from HuggingFace repository
    if model_size in ["1.7B"]:
        # Single file for smaller models
        weights_file = hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors",
            local_dir=local_dir,
        )
        weights_dict = load_file(weights_file)
    else:
        # Multiple files for larger models
        repo_dir = snapshot_download(repo_id=repo_id, local_dir=local_dir)
        index_path = os.path.join(repo_dir, "model.safetensors.index.json")
        
        with open(index_path, "r") as f:
            index = json.load(f)
        
        weights_dict = {}
        for filename in set(index["weight_map"].values()):
            shard_path = os.path.join(repo_dir, filename)
            shard = load_file(shard_path)
            weights_dict.update(shard)
    
    return weights_dict


def load_weights_into_qwen3(model, config, params):
    """
    Load pretrained weights into Qwen3 model with proper mapping.
    
    Args:
        model: Qwen3 model instance
        config: Model configuration
        params: Dictionary of pretrained parameters
    """
    def assign(left, right, tensor_name="unknown"):
        """Safely assign weights with shape validation."""
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}")
        return torch.nn.Parameter(right.clone().detach() if isinstance(right, torch.Tensor) else torch.tensor(right))
    
    # Token embedding
    model.token_embedding.weight = assign(
        model.token_embedding.weight, 
        params["model.embed_tokens.weight"], 
        "model.embed_tokens.weight"
    )
    
    # Process each transformer block
    for l in range(config["num_layers"]):
        block = model.blocks[l]
        att = block.attention
        
        # Attention projections
        att.q_proj.weight = assign(
            att.q_proj.weight,
            params[f"model.layers.{l}.self_attn.q_proj.weight"],
            f"model.layers.{l}.self_attn.q_proj.weight"
        )
        att.k_proj.weight = assign(
            att.k_proj.weight,
            params[f"model.layers.{l}.self_attn.k_proj.weight"],
            f"model.layers.{l}.self_attn.k_proj.weight"
        )
        att.v_proj.weight = assign(
            att.v_proj.weight,
            params[f"model.layers.{l}.self_attn.v_proj.weight"],
            f"model.layers.{l}.self_attn.v_proj.weight"
        )
        att.output_proj.weight = assign(
            att.output_proj.weight,
            params[f"model.layers.{l}.self_attn.o_proj.weight"],
            f"model.layers.{l}.self_attn.o_proj.weight"
        )
        
        # QK normalization layers (if enabled)
        if hasattr(att, "q_norm") and att.q_norm is not None:
            att.q_norm.scale = assign(
                att.q_norm.scale,
                params[f"model.layers.{l}.self_attn.q_norm.weight"],
                f"model.layers.{l}.self_attn.q_norm.weight"
            )
        if hasattr(att, "k_norm") and att.k_norm is not None:
            att.k_norm.scale = assign(
                att.k_norm.scale,
                params[f"model.layers.{l}.self_attn.k_norm.weight"],
                f"model.layers.{l}.self_attn.k_norm.weight"
            )
        
        # Layer norms
        block.norm1.scale = assign(
            block.norm1.scale,
            params[f"model.layers.{l}.input_layernorm.weight"],
            f"model.layers.{l}.input_layernorm.weight"
        )
        block.norm2.scale = assign(
            block.norm2.scale,
            params[f"model.layers.{l}.post_attention_layernorm.weight"],
            f"model.layers.{l}.post_attention_layernorm.weight"
        )
        
        # Feed-forward weights
        block.ffn.fc1.weight = assign(
            block.ffn.fc1.weight,
            params[f"model.layers.{l}.mlp.gate_proj.weight"],
            f"model.layers.{l}.mlp.gate_proj.weight"
        )
        block.ffn.fc2.weight = assign(
            block.ffn.fc2.weight,
            params[f"model.layers.{l}.mlp.up_proj.weight"],
            f"model.layers.{l}.mlp.up_proj.weight"
        )
        block.ffn.fc3.weight = assign(
            block.ffn.fc3.weight,
            params[f"model.layers.{l}.mlp.down_proj.weight"],
            f"model.layers.{l}.mlp.down_proj.weight"
        )
    
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
        # Weight tying - reuse embedding weights
        print("Model uses weight tying.")
        model.lm_head.weight = assign(
            model.lm_head.weight, 
            params["model.embed_tokens.weight"], 
            "model.embed_tokens.weight"
        )


# Weight mapping registry
WEIGHT_MAPPINGS = {
    "qwen3_mapping": load_weights_into_qwen3
}


def load_pretrained_model(
    model_type: str = "qwen3",
    size: str = "1.7B",
    variant: str = "base",
    config_path: Optional[str] = None,
    local_dir: Optional[str] = None,
    device: str = "cpu",
    **tokenizer_kwargs
):
    """
    Complete pipeline to load a pretrained model with tokenizer.
    
    Args:
        model_type: Type of model ("qwen3")
        size: Model size ("1.7B", "4B", "8B", etc.)
        variant: Model variant ("base", "instruct", "reasoning")
        config_path: Path to local config file (optional)
        local_dir: Local directory for model files
        device: Device to load model on
        **tokenizer_kwargs: Additional tokenizer arguments
        
    Returns:
        Tuple of (model, tokenizer, config)
    """
    # Generate repository ID
    repo_id = ModelRegistry.get_repo_id(model_type, size, variant)
    
    # Load configuration
    if config_path:
        config = load_config(config_path)
    else:
        # Use default config path
        config_path = f"configs/{model_type}/{model_type}_{size}.yaml"
        config = load_config(config_path)
    
    # Load model
    model = load_model(config, model_type)
    
    # Load tokenizer
    tokenizer = load_tokenizer(repo_id, local_dir, **tokenizer_kwargs)
    
    # Load and apply weights
    weights = load_weights(repo_id, size, local_dir)
    weight_mapping_fn = WEIGHT_MAPPINGS[ModelRegistry.get_weight_mapping(model_type)]
    weight_mapping_fn(model, config, weights)
    
    # Move to device
    model.to(device)
    model.eval()
    
    print(f"✅ Successfully loaded {model_type.upper()} {size} ({variant}) model")
    print(f"📍 Repository: {repo_id}")
    print(f"🖥️  Device: {device}")
    
    return model, tokenizer, config


# Convenience functions for common use cases
def load_qwen3(size: str = "1.7B", variant: str = "base", **kwargs):
    """Convenience function to load Qwen3 model."""
    return load_pretrained_model("qwen3", size, variant, **kwargs)
