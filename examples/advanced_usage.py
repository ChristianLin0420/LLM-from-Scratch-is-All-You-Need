#!/usr/bin/env python3
"""
Advanced Usage Examples for LLM from Scratch

Demonstrates advanced features like:
1. Custom sampling parameters
2. Streaming generation
3. Batch generation
4. Model comparison
5. Performance benchmarking
"""

import torch
import time
from modules import load_pretrained_model, TextGenerator, benchmark_generation

def streaming_example():
    """Demonstrate streaming text generation."""
    print("🌊 Streaming Generation Example")
    print("=" * 50)
    
    # Load model
    model, tokenizer, config = load_pretrained_model(
        "qwen3", "1.7B", "base",
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    generator = TextGenerator(model, tokenizer, model.device)
    
    prompt = "Write a story about a magical forest where"
    print(f"Prompt: {prompt}")
    print("\nStreaming output:")
    print("-" * 30)
    
    # Stream the generation token by token
    for token in generator.generate(
        prompt,
        max_new_tokens=100,
        temperature=0.8,
        stream=True,
        apply_chat_template=False
    ):
        print(token, end='', flush=True)
        time.sleep(0.05)  # Simulate real-time streaming
    
    print("\n")

def sampling_comparison():
    """Compare different sampling strategies."""
    print("🎯 Sampling Strategy Comparison")
    print("=" * 50)
    
    # Load model
    model, tokenizer, config = load_pretrained_model(
        "qwen3", "1.7B", "base",
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    generator = TextGenerator(model, tokenizer, model.device)
    
    prompt = "The secret to happiness is"
    
    # Different sampling configurations
    configs = [
        {"name": "Greedy", "temperature": 0.0, "top_k": None, "top_p": None},
        {"name": "Low Temp", "temperature": 0.3, "top_k": None, "top_p": None},
        {"name": "Medium Temp", "temperature": 0.7, "top_k": None, "top_p": None},
        {"name": "High Temp", "temperature": 1.2, "top_k": None, "top_p": None},
        {"name": "Top-K", "temperature": 0.8, "top_k": 20, "top_p": None},
        {"name": "Top-P", "temperature": 0.8, "top_k": None, "top_p": 0.9},
        {"name": "Combined", "temperature": 0.8, "top_k": 40, "top_p": 0.85}
    ]
    
    print(f"Prompt: '{prompt}'\n")
    
    for config in configs:
        result = generator.complete(
            prompt,
            max_new_tokens=50,
            temperature=config["temperature"],
            top_k=config["top_k"],
            top_p=config["top_p"]
        )
        print(f"{config['name']:12}: {result}")
        print()

def batch_generation_example():
    """Demonstrate generating multiple sequences."""
    print("📦 Batch Generation Example")
    print("=" * 50)
    
    # Load model
    model, tokenizer, config = load_pretrained_model(
        "qwen3", "1.7B", "base",
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    generator = TextGenerator(model, tokenizer, model.device)
    
    prompt = "A creative idea for a mobile app:"
    
    # Generate multiple diverse responses
    results = generator.generate(
        prompt,
        max_new_tokens=80,
        temperature=0.9,
        num_return_sequences=3,
        apply_chat_template=False
    )
    
    print(f"Prompt: '{prompt}'\n")
    
    for i, result in enumerate(results, 1):
        print(f"Idea {i}: {result}")
        print()

def performance_benchmark():
    """Run comprehensive performance benchmarks."""
    print("⚡ Performance Benchmark")
    print("=" * 50)
    
    # Load model
    model, tokenizer, config = load_pretrained_model(
        "qwen3", "1.7B", "base",
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    generator = TextGenerator(model, tokenizer, model.device)
    
    # Test different sequence lengths
    test_cases = [
        {"prompt": "Artificial intelligence", "tokens": 25},
        {"prompt": "The history of computers", "tokens": 50},
        {"prompt": "Climate change and renewable energy", "tokens": 100},
        {"prompt": "Space exploration and the future", "tokens": 200}
    ]
    
    print(f"Device: {generator.device}")
    print(f"Model: Qwen3 1.7B")
    print()
    
    total_tokens = 0
    total_time = 0
    
    for i, case in enumerate(test_cases, 1):
        print(f"Test {i}: {case['tokens']} tokens")
        result = benchmark_generation(
            generator, 
            case["prompt"], 
            case["tokens"]
        )
        
        print(f"  Speed: {result['tokens_per_second']:.2f} tok/s")
        print(f"  Time:  {result['total_time']:.3f}s")
        
        total_tokens += result['tokens_generated']
        total_time += result['total_time']
        print()
    
    avg_speed = total_tokens / total_time
    print(f"📊 Overall Average: {avg_speed:.2f} tokens/second")
    print(f"📈 Total Tokens: {total_tokens}")
    print(f"⏱️  Total Time: {total_time:.2f}s")

def main():
    """Run all advanced examples."""
    print("🚀 LLM from Scratch - Advanced Usage Examples")
    print("=" * 60)
    print()
    
    examples = [
        ("Streaming Generation", streaming_example),
        ("Sampling Comparison", sampling_comparison),
        ("Batch Generation", batch_generation_example),
        ("Performance Benchmark", performance_benchmark)
    ]
    
    for name, func in examples:
        try:
            print(f"\n🔍 Running: {name}")
            func()
            print("✅ Completed successfully!")
        except Exception as e:
            print(f"❌ Error in {name}: {e}")
        
        print("\n" + "="*60 + "\n")
    
    print("🎉 All advanced examples completed!")

if __name__ == "__main__":
    main()