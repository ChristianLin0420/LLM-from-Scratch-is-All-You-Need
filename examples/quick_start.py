#!/usr/bin/env python3
"""
Quick Start Example for LLM from Scratch

This example demonstrates the basic usage of the framework:
1. Loading a model
2. Generating text
3. Interactive chat
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch

from modules import load_qwen3, TextGenerator

def main():
    print("🚀 LLM from Scratch - Quick Start Example")
    print()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Using device: {device}")
    
    # Load model (using 1.7B model for demo)
    print("📦 Loading Qwen3 1.7B model...")
    model, tokenizer, config = load_qwen3(
        size="1.7B",
        variant="base",
        device=device,
        add_generation_prompt=True
    )
    
    # Create text generator
    generator = TextGenerator(model, tokenizer, device)
    
    print("✅ Model loaded successfully!")
    print()
    
    # Example 1: Simple text completion
    print("🔄 Example 1: Text Completion")
    print("-" * 40)
    prompt = "The future of artificial intelligence"
    print(f"Prompt: {prompt}")
    
    completion = generator.complete(
        prompt,
        max_new_tokens=100,
        temperature=0.7
    )
    print(f"Completion: {completion}")
    print()
    
    # Example 2: Chat mode
    print("💬 Example 2: Chat Mode")
    print("-" * 40)
    user_message = "Explain quantum computing in simple terms"
    print(f"User: {user_message}")
    
    response = generator.chat(
        user_message,
        max_new_tokens=150,
        temperature=0.7
    )
    print(f"Assistant: {response}")
    print()
    
    # Example 3: Multiple generations with different temperatures
    print("🎲 Example 3: Temperature Comparison")
    print("-" * 40)
    prompt = "Write a creative story about"
    temperatures = [0.1, 0.7, 1.2]
    
    for temp in temperatures:
        print(f"\nTemperature {temp}:")
        result = generator.complete(
            prompt,
            max_new_tokens=50,
            temperature=temp
        )
        print(f"  {result}")
    
    print()
    print("🎉 Quick start example completed!")
    print("💡 Try running 'python main.py --help' for more options")

if __name__ == "__main__":
    main()