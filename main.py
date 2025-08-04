#!/usr/bin/env python3
"""
LLM from Scratch - Main Application

A flexible framework for loading and running large language models.
Supports multiple model types, sizes, and generation strategies.

Usage:
    python main.py --model qwen3 --size 1.7B --prompt "Hello, how are you?"
    python main.py --chat  # Interactive chat mode
    python main.py --benchmark  # Run benchmarks
"""

import argparse
import torch
import sys
from pathlib import Path

# Set random seed for reproducibility
torch.manual_seed(42)

# Import our modules
from modules.load_model import load_pretrained_model, ModelRegistry
from modules.text_generation import TextGenerator, benchmark_generation


def main():
    parser = argparse.ArgumentParser(
        description="LLM from Scratch - Flexible Language Model Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --model qwen3 --size 1.7B --prompt "Explain quantum computing"
  python main.py --model qwen3 --size 1.7B --chat
  python main.py --model qwen3 --size 1.7B --benchmark
  python main.py --model qwen3 --size 4B --variant instruct --interactive
        """
    )
    
    # Model configuration
    parser.add_argument('--model', type=str, default='qwen3', 
                       choices=list(ModelRegistry.SUPPORTED_MODELS.keys()),
                       help='Model type to load')
    parser.add_argument('--size', type=str, default='1.7B',
                       choices=['1.7B', '4B', '8B', '14B', '32B', '30B_A3B'],
                       help='Model size')
    parser.add_argument('--variant', type=str, default='base',
                       choices=['base', 'instruct', 'reasoning'],
                       help='Model variant')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to custom config file')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to run on (auto, cpu, cuda)')
    
    # Generation parameters
    parser.add_argument('--max-tokens', type=int, default=200,
                       help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=None,
                       help='Top-k sampling parameter')
    parser.add_argument('--top-p', type=float, default=None,
                       help='Top-p (nucleus) sampling parameter')
    parser.add_argument('--repetition-penalty', type=float, default=1.0,
                       help='Repetition penalty')
    
    # Modes of operation
    parser.add_argument('--prompt', type=str, default=None,
                       help='Single prompt to generate from')
    parser.add_argument('--chat', action='store_true',
                       help='Start interactive chat mode')
    parser.add_argument('--interactive', action='store_true',
                       help='Start interactive completion mode')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run generation benchmarks')
    parser.add_argument('--stream', action='store_true',
                       help='Enable streaming output')
    
    # Tokenizer options
    parser.add_argument('--add-generation-prompt', action='store_true',
                       help='Add generation prompt for chat')
    parser.add_argument('--add-thinking', action='store_true',
                       help='Add thinking tags')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"🚀 LLM from Scratch - Loading {args.model.upper()} {args.size} ({args.variant})")
    print(f"🖥️  Device: {device}")
    print()
    
    try:
        # Load model, tokenizer, and config
        model, tokenizer, config = load_pretrained_model(
            model_type=args.model,
            size=args.size,
            variant=args.variant,
            config_path=args.config,
            device=device,
            add_generation_prompt=args.add_generation_prompt,
            add_thinking=args.add_thinking
        )
        
        # Create text generator
        generator = TextGenerator(model, tokenizer, device)
        
        # Execute based on mode
        if args.benchmark:
            run_benchmark(generator)
        elif args.chat:
            run_chat_mode(generator, args)
        elif args.interactive:
            run_interactive_mode(generator, args)
        elif args.prompt:
            run_single_prompt(generator, args)
        else:
            # Default to interactive mode if no specific mode chosen
            print("No mode specified. Starting interactive completion mode.")
            print("Use --help for more options.")
            print()
            run_interactive_mode(generator, args)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def run_single_prompt(generator, args):
    """Run a single prompt and output the result."""
    print(f"📝 Prompt: {args.prompt}")
    print("🤖 Generating...")
    print()
    
    if args.stream:
        print("📜 Generated text (streaming):")
        print("-" * 50)
        for token in generator.generate(
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            stream=True
        ):
            print(token, end='', flush=True)
        print()
    else:
        result = generator.generate(
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty
        )
        print("📜 Generated text:")
        print("-" * 50)
        print(result)


def run_chat_mode(generator, args):
    """Run interactive chat mode."""
    print("💬 Chat Mode - Type 'quit' or 'exit' to end")
    print("📝 Settings:")
    print(f"   Max tokens: {args.max_tokens}")
    print(f"   Temperature: {args.temperature}")
    print(f"   Top-k: {args.top_k}")
    print(f"   Top-p: {args.top_p}")
    print()
    
    while True:
        try:
            user_input = input("🧑 You: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            print("🤖 Assistant: ", end='')
            
            if args.stream:
                for token in generator.generate(
                    user_input,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    apply_chat_template=True,
                    stream=True
                ):
                    print(token, end='', flush=True)
                print()  # New line after streaming
            else:
                response = generator.chat(
                    user_input,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty
                )
                print(response)
            
            print()  # Add spacing between exchanges
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except EOFError:
            print("\n👋 Goodbye!")
            break


def run_interactive_mode(generator, args):
    """Run interactive text completion mode."""
    print("📝 Interactive Mode - Type 'quit' or 'exit' to end")
    print("💡 This mode completes your text without chat formatting")
    print("📝 Settings:")
    print(f"   Max tokens: {args.max_tokens}")
    print(f"   Temperature: {args.temperature}")
    print(f"   Top-k: {args.top_k}")
    print(f"   Top-p: {args.top_p}")
    print()
    
    while True:
        try:
            user_input = input("📝 Prompt: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            print("📜 Completion: ", end='')
            
            if args.stream:
                for token in generator.generate(
                    user_input,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    apply_chat_template=False,
                    stream=True
                ):
                    print(token, end='', flush=True)
                print()  # New line after streaming
            else:
                completion = generator.complete(
                    user_input,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty
                )
                print(completion)
            
            print()  # Add spacing between completions
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except EOFError:
            print("\n👋 Goodbye!")
            break


def run_benchmark(generator):
    """Run generation benchmarks."""
    print("🏃 Running Benchmarks...")
    print()
    
    test_prompts = [
        "The future of artificial intelligence",
        "Explain the theory of relativity",
        "Write a short story about a robot",
        "What are the benefits of renewable energy?"
    ]
    
    token_counts = [50, 100, 200]
    
    print("📊 Benchmark Results:")
    print("=" * 70)
    
    for prompt in test_prompts:
        print(f"\n📝 Prompt: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
        print("-" * 50)
        
        for num_tokens in token_counts:
            result = benchmark_generation(generator, prompt, num_tokens)
            print(f"  {num_tokens:3d} tokens: {result['tokens_per_second']:.2f} tok/s ({result['total_time']:.2f}s)")
    
    print("\n🎯 Overall Performance:")
    overall_result = benchmark_generation(generator, "The quick brown fox", 100)
    print(f"   Average Speed: {overall_result['tokens_per_second']:.2f} tokens/second")
    print(f"   Device: {generator.device}")


if __name__ == "__main__":
    main()