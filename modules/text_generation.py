"""
Text generation utilities for LLM from Scratch project.
Provides flexible text generation with various sampling strategies.
"""
import torch
import torch.nn.functional as F
from typing import List, Optional, Union, Dict, Any
import time


class TextGenerator:
    """
    Flexible text generator with multiple sampling strategies.
    """
    
    def __init__(self, model, tokenizer, device: str = "cpu"):
        """
        Initialize text generator.
        
        Args:
            model: The language model
            tokenizer: The tokenizer
            device: Device to run generation on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
        early_stopping: bool = False,
        num_return_sequences: int = 1,
        use_cache: bool = True,
        apply_chat_template: bool = True,
        stream: bool = False,
        **kwargs
    ) -> Union[str, List[str], Dict[str, Any]]:
        """
        Generate text from a given prompt.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling or greedy decoding
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
            repetition_penalty: Penalty for repeating tokens
            length_penalty: Penalty for sequence length
            early_stopping: Whether to stop early on EOS
            num_return_sequences: Number of sequences to return
            use_cache: Whether to use KV cache (not implemented yet)
            apply_chat_template: Whether to apply chat template
            stream: Whether to return streaming generator
            **kwargs: Additional arguments
            
        Returns:
            Generated text string, list of strings, or streaming generator
        """
        # Set default token IDs if not provided
        if pad_token_id is None:
            pad_token_id = getattr(self.tokenizer, 'pad_token_id', self.tokenizer.eos_token_id)
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id
        
        # Encode input
        input_ids = self.tokenizer.encode(prompt, chat_wrapped=apply_chat_template)
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        if stream:
            return self._generate_streaming(
                input_ids, max_new_tokens, temperature, top_k, top_p,
                do_sample, eos_token_id, repetition_penalty
            )
        else:
            return self._generate_batch(
                input_ids, max_new_tokens, temperature, top_k, top_p,
                do_sample, eos_token_id, repetition_penalty, num_return_sequences
            )
    
    def _generate_streaming(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        do_sample: bool,
        eos_token_id: int,
        repetition_penalty: float
    ):
        """Generator function for streaming text generation."""
        current_ids = input_ids.clone()
        original_length = current_ids.size(1)
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get model predictions
                logits = self.model(current_ids)
                next_token_logits = logits[0, -1, :]
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    next_token_logits = self._apply_repetition_penalty(
                        next_token_logits, current_ids[0], repetition_penalty
                    )
                
                # Sample next token
                if do_sample:
                    next_token_id = self._sample_token(
                        next_token_logits, temperature, top_k, top_p
                    )
                else:
                    next_token_id = torch.argmax(next_token_logits, dim=-1)
                
                # Add to sequence
                current_ids = torch.cat([current_ids, next_token_id.unsqueeze(0).unsqueeze(0)], dim=1)
                
                # Decode the new token
                new_token = self.tokenizer.decode([next_token_id.item()])
                yield new_token
                
                # Check for early stopping
                if next_token_id.item() == eos_token_id:
                    break
    
    def _generate_batch(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        do_sample: bool,
        eos_token_id: int,
        repetition_penalty: float,
        num_return_sequences: int
    ) -> Union[str, List[str]]:
        """Generate text in batch mode."""
        # Expand input for multiple sequences if needed
        if num_return_sequences > 1:
            input_ids = input_ids.repeat(num_return_sequences, 1)
        
        current_ids = input_ids.clone()
        original_length = current_ids.size(1)
        finished = torch.zeros(current_ids.size(0), dtype=torch.bool, device=self.device)
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get model predictions
                logits = self.model(current_ids)
                next_token_logits = logits[:, -1, :]
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for i in range(current_ids.size(0)):
                        if not finished[i]:
                            next_token_logits[i] = self._apply_repetition_penalty(
                                next_token_logits[i], current_ids[i], repetition_penalty
                            )
                
                # Sample next tokens
                if do_sample:
                    next_token_ids = torch.stack([
                        self._sample_token(next_token_logits[i], temperature, top_k, top_p)
                        if not finished[i] else torch.tensor(eos_token_id, device=self.device)
                        for i in range(current_ids.size(0))
                    ])
                else:
                    next_token_ids = torch.argmax(next_token_logits, dim=-1)
                    next_token_ids[finished] = eos_token_id
                
                # Add to sequences
                current_ids = torch.cat([current_ids, next_token_ids.unsqueeze(1)], dim=1)
                
                # Update finished status
                finished |= (next_token_ids == eos_token_id)
                
                # Check if all sequences are finished
                if finished.all():
                    break
        
        # Decode generated sequences
        generated_sequences = []
        for i in range(current_ids.size(0)):
            # Extract only the newly generated tokens
            new_tokens = current_ids[i, original_length:].tolist()
            # Remove tokens after EOS if present
            if eos_token_id in new_tokens:
                eos_idx = new_tokens.index(eos_token_id)
                new_tokens = new_tokens[:eos_idx]
            
            generated_text = self.tokenizer.decode(new_tokens)
            generated_sequences.append(generated_text)
        
        return generated_sequences[0] if num_return_sequences == 1 else generated_sequences
    
    def _sample_token(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float]
    ) -> torch.Tensor:
        """Sample a token from logits using various strategies."""
        if temperature <= 0:
            return torch.argmax(logits, dim=-1)
        
        # Apply temperature
        logits = logits / temperature
        
        # Apply top-k filtering
        if top_k is not None and top_k > 0:
            top_k = min(top_k, logits.size(-1))
            top_k_values, top_k_indices = torch.topk(logits, top_k)
            logits = torch.full_like(logits, float('-inf'))
            logits.scatter_(-1, top_k_indices, top_k_values)
        
        # Apply top-p (nucleus) filtering
        if top_p is not None and 0 < top_p < 1:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            
            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
        
        # Sample from the filtered distribution
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        penalty: float
    ) -> torch.Tensor:
        """Apply repetition penalty to logits."""
        if penalty == 1.0:
            return logits
        
        # Get unique tokens in the input
        unique_ids = torch.unique(input_ids)
        
        # Apply penalty
        for token_id in unique_ids:
            if logits[token_id] > 0:
                logits[token_id] /= penalty
            else:
                logits[token_id] *= penalty
        
        return logits
    
    def chat(
        self,
        message: str,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Simple chat interface.
        
        Args:
            message: User message
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation arguments
            
        Returns:
            Assistant response
        """
        return self.generate(
            message,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            apply_chat_template=True,
            **kwargs
        )
    
    def complete(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        """
        Simple text completion interface.
        
        Args:
            prompt: Text prompt to complete
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            **kwargs: Additional generation arguments
            
        Returns:
            Completed text
        """
        return self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            apply_chat_template=False,
            do_sample=temperature > 0,
            **kwargs
        )


def benchmark_generation(generator: TextGenerator, prompt: str, num_tokens: int = 100):
    """
    Benchmark text generation speed.
    
    Args:
        generator: TextGenerator instance
        prompt: Test prompt
        num_tokens: Number of tokens to generate
        
    Returns:
        Dictionary with benchmark results
    """
    start_time = time.time()
    
    # Generate text
    result = generator.generate(
        prompt,
        max_new_tokens=num_tokens,
        temperature=0.0,  # Use greedy for consistency
        apply_chat_template=False
    )
    
    end_time = time.time()
    
    # Calculate metrics
    total_time = end_time - start_time
    tokens_per_second = num_tokens / total_time
    
    return {
        "total_time": total_time,
        "tokens_generated": num_tokens,
        "tokens_per_second": tokens_per_second,
        "generated_text": result
    }