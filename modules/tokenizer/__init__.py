from .qwen_tokenizers import Qwen3Tokenizer

def get_tokenizer(repo_id, tokenizer_file_path, add_generation_prompt=True, add_thinking=False):
    if "qwen3" in repo_id:
        return Qwen3Tokenizer(tokenizer_file_path, repo_id, add_generation_prompt=add_generation_prompt, add_thinking=add_thinking)
    else:
        raise ValueError(f"Tokenizer for {repo_id} not found")