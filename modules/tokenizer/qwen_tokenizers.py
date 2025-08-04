import re

from tokenizers import Tokenizer

import re
from pathlib import Path
from tokenizers import Tokenizer

class Qwen3Tokenizer:
    """
    Qwen3Tokenizer provides encoding and decoding functionality for Qwen3 models,
    handling special tokens, chat formatting, and integration with HuggingFace tokenizers.

    Args:
        tokenizer_file_path (str or Path): Path to the tokenizer JSON file.
        repo_id (str): Identifier for the model repository, used to determine EOS token.
        apply_chat_template (bool, optional): Whether to wrap input in chat template. Default: True.
        add_generation_prompt (bool, optional): Whether to add assistant generation prompt. Default: False.
        add_thinking (bool, optional): Whether to add a thinking placeholder in the prompt. Default: False.

    Attributes:
        tokenizer (Tokenizer): The underlying HuggingFace Tokenizer object.
        special_to_id (dict): Mapping from special token string to token id.
        pad_token_id (int): Token id for padding (usually <|endoftext|>).
        eos_token_id (int): Token id for end-of-sequence.
        apply_chat_template (bool): Whether to wrap input in chat template.
        add_generation_prompt (bool): Whether to add assistant generation prompt.
        add_thinking (bool): Whether to add a thinking placeholder in the prompt.
    """

    # List of special tokens used by Qwen3Tokenizer for various purposes
    _SPECIALS = [
        "<|endoftext|>",                                        # End of text
        "<|im_start|>", "<|im_end|>",                           # Instruction/message delimiters
        "<|object_ref_start|>", "<|object_ref_end|>",           # Object reference delimiters
        "<|box_start|>", "<|box_end|>",                         # Box delimiters
        "<|quad_start|>", "<|quad_end|>",                       # Quadrilateral delimiters
        "<|vision_start|>", "<|vision_end|>",                   # Vision input delimiters
        "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",     # Padding tokens for vision, image, and video
    ]

    # Regular expression to split text on special tokens (e.g., <|...|>)
    _SPLIT_RE = re.compile(
        r"(<\|[^>]+?\|>)"
    )

    def __init__(self, tokenizer_file_path, repo_id, apply_chat_template=True, add_generation_prompt=False, add_thinking=False):
        """
        Initialize the Qwen3Tokenizer.

        Args:
            tokenizer_file_path (str or Path): Path to the tokenizer JSON file.
            repo_id (str): Model repository identifier.
            apply_chat_template (bool): Whether to wrap input in chat template.
            add_generation_prompt (bool): Whether to add assistant generation prompt.
            add_thinking (bool): Whether to add a thinking placeholder in the prompt.

        Sets up the tokenizer, special token mappings, and determines EOS/pad token ids.
        """
        self.apply_chat_template = apply_chat_template
        self.add_generation_prompt = add_generation_prompt
        self.add_thinking = add_thinking

        tokenizer_file = Path(tokenizer_file_path)
        self.tokenizer = Tokenizer.from_file(str(tokenizer_file))

        # Map each special token to its corresponding token id in the tokenizer
        self.special_to_id = {t: self.tokenizer.token_to_id(t) for t in self._SPECIALS}

        # Set pad_token_id and eos_token_id based on repo_id and available special tokens
        self.pad_token_id = self.special_to_id.get("<|endoftext|>")
        self.eos_token_id = self.pad_token_id

        if repo_id and "Base" not in repo_id:
            eos_token = "<|im_end|>"
        else:
            eos_token = "<|endoftext|>"

        if eos_token in self.special_to_id:
            self.eos_token_id = self.special_to_id[eos_token]

    def encode(self, text, chat_wrapped=None):
        """
        Encode a string into a list of token ids, handling special tokens and chat formatting.

        Args:
            text (str): The input text to encode.
            chat_wrapped (bool, optional): Whether to wrap the text in a chat template.
                If None, uses self.apply_chat_template.

        Returns:
            List[int]: List of token ids representing the input text.
        """
        if chat_wrapped is None:
            chat_wrapped = self.apply_chat_template

        stripped = text.strip()

        # If the input is a single special token (and not a multi-line string), return its id directly
        if stripped in self.special_to_id and "\n" not in stripped:
            return [self.special_to_id[stripped]]

        # Optionally wrap the text in a chat template
        if chat_wrapped:
            text = self.wrap_chat(text)

        ids = []
        # Split the text on special tokens, preserving them as separate parts
        for part in filter(None, self._SPLIT_RE.split(text)):
            if part in self.special_to_id:
                ids.append(self.special_to_id[part])
            else:
                # Encode normal text using the underlying tokenizer
                ids.extend(self.tokenizer.encode(part).ids)

        return ids

    def decode(self, ids):
        """
        Decode a list of token ids back into a string, including special tokens.

        Args:
            ids (List[int]): List of token ids to decode.

        Returns:
            str: The decoded string, with special tokens preserved.
        """
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def wrap_chat(self, user_msg):
        """
        Wrap a user message in the Qwen3 chat template, optionally adding assistant and thinking prompts.

        Args:
            user_msg (str): The user message to wrap.

        Returns:
            str: The formatted chat string, ready for encoding.
        """
        # Start with user message in chat format
        s = f"<|im_start|>user\n{user_msg}<|im_end|>\n"

        # Optionally add assistant prompt and thinking placeholder
        if self.add_generation_prompt:
            s += f"<|im_start|>assistant\n"
            if self.add_thinking:
                s += f"\n"
            else:
                s += "<think>\n\n</think>\n\n"

        return s
            