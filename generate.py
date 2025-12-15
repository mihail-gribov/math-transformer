#!/usr/bin/env python3
"""Generation script for MathTransformer."""

import argparse
import torch
from pathlib import Path

from model import MathTransformer


class CharTokenizer:
    """Simple character-level tokenizer for math expressions."""

    def __init__(
        self,
        pad_token: str = "<PAD>",
        eos_token: str = "<EOS>",
        bos_token: str = "<BOS>",
    ):
        # Special tokens
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.bos_token = bos_token

        # Build vocabulary: special tokens + digits + operators + letters
        self.special_tokens = [pad_token, eos_token, bos_token]
        self.chars = list("0123456789+-*/=()., |%><abcdefghijklmnopqrstuvwxyz\n")

        self.vocab = self.special_tokens + self.chars
        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}
        self.id_to_token = {i: tok for i, tok in enumerate(self.vocab)}

        self.pad_token_id = self.token_to_id[pad_token]
        self.eos_token_id = self.token_to_id[eos_token]
        self.bos_token_id = self.token_to_id[bos_token]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> list[int]:
        """Encode text to token ids."""
        ids = []
        if add_bos:
            ids.append(self.bos_token_id)
        for char in text:
            if char in self.token_to_id:
                ids.append(self.token_to_id[char])
            # Skip unknown characters
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Decode token ids to text."""
        tokens = []
        for i in ids:
            if i in self.id_to_token:
                tok = self.id_to_token[i]
                if skip_special and tok in self.special_tokens:
                    continue
                tokens.append(tok)
        return "".join(tokens)


def load_model(
    checkpoint_path: Path | None,
    device: torch.device,
    tokenizer: CharTokenizer,
) -> MathTransformer:
    """Load model from checkpoint or create new one."""
    model = MathTransformer(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )

    if checkpoint_path and checkpoint_path.exists():
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print("Using randomly initialized model (no checkpoint)")

    return model.to(device).eval()


def generate(
    model: MathTransformer,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_k: int | None = None,
    device: torch.device = torch.device("cpu"),
) -> str:
    """Generate text from prompt."""
    # Encode prompt
    input_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    # Decode only the generated part (skip input)
    generated_ids = output_ids[0, len(input_ids):].tolist()
    return tokenizer.decode(generated_ids, skip_special=True)


def main():
    parser = argparse.ArgumentParser(description="Generate with MathTransformer")
    parser.add_argument("prompt", nargs="?", help="Input prompt")
    parser.add_argument("-c", "--checkpoint", type=Path, help="Model checkpoint path")
    parser.add_argument("-n", "--max-tokens", type=int, default=64, help="Max new tokens")
    parser.add_argument("-t", "--temperature", type=float, default=0.8, help="Temperature")
    parser.add_argument("-k", "--top-k", type=int, default=None, help="Top-k sampling")
    parser.add_argument("--device", default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Setup tokenizer and model
    tokenizer = CharTokenizer()
    model = load_model(args.checkpoint, device, tokenizer)
    print(f"Vocab size: {tokenizer.vocab_size}")
    print()

    def run_generation(prompt: str) -> None:
        output = generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device,
        )
        print(f"{prompt}{output}")

    if args.interactive:
        print("Interactive mode. Type 'quit' to exit.")
        print("-" * 40)
        while True:
            try:
                prompt = input("Input: ").strip()
                if prompt.lower() == "quit":
                    break
                if not prompt:
                    continue
                run_generation(prompt)
                print()
            except (KeyboardInterrupt, EOFError):
                break
        print("\nBye!")
    elif args.prompt:
        run_generation(args.prompt)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
