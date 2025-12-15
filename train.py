#!/usr/bin/env python3
"""Training script for MathTransformer."""

import argparse
import json
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from model import MathTransformer
from generate import CharTokenizer


@dataclass
class Example:
    """Single training example with token positions."""

    input_ids: list[int]
    # Token index ranges (start, end) for loss weighting
    context_range: tuple[int, int]  # No loss
    reasoning_range: tuple[int, int]  # weight_reasoning
    answer_range: tuple[int, int]  # weight_answer
    format_positions: list[int]  # weight_format (delimiter tokens)


class MathDataset(Dataset):
    """
    Dataset for math examples from JSONL.

    JSONL format:
    {"context": "4+3", "reasoning": "four plus three", "answer": "7"}

    Produces: <BOS>context\n<w>reasoning</w><a>answer</a><EOS>
    """

    def __init__(
        self,
        data_path: Path,
        tokenizer: CharTokenizer,
        max_seq_len: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples: list[Example] = []

        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                example = self._process_example(data)
                if example:
                    self.examples.append(example)

    def _process_example(self, data: dict) -> Example | None:
        """Process single example and track positions."""
        context = data.get("context", "")
        reasoning = data.get("reasoning", "")
        answer = data.get("answer", "")

        # Build parts and track positions
        # Format: <BOS>context\n<w>reasoning</w><a>answer</a><EOS>
        parts = []
        positions = {}
        format_positions = []  # Track format/delimiter token positions

        # BOS
        parts.append(self.tokenizer.bos_token_id)
        pos = 1

        # Context (no loss)
        context_ids = self.tokenizer.encode(context, add_bos=False, add_eos=False)
        positions["context"] = (pos, pos + len(context_ids))
        parts.extend(context_ids)
        pos += len(context_ids)

        # Newline + <w> (format tokens)
        for char in "\n<w>":
            if char in self.tokenizer.token_to_id:
                parts.append(self.tokenizer.token_to_id[char])
                format_positions.append(pos)
                pos += 1

        # Reasoning (weight_reasoning)
        reasoning_start = pos
        reasoning_ids = self.tokenizer.encode(reasoning, add_bos=False, add_eos=False)
        parts.extend(reasoning_ids)
        pos += len(reasoning_ids)

        # </w> (format tokens)
        for char in "</w>":
            if char in self.tokenizer.token_to_id:
                parts.append(self.tokenizer.token_to_id[char])
                format_positions.append(pos)
                pos += 1
        positions["reasoning"] = (reasoning_start, pos)

        # <a> (format tokens)
        for char in "<a>":
            if char in self.tokenizer.token_to_id:
                parts.append(self.tokenizer.token_to_id[char])
                format_positions.append(pos)
                pos += 1

        # Answer (weight_answer)
        answer_start = pos
        answer_ids = self.tokenizer.encode(answer, add_bos=False, add_eos=False)
        parts.extend(answer_ids)
        pos += len(answer_ids)

        # </a> (format tokens)
        for char in "</a>":
            if char in self.tokenizer.token_to_id:
                parts.append(self.tokenizer.token_to_id[char])
                format_positions.append(pos)
                pos += 1
        positions["answer"] = (answer_start, pos)

        # EOS (format token - model must learn to stop)
        parts.append(self.tokenizer.eos_token_id)
        format_positions.append(pos)

        # Truncate if needed
        if len(parts) > self.max_seq_len:
            return None  # Skip too long examples

        return Example(
            input_ids=parts,
            context_range=positions["context"],
            reasoning_range=positions["reasoning"],
            answer_range=positions["answer"],
            format_positions=format_positions,
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Example:
        return self.examples[idx]


def collate_fn(batch: list[Example], pad_token_id: int) -> dict:
    """Collate batch with padding."""
    max_len = max(len(ex.input_ids) for ex in batch)

    input_ids = []
    reasoning_ranges = []
    answer_ranges = []
    format_positions = []

    for ex in batch:
        # Pad input_ids
        padding = [pad_token_id] * (max_len - len(ex.input_ids))
        input_ids.append(ex.input_ids + padding)
        reasoning_ranges.append(ex.reasoning_range)
        answer_ranges.append(ex.answer_range)
        format_positions.append(ex.format_positions)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "reasoning_ranges": reasoning_ranges,
        "answer_ranges": answer_ranges,
        "format_positions": format_positions,
    }


def create_loss_weights(
    seq_len: int,
    reasoning_ranges: list[tuple[int, int]],
    answer_ranges: list[tuple[int, int]],
    format_positions: list[list[int]],
    weight_reasoning: float,
    weight_answer: float,
    weight_format: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Create per-token loss weights based on regions.

    For targets (shifted by 1 from inputs):
    - Context: weight 0
    - Reasoning: weight_reasoning
    - Answer: weight_answer
    - Format tokens: weight_format
    - PAD: weight 0
    """
    batch_size = len(reasoning_ranges)
    weights = torch.zeros(batch_size, seq_len, device=device)

    for b in range(batch_size):
        r_start, r_end = reasoning_ranges[b]
        a_start, a_end = answer_ranges[b]

        # Shift ranges by -1 for target alignment (predict next token)
        # Token at position i in input predicts token at position i+1
        # So weight for target[i] should be based on what position i+1 is
        r_start = max(0, r_start - 1)
        r_end = min(seq_len, r_end - 1)
        a_start = max(0, a_start - 1)
        a_end = min(seq_len, a_end - 1)

        if r_start < r_end:
            weights[b, r_start:r_end] = weight_reasoning
        if a_start < a_end:
            weights[b, a_start:a_end] = weight_answer

        # Format tokens (applied after ranges, so they override)
        for pos in format_positions[b]:
            target_pos = pos - 1  # Shift for target alignment
            if 0 <= target_pos < seq_len:
                weights[b, target_pos] = weight_format

    return weights


def train_step(
    model: MathTransformer,
    batch: dict,
    weight_reasoning: float,
    weight_answer: float,
    weight_format: float,
    device: torch.device,
) -> torch.Tensor:
    """Single training step with weighted loss."""
    input_ids = batch["input_ids"].to(device)

    # Shift for next-token prediction
    inputs = input_ids[:, :-1]
    targets = input_ids[:, 1:]

    # Forward pass
    logits = model(inputs)

    # Create loss weights
    loss_weights = create_loss_weights(
        targets.size(1),
        batch["reasoning_ranges"],
        batch["answer_ranges"],
        batch["format_positions"],
        weight_reasoning,
        weight_answer,
        weight_format,
        device,
    )

    # Compute weighted cross-entropy loss
    logits_flat = logits.reshape(-1, logits.size(-1))
    targets_flat = targets.reshape(-1)
    weights_flat = loss_weights.reshape(-1)

    # Per-token loss
    loss_per_token = F.cross_entropy(logits_flat, targets_flat, reduction="none")

    # Weighted mean
    weighted_loss = (loss_per_token * weights_flat).sum()
    total_weight = weights_flat.sum()

    if total_weight > 0:
        loss = weighted_loss / total_weight
    else:
        loss = weighted_loss

    return loss


def main():
    parser = argparse.ArgumentParser(description="Train MathTransformer")
    parser.add_argument("data", type=Path, help="Training data path (JSONL)")
    parser.add_argument("-c", "--checkpoint", type=Path, help="Resume from checkpoint")
    parser.add_argument("-o", "--output", type=Path, default=Path("checkpoints/model.pt"), help="Output path")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-reasoning", type=float, default=0.5, help="Loss weight for reasoning")
    parser.add_argument("--weight-answer", type=float, default=1.0, help="Loss weight for answer")
    parser.add_argument("--weight-format", type=float, default=2.0, help="Loss weight for format tokens")
    parser.add_argument("--device", default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N steps")
    parser.add_argument("--save-interval", type=int, default=1000, help="Save every N steps")
    args = parser.parse_args()

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Tokenizer
    tokenizer = CharTokenizer()
    print(f"Vocab size: {tokenizer.vocab_size}")

    # Dataset
    dataset = MathDataset(args.data, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )
    print(f"Dataset size: {len(dataset)}")

    # Model
    model = MathTransformer(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
    ).to(device)

    if args.checkpoint and args.checkpoint.exists():
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
        print(f"Loaded checkpoint: {args.checkpoint}")

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Training loop
    print(f"\nTraining config:")
    print(f"  weight_reasoning: {args.weight_reasoning}")
    print(f"  weight_answer: {args.weight_answer}")
    print(f"  weight_format: {args.weight_format}")
    print()

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()

            loss = train_step(
                model,
                batch,
                args.weight_reasoning,
                args.weight_answer,
                args.weight_format,
                device,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            global_step += 1

            if global_step % args.log_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"Epoch {epoch+1} | Step {global_step} | Loss: {avg_loss:.4f}")

            if global_step % args.save_interval == 0:
                torch.save(model.state_dict(), args.output)
                print(f"Saved checkpoint: {args.output}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} complete | Avg loss: {avg_loss:.4f}")

        # Save after each epoch
        torch.save(model.state_dict(), args.output)

    print(f"\nTraining complete. Model saved to {args.output}")


if __name__ == "__main__":
    main()
