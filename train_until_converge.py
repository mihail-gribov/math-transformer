#!/usr/bin/env python3
"""Training script that trains until test loss converges below threshold."""

import argparse
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader

from model import MathTransformer
from train import MathDataset, collate_fn, create_loss_weights
from generate import CharTokenizer


@dataclass
class EvalMetrics:
    """Evaluation metrics."""

    loss_total: float
    loss_reasoning: float  # reasoning only, without </w> tag
    loss_answer: float  # answer only, without </a> tag
    accuracy: float  # Exact match on answer


# Length of closing tags in tokens (</w> = 4 chars, </a> = 4 chars)
CLOSING_TAG_LEN = 4


def compute_separate_losses(
    logits: torch.Tensor,
    targets: torch.Tensor,
    reasoning_ranges: list[tuple[int, int]],
    answer_ranges: list[tuple[int, int]],
) -> tuple[float, float]:
    """
    Compute separate losses for reasoning and answer parts.

    Returns: (loss_reasoning, loss_answer)
    - Excludes closing tags (</w>, </a>) - content only
    """
    batch_size, seq_len = targets.shape

    loss_per_token = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).reshape(batch_size, seq_len)

    reasoning_loss_sum = 0.0
    reasoning_count = 0
    answer_loss_sum = 0.0
    answer_count = 0

    for b in range(batch_size):
        r_start, r_end = reasoning_ranges[b]
        # Shift by -1 because targets are shifted
        r_start_t, r_end_t = max(0, r_start - 1), max(0, r_end - 1)
        # Exclude closing tag </w>
        r_end_t = max(r_start_t, r_end_t - CLOSING_TAG_LEN)

        if r_end_t > r_start_t and r_end_t <= seq_len:
            reasoning_loss_sum += loss_per_token[b, r_start_t:r_end_t].sum().item()
            reasoning_count += r_end_t - r_start_t

        a_start, a_end = answer_ranges[b]
        a_start_t, a_end_t = max(0, a_start - 1), max(0, a_end - 1)
        # Exclude closing tag </a>
        a_end_t = max(a_start_t, a_end_t - CLOSING_TAG_LEN)

        if a_end_t > a_start_t and a_end_t <= seq_len:
            answer_loss_sum += loss_per_token[b, a_start_t:a_end_t].sum().item()
            answer_count += a_end_t - a_start_t

    loss_reasoning = reasoning_loss_sum / reasoning_count if reasoning_count > 0 else 0.0
    loss_answer = answer_loss_sum / answer_count if answer_count > 0 else 0.0

    return loss_reasoning, loss_answer


def compute_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    answer_ranges: list[tuple[int, int]],
) -> float:
    """Compute exact match accuracy on answer part."""
    batch_size, seq_len = targets.shape
    predictions = logits.argmax(dim=-1)

    correct = 0
    total = 0

    for b in range(batch_size):
        a_start, a_end = answer_ranges[b]
        # Shift by -1 because targets are shifted
        a_start_t, a_end_t = max(0, a_start - 1), max(0, a_end - 1)
        if a_end_t > a_start_t and a_end_t <= seq_len:
            pred_answer = predictions[b, a_start_t:a_end_t]
            true_answer = targets[b, a_start_t:a_end_t]
            if torch.equal(pred_answer, true_answer):
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.0


def evaluate(
    model: MathTransformer,
    dataloader: DataLoader,
    weight_reasoning: float,
    weight_answer: float,
    weight_format: float,
    device: torch.device,
) -> EvalMetrics:
    """Evaluate model on dataset, return metrics."""
    model.eval()
    total_loss = 0.0
    total_loss_reasoning = 0.0
    total_loss_answer = 0.0
    total_correct = 0
    total_examples = 0
    total_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]

            logits = model(inputs)

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

            logits_flat = logits.reshape(-1, logits.size(-1))
            targets_flat = targets.reshape(-1)
            weights_flat = loss_weights.reshape(-1)

            loss_per_token = F.cross_entropy(logits_flat, targets_flat, reduction="none")
            weighted_loss = (loss_per_token * weights_flat).sum()
            total_weight = weights_flat.sum()

            if total_weight > 0:
                loss = weighted_loss / total_weight
            else:
                loss = weighted_loss

            total_loss += loss.item()

            # Compute separate losses
            loss_r, loss_a = compute_separate_losses(
                logits, targets, batch["reasoning_ranges"], batch["answer_ranges"]
            )
            total_loss_reasoning += loss_r
            total_loss_answer += loss_a

            # Compute accuracy
            acc = compute_accuracy(logits, targets, batch["answer_ranges"])
            total_correct += acc * len(batch["answer_ranges"])
            total_examples += len(batch["answer_ranges"])

            total_batches += 1

    return EvalMetrics(
        loss_total=total_loss / total_batches if total_batches > 0 else 0.0,
        loss_reasoning=total_loss_reasoning / total_batches if total_batches > 0 else 0.0,
        loss_answer=total_loss_answer / total_batches if total_batches > 0 else 0.0,
        accuracy=total_correct / total_examples if total_examples > 0 else 0.0,
    )


TSV_COLUMNS = [
    "timestamp",
    "epoch",
    "difficulty",
    "max_seq_len",
    "layer_weight_norms",
    "layer_weight_rms_changes",
    "train_loss",
    "train_loss_w",
    "train_loss_a",
    "train_acc",
    "test_loss",
    "test_loss_w",
    "test_loss_a",
    "test_acc",
    "complex_loss",
    "complex_loss_w",
    "complex_loss_a",
    "complex_acc",
]


def log_metrics(
    log_path: Path,
    epoch: int,
    difficulty: int,
    train_metrics: EvalMetrics,
    test_metrics: EvalMetrics | None,
    test_complex_metrics: EvalMetrics | None = None,
    max_seq_len: int = 0,
    layer_weight_norms: str = "",
    layer_weight_rms_changes: str = "",
):
    """Append metrics to TSV log file."""
    # Write header if file doesn't exist
    write_header = not log_path.exists()

    timestamp = datetime.now().isoformat(timespec="seconds")
    values = [
        timestamp,
        epoch,
        difficulty,
        max_seq_len,
        layer_weight_norms,
        layer_weight_rms_changes,
        f"{train_metrics.loss_total:.6f}",
        f"{train_metrics.loss_reasoning:.6f}",
        f"{train_metrics.loss_answer:.6f}",
        f"{train_metrics.accuracy:.4f}",
        f"{test_metrics.loss_total:.6f}" if test_metrics else "",
        f"{test_metrics.loss_reasoning:.6f}" if test_metrics else "",
        f"{test_metrics.loss_answer:.6f}" if test_metrics else "",
        f"{test_metrics.accuracy:.4f}" if test_metrics else "",
        f"{test_complex_metrics.loss_total:.6f}" if test_complex_metrics else "",
        f"{test_complex_metrics.loss_reasoning:.6f}" if test_complex_metrics else "",
        f"{test_complex_metrics.loss_answer:.6f}" if test_complex_metrics else "",
        f"{test_complex_metrics.accuracy:.4f}" if test_complex_metrics else "",
    ]

    with open(log_path, "a") as f:
        if write_header:
            f.write("\t".join(TSV_COLUMNS) + "\n")
        f.write("\t".join(str(v) for v in values) + "\n")


def get_difficulty_files(data_dir: Path, difficulty: int) -> tuple[Path, Path, Path] | None:
    """
    Get train, test, and test-complex files for given difficulty level.

    Returns (train_file, test_file, test_complex_file) or None if train file doesn't exist.
    """
    train_file = data_dir / f"examples-{difficulty}.jsonl"
    test_file = data_dir / f"test-{difficulty}.jsonl"
    test_complex_file = data_dir / f"test-complex-{difficulty}.jsonl"

    if not train_file.exists():
        return None

    return train_file, test_file, test_complex_file


def snapshot_layer_weights(model: MathTransformer) -> dict[int, torch.Tensor]:
    """Create a snapshot of all layer weights (flattened and concatenated)."""
    snapshots = {}
    for i, layer in enumerate(model.layers):
        params = [p.detach().clone().flatten() for p in layer.parameters()]
        if params:
            snapshots[i] = torch.cat(params)
    return snapshots


def compute_layer_weight_norms(model: MathTransformer) -> str:
    """Compute L2 norm of weights for each layer. Format: 'layer_idx:norm,...'"""
    parts = []
    for i, layer in enumerate(model.layers):
        total_sq = 0.0
        total_count = 0
        for p in layer.parameters():
            total_sq += (p.detach() ** 2).sum().item()
            total_count += p.numel()
        if total_count > 0:
            norm = (total_sq / total_count) ** 0.5  # RMS norm
            parts.append(f"{i}:{norm:.4e}")
    return ",".join(parts)


def compute_layer_weight_rms_changes(
    model: MathTransformer,
    prev_snapshots: dict[int, torch.Tensor],
) -> str:
    """Compute RMS of weight changes for each layer. Format: 'layer_idx:rms,...'"""
    parts = []
    for i, layer in enumerate(model.layers):
        if i not in prev_snapshots:
            continue
        params = [p.detach().flatten() for p in layer.parameters()]
        if params:
            current = torch.cat(params)
            prev = prev_snapshots[i].to(current.device)
            diff = current - prev
            rms = (diff ** 2).mean().sqrt().item()
            parts.append(f"{i}:{rms:.4e}")
    return ",".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Train until convergence")
    parser.add_argument("-e", "--experiment", type=str, help="Experiment name (uses data/exp-{name}/ and checkpoints/exp-{name}/)")
    parser.add_argument("--output", type=Path, default=Path("model.pt"), help="Model output filename")
    parser.add_argument("--log", type=Path, default=Path("training.tsv"), help="Log filename (TSV)")
    parser.add_argument("--threshold", type=float, default=0.001, help="Target test loss")
    parser.add_argument("--max-epochs", type=int, default=10000, help="Max epochs")
    parser.add_argument("--eval-interval", type=int, default=1, help="Evaluate every N epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-reasoning", type=float, default=0.5, help="Loss weight for reasoning")
    parser.add_argument("--weight-answer", type=float, default=1.0, help="Loss weight for answer")
    parser.add_argument("--weight-format", type=float, default=2.0, help="Loss weight for format")
    parser.add_argument("--device", default="auto", help="Device")
    parser.add_argument("-c", "--checkpoint", type=Path, help="Resume from checkpoint")
    parser.add_argument("--difficulty-threshold", type=float, default=1e-3, help="Train loss threshold to advance difficulty")
    parser.add_argument("--difficulty-epochs", type=int, default=5, help="Epochs between difficulty checks")
    args = parser.parse_args()

    # Resolve paths with experiment subdirectory
    exp_name = args.experiment or "default"
    exp_dir = f"exp-{exp_name}"
    data_dir = Path("data") / exp_dir
    checkpoint_dir = Path("checkpoints") / exp_dir
    print(f"Experiment: {exp_name} ({exp_dir})")

    args.output = checkpoint_dir / args.output
    args.log = checkpoint_dir / args.log

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

    # Difficulty-based data loading
    def load_difficulty_data(difficulty: int) -> tuple[DataLoader, DataLoader, DataLoader | None] | None:
        """Load train, test, test-complex loaders for given difficulty. Returns None if no data."""
        files = get_difficulty_files(data_dir, difficulty)
        if files is None:
            return None

        train_file, test_file, test_complex_file = files

        train_dataset = MathDataset(train_file, tokenizer)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        )
        print(f"Difficulty {difficulty}: train={train_file} ({len(train_dataset)} examples)")

        test_loader = None
        if test_file.exists():
            test_dataset = MathDataset(test_file, tokenizer)
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
            )
            print(f"Difficulty {difficulty}: test={test_file} ({len(test_dataset)} examples)")

        test_complex_loader = None
        if test_complex_file.exists():
            test_complex_dataset = MathDataset(test_complex_file, tokenizer)
            test_complex_loader = DataLoader(
                test_complex_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
            )
            print(f"Difficulty {difficulty}: test-complex={test_complex_file} ({len(test_complex_dataset)} examples)")

        return train_loader, test_loader, test_complex_loader

    # Initial difficulty
    difficulty = 0
    loaders = load_difficulty_data(difficulty)
    if loaders is None:
        print(f"Error: No data files for difficulty {difficulty}")
        return
    train_loader, test_loader, test_complex_loader = loaders

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

    num_layers = len(model.layers)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,} ({num_layers} layers)")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Training config
    print(f"\nConfig:")
    print(f"  threshold: {args.threshold}")
    print(f"  difficulty_threshold: {args.difficulty_threshold}")
    print(f"  difficulty_epochs: {args.difficulty_epochs}")
    print(f"  eval_interval: {args.eval_interval}")
    print(f"  weight_reasoning: {args.weight_reasoning}")
    print(f"  weight_answer: {args.weight_answer}")
    print(f"  weight_format: {args.weight_format}")
    print(f"  lr: {args.lr}")
    print()

    # Training loop
    last_train_loss_answer = float("inf")
    for epoch in range(1, args.max_epochs + 1):
        # Check if we can advance difficulty
        if epoch > 1 and (epoch - 1) % args.difficulty_epochs == 0:
            if last_train_loss_answer < args.difficulty_threshold:
                new_difficulty = difficulty + 1
                new_loaders = load_difficulty_data(new_difficulty)
                if new_loaders is None:
                    print(f"\nNo data for difficulty {new_difficulty}. Training complete.")
                    break
                difficulty = new_difficulty
                train_loader, test_loader, test_complex_loader = new_loaders
                print(f"Advanced to difficulty {difficulty} (loss_a {last_train_loss_answer:.6f} < {args.difficulty_threshold})")

        # Snapshot weights before training epoch
        weight_snapshots = snapshot_layer_weights(model)

        model.train()
        total_loss = 0.0
        max_seq_len = 0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]

            # Track max sequence length
            max_seq_len = max(max_seq_len, input_ids.size(1))

            logits = model(inputs)

            loss_weights = create_loss_weights(
                targets.size(1),
                batch["reasoning_ranges"],
                batch["answer_ranges"],
                batch["format_positions"],
                args.weight_reasoning,
                args.weight_answer,
                args.weight_format,
                device,
            )

            logits_flat = logits.reshape(-1, logits.size(-1))
            targets_flat = targets.reshape(-1)
            weights_flat = loss_weights.reshape(-1)

            loss_per_token = F.cross_entropy(logits_flat, targets_flat, reduction="none")
            weighted_loss = (loss_per_token * weights_flat).sum()
            total_weight = weights_flat.sum()

            if total_weight > 0:
                loss = weighted_loss / total_weight
            else:
                loss = weighted_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        # Evaluate on train and test sets
        if epoch % args.eval_interval == 0:
            train_metrics = evaluate(
                model,
                train_loader,
                args.weight_reasoning,
                args.weight_answer,
                args.weight_format,
                device,
            )
            last_train_loss_answer = train_metrics.loss_answer

            # Evaluate on test set if available
            test_metrics = None
            if test_loader is not None:
                test_metrics = evaluate(
                    model,
                    test_loader,
                    args.weight_reasoning,
                    args.weight_answer,
                    args.weight_format,
                    device,
                )

            # Evaluate on complex test set if available
            test_complex_metrics = None
            if test_complex_loader is not None:
                test_complex_metrics = evaluate(
                    model,
                    test_complex_loader,
                    args.weight_reasoning,
                    args.weight_answer,
                    args.weight_format,
                    device,
                )

            # Print metrics
            line = (
                f"Epoch {epoch:5d} D{difficulty} | "
                f"Train: {train_metrics.loss_total:.4f} (w:{train_metrics.loss_reasoning:.4f} a:{train_metrics.loss_answer:.4f} acc:{train_metrics.accuracy:.2%})"
            )
            if test_metrics:
                line += f" | Test: {test_metrics.loss_total:.4f} (w:{test_metrics.loss_reasoning:.4f} a:{test_metrics.loss_answer:.4f} acc:{test_metrics.accuracy:.2%})"
            if test_complex_metrics:
                line += f" | Complex: {test_complex_metrics.loss_total:.4f} (acc:{test_complex_metrics.accuracy:.2%})"
            line += f" | MaxSeq: {max_seq_len}"
            print(line)

            layer_weight_norms = compute_layer_weight_norms(model)
            layer_weight_rms_changes = compute_layer_weight_rms_changes(model, weight_snapshots)
            log_metrics(
                args.log, epoch, difficulty, train_metrics, test_metrics, test_complex_metrics,
                max_seq_len=max_seq_len,
                layer_weight_norms=layer_weight_norms,
                layer_weight_rms_changes=layer_weight_rms_changes,
            )

            # Save checkpoint
            torch.save(model.state_dict(), args.output)

            # Check convergence (only if test_metrics available)
            if test_metrics and test_metrics.loss_total < args.threshold:
                print(f"\nConverged! Test loss {test_metrics.loss_total:.6f} < {args.threshold}")
                break
        else:
            print(f"Epoch {epoch:5d} D{difficulty} | Train loss: {train_loss:.6f}")

    print(f"\nTraining complete. Model saved to {args.output}")
    print(f"Log saved to {args.log}")


if __name__ == "__main__":
    main()
