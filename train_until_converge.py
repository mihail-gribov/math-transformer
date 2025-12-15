#!/usr/bin/env python3
"""Training script that trains until test loss converges below threshold."""

import argparse
import json
import torch
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader

from model import MathTransformer
from train import MathDataset, collate_fn, create_loss_weights
from generate import CharTokenizer


def evaluate(
    model: MathTransformer,
    dataloader: DataLoader,
    weight_reasoning: float,
    weight_answer: float,
    weight_format: float,
    device: torch.device,
) -> float:
    """Evaluate model on dataset, return average loss."""
    model.eval()
    total_loss = 0.0
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
            total_batches += 1

    return total_loss / total_batches if total_batches > 0 else 0.0


def log_metrics(
    log_path: Path,
    epoch: int,
    train_loss: float,
    test_loss: float | None,
    start_layer: int | None = None,
    end_layer: int | None = None,
):
    """Append metrics to log file."""
    timestamp = datetime.now().isoformat(timespec="seconds")
    entry = {
        "timestamp": timestamp,
        "epoch": epoch,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "unfrozen_start": start_layer,
        "unfrozen_end": end_layer,
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def get_train_file(base_path: Path, layer_idx: int) -> Path:
    """
    Get training file for given layer index.

    Tries examples-{layer_idx}.jsonl first, falls back to base_path.
    """
    layer_file = base_path.parent / f"examples-{layer_idx}.jsonl"
    if layer_file.exists():
        return layer_file
    return base_path


def freeze_layers(model: MathTransformer, start_layer: int, end_layer: int):
    """
    Freeze all layers except those in range [start_layer, end_layer).

    Embeddings and output are always trainable.
    """
    # Unfreeze embeddings and output
    model.token_embedding.requires_grad_(True)
    model.norm.requires_grad_(True)
    model.output.requires_grad_(True)

    # Freeze/unfreeze transformer layers
    num_layers = len(model.layers)
    for i, layer in enumerate(model.layers):
        if start_layer <= i < end_layer:
            layer.requires_grad_(True)
        else:
            layer.requires_grad_(False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Unfrozen layers: {start_layer}-{end_layer-1} ({end_layer - start_layer}/{num_layers}) | Trainable params: {trainable:,}/{total:,}")


def main():
    parser = argparse.ArgumentParser(description="Train until convergence")
    parser.add_argument("--train", type=Path, default=Path("data/examples.jsonl"), help="Training data")
    parser.add_argument("--test", type=Path, default=Path("data/test.jsonl"), help="Test data")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/model.pt"), help="Model output")
    parser.add_argument("--log", type=Path, default=Path("checkpoints/training.log"), help="Log file")
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
    parser.add_argument("--unfreeze-epochs", type=int, default=0, help="Epochs per layer unfreeze (0 = all unfrozen)")
    parser.add_argument("--max-unfrozen", type=int, default=8, help="Max unfrozen layers (sliding window, 0 = no limit)")
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

    # Datasets
    def load_train_data(layer_idx: int) -> DataLoader:
        train_file = get_train_file(args.train, layer_idx)
        dataset = MathDataset(train_file, tokenizer)
        print(f"Train data: {train_file} ({len(dataset)} examples)")
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        )

    train_loader = load_train_data(0)

    test_dataset = MathDataset(args.test, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )
    print(f"Test examples: {len(test_dataset)}")

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

    # Gradual unfreezing setup
    num_layers = len(model.layers)
    if args.unfreeze_epochs > 0:
        start_layer = 0
        window_size = args.max_unfrozen if args.max_unfrozen > 0 else num_layers
        end_layer = min(1, num_layers)  # Start with 1 layer
        freeze_layers(model, start_layer, end_layer)
    else:
        start_layer = 0
        end_layer = num_layers
        window_size = num_layers

    # Optimizer (only trainable params)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # Training config
    print(f"\nConfig:")
    print(f"  threshold: {args.threshold}")
    print(f"  eval_interval: {args.eval_interval}")
    print(f"  weight_reasoning: {args.weight_reasoning}")
    print(f"  weight_answer: {args.weight_answer}")
    print(f"  weight_format: {args.weight_format}")
    print(f"  unfreeze_epochs: {args.unfreeze_epochs}")
    print(f"  max_unfrozen: {args.max_unfrozen} (window_size: {window_size})")
    print()

    # Training loop
    current_layer_idx = 0
    for epoch in range(1, args.max_epochs + 1):
        # Check if we need to slide the unfrozen window
        if args.unfreeze_epochs > 0 and end_layer < num_layers:
            if epoch > 1 and (epoch - 1) % args.unfreeze_epochs == 0:
                end_layer += 1
                current_layer_idx = end_layer - 1
                # Slide window: if we exceed max_unfrozen, move start_layer up
                if end_layer - start_layer > window_size:
                    start_layer += 1
                freeze_layers(model, start_layer, end_layer)
                # Reload training data for new layer
                train_loader = load_train_data(current_layer_idx)
                # Recreate optimizer with new trainable params
                optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]

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

        # Evaluate on test set
        if epoch % args.eval_interval == 0:
            test_loss = evaluate(
                model,
                test_loader,
                args.weight_reasoning,
                args.weight_answer,
                args.weight_format,
                device,
            )

            print(f"Epoch {epoch:5d} | Train: {train_loss:.6f} | Test: {test_loss:.6f} | Layers: {start_layer}-{end_layer-1}/{num_layers}")
            log_metrics(args.log, epoch, train_loss, test_loss, start_layer, end_layer)

            # Save checkpoint
            torch.save(model.state_dict(), args.output)

            # Check convergence
            if test_loss < args.threshold:
                print(f"\nConverged! Test loss {test_loss:.6f} < {args.threshold}")
                break
        else:
            print(f"Epoch {epoch:5d} | Train: {train_loss:.6f} | Layers: {start_layer}-{end_layer-1}/{num_layers}")
            log_metrics(args.log, epoch, train_loss, None, start_layer, end_layer)

    print(f"\nTraining complete. Model saved to {args.output}")
    print(f"Log saved to {args.log}")


if __name__ == "__main__":
    main()
