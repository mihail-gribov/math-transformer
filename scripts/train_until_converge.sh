#!/bin/bash
cd "$(dirname "$0")/.."
uv run python train_until_converge.py \
    --train data/examples.jsonl \
    --test data/test.jsonl \
    --threshold 0.001 \
    --unfreeze-epochs 4