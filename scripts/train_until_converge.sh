#!/bin/bash
cd "$(dirname "$0")/.."

EXP="${1:-}"
if [ -n "$EXP" ]; then
    EXP_ARG="-e $EXP"
else
    EXP_ARG=""
fi

uv run python train_until_converge.py $EXP_ARG \
    --threshold 0.001 \
    --difficulty-threshold 1e-3 \
    --unfreeze-epochs 5 \
    --max-unfrozen 10
