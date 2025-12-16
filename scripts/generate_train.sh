#!/bin/bash
cd "$(dirname "$0")/.."

EXP="${1:-}"
if [ -n "$EXP" ]; then
    EXP_ARG="-e $EXP"
else
    EXP_ARG=""
fi

uv run python example_generator.py $EXP_ARG -n 50000 --min-digits 1 --max-digits 2 -o examples.jsonl --show
