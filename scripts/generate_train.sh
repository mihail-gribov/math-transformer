#!/bin/bash
cd "$(dirname "$0")/.."
uv run python example_generator.py -n 50000 --min-digits 1 --max-digits 2 -o data/examples.jsonl --show
