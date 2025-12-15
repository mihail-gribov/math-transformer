#!/bin/bash
cd "$(dirname "$0")/.."
uv run python example_generator.py -n 1000 --weights "1:1,3:0" --min-digits 6 --max-digits 7 -o data/test.jsonl