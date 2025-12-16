#!/bin/bash
cd "$(dirname "$0")/.."
uv run python example_generator.py -n 1000 --weights "1:1,3:0" --min-digits 7 --max-digits 12 -o data/test.jsonl
uv run python example_generator.py -n 1000 --weights "1:1,3:0" --min-digits 12 --max-digits 20 -o data/test-complex.jsonl