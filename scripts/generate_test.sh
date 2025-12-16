#!/bin/bash
cd "$(dirname "$0")/.."

EXP="${1:-}"
if [ -n "$EXP" ]; then
    EXP_ARG="-e $EXP"
else
    EXP_ARG=""
fi

# Generate test files for each difficulty level
# Difficulty 0-5: small numbers
for i in {0..5}; do
    MIN=$((i + 1))
    MAX=$((i + 2))
    uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits $MIN --max-digits $MAX -o test-$i.jsonl
    uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits $((MAX + 2)) --max-digits $((MAX + 5)) -o test-complex-$i.jsonl
done

# Difficulty 6+: larger numbers
for i in {6..15}; do
    MIN=$((i - 3))
    MAX=$((i))
    uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits $MIN --max-digits $MAX -o test-$i.jsonl
    uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits $((MAX + 2)) --max-digits $((MAX + 8)) -o test-complex-$i.jsonl
done
