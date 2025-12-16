#!/bin/bash
cd "$(dirname "$0")/.."

EXP="${1:-}"
if [ -n "$EXP" ]; then
    EXP_ARG="-e $EXP"
else
    EXP_ARG=""
fi

uv run python example_generator.py $EXP_ARG -n 50000 --min-digits 1 --max-digits 2 -o examples-0.jsonl --show
uv run python example_generator.py $EXP_ARG -n 50000 --min-digits 1 --max-digits 3 -o examples-1.jsonl --show
uv run python example_generator.py $EXP_ARG -n 50000 --min-digits 2 --max-digits 3 -o examples-2.jsonl --show
uv run python example_generator.py $EXP_ARG -n 50000 --min-digits 1 --max-digits 4 -o examples-3.jsonl --show
uv run python example_generator.py $EXP_ARG -n 50000 --min-digits 2 --max-digits 4 -o examples-4.jsonl --show
uv run python example_generator.py $EXP_ARG -n 50000 --min-digits 3 --max-digits 4 -o examples-5.jsonl --show
uv run python example_generator.py $EXP_ARG -n 50000 --min-digits 1 --max-digits 5 -o examples-6.jsonl --show
uv run python example_generator.py $EXP_ARG -n 50000 --min-digits 2 --max-digits 5 -o examples-7.jsonl --show
uv run python example_generator.py $EXP_ARG -n 50000 --min-digits 3 --max-digits 5 -o examples-8.jsonl --show
uv run python example_generator.py $EXP_ARG -n 50000 --min-digits 4 --max-digits 5 -o examples-9.jsonl --show
uv run python example_generator.py $EXP_ARG -n 50000 --min-digits 2 --max-digits 6 -o examples-10.jsonl --show
uv run python example_generator.py $EXP_ARG -n 50000 --min-digits 3 --max-digits 6 -o examples-11.jsonl --show
uv run python example_generator.py $EXP_ARG -n 50000 --min-digits 4 --max-digits 6 -o examples-12.jsonl --show
uv run python example_generator.py $EXP_ARG -n 50000 --min-digits 5 --max-digits 6 -o examples-13.jsonl --show
uv run python example_generator.py $EXP_ARG -n 50000 --min-digits 2 --max-digits 7 -o examples-14.jsonl --show
uv run python example_generator.py $EXP_ARG -n 50000 --min-digits 3 --max-digits 7 -o examples-15.jsonl --show
uv run python example_generator.py $EXP_ARG -n 50000 --min-digits 4 --max-digits 7 -o examples-16.jsonl --show
uv run python example_generator.py $EXP_ARG -n 50000 --min-digits 5 --max-digits 7 -o examples-17.jsonl --show
uv run python example_generator.py $EXP_ARG -n 50000 --min-digits 6 --max-digits 7 -o examples-18.jsonl --show
uv run python example_generator.py $EXP_ARG -n 50000 --min-digits 2 --max-digits 8 -o examples-19.jsonl --show
uv run python example_generator.py $EXP_ARG -n 50000 --min-digits 3 --max-digits 8 -o examples-20.jsonl --show
uv run python example_generator.py $EXP_ARG -n 50000 --min-digits 4 --max-digits 8 -o examples-21.jsonl --show
uv run python example_generator.py $EXP_ARG -n 50000 --min-digits 5 --max-digits 8 -o examples-22.jsonl --show
uv run python example_generator.py $EXP_ARG -n 50000 --min-digits 6 --max-digits 8 -o examples-23.jsonl --show
uv run python example_generator.py $EXP_ARG -n 50000 --min-digits 7 --max-digits 8 -o examples-24.jsonl --show
uv run python example_generator.py $EXP_ARG -n 50000 --min-digits 2 --max-digits 9 -o examples-19.jsonl --show
uv run python example_generator.py $EXP_ARG -n 50000 --min-digits 3 --max-digits 9 -o examples-20.jsonl --show
uv run python example_generator.py $EXP_ARG -n 50000 --min-digits 4 --max-digits 9 -o examples-21.jsonl --show
uv run python example_generator.py $EXP_ARG -n 50000 --min-digits 5 --max-digits 9 -o examples-22.jsonl --show
uv run python example_generator.py $EXP_ARG -n 50000 --min-digits 6 --max-digits 9 -o examples-23.jsonl --show
uv run python example_generator.py $EXP_ARG -n 50000 --min-digits 7 --max-digits 9 -o examples-24.jsonl --show
uv run python example_generator.py $EXP_ARG -n 50000 --min-digits 8 --max-digits 9 -o examples-24.jsonl --show
uv run python example_generator.py $EXP_ARG -n 50000 --min-digits 2 --max-digits 10 -o examples-24.jsonl --show
