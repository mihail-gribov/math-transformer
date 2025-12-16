#!/bin/bash
cd "$(dirname "$0")/.."

EXP="${1:-}"
if [ -n "$EXP" ]; then
    EXP_ARG="-e $EXP"
else
    EXP_ARG=""
fi

# Generate test files matching train difficulty levels
# test: min = train_max + 1, max = train_max * 2
# test-complex: min = test_max, max = test_max + 2

# examples-0: 1-2 → test 3-4, complex 5-6
uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits 3 --max-digits 4 -o test-0.jsonl
uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits 5 --max-digits 6 -o test-complex-0.jsonl

# examples-1: 1-3 → test 4-6, complex 7-8
uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits 4 --max-digits 6 -o test-1.jsonl
uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits 7 --max-digits 8 -o test-complex-1.jsonl

# examples-2: 2-3 → test 4-6, complex 7-8
uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits 4 --max-digits 6 -o test-2.jsonl
uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits 7 --max-digits 8 -o test-complex-2.jsonl

# examples-3: 1-4 → test 5-8, complex 9-10
uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits 5 --max-digits 8 -o test-3.jsonl
uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits 9 --max-digits 10 -o test-complex-3.jsonl

# examples-4: 2-4 → test 5-8, complex 9-10
uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits 5 --max-digits 8 -o test-4.jsonl
uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits 9 --max-digits 10 -o test-complex-4.jsonl

# examples-5: 3-4 → test 5-8, complex 9-10
uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits 5 --max-digits 8 -o test-5.jsonl
uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits 9 --max-digits 10 -o test-complex-5.jsonl

# examples-6: 1-5 → test 6-10, complex 11-12
uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits 6 --max-digits 10 -o test-6.jsonl
uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits 11 --max-digits 12 -o test-complex-6.jsonl

# examples-7: 2-5 → test 6-10, complex 11-12
uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits 6 --max-digits 10 -o test-7.jsonl
uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits 11 --max-digits 12 -o test-complex-7.jsonl

# examples-8: 3-5 → test 6-10, complex 11-12
uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits 6 --max-digits 10 -o test-8.jsonl
uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits 11 --max-digits 12 -o test-complex-8.jsonl

# examples-9: 4-5 → test 6-10, complex 11-12
uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits 6 --max-digits 10 -o test-9.jsonl
uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits 11 --max-digits 12 -o test-complex-9.jsonl

# examples-10: 2-6 → test 7-12, complex 13-14
uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits 7 --max-digits 12 -o test-10.jsonl
uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits 13 --max-digits 14 -o test-complex-10.jsonl

# examples-11: 3-6 → test 7-12, complex 13-14
uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits 7 --max-digits 12 -o test-11.jsonl
uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits 13 --max-digits 14 -o test-complex-11.jsonl

# examples-12: 4-6 → test 7-12, complex 13-14
uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits 7 --max-digits 12 -o test-12.jsonl
uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits 13 --max-digits 14 -o test-complex-12.jsonl

# examples-13: 5-6 → test 7-12, complex 13-14
uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits 7 --max-digits 12 -o test-13.jsonl
uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits 13 --max-digits 14 -o test-complex-13.jsonl

# examples-14: 2-7 → test 8-14, complex 15-16
uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits 8 --max-digits 14 -o test-14.jsonl
uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits 15 --max-digits 16 -o test-complex-14.jsonl

# examples-15: 3-7 → test 8-14, complex 15-16
uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits 8 --max-digits 14 -o test-15.jsonl
uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits 15 --max-digits 16 -o test-complex-15.jsonl

# examples-16: 4-7 → test 8-14, complex 15-16
uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits 8 --max-digits 14 -o test-16.jsonl
uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits 15 --max-digits 16 -o test-complex-16.jsonl

# examples-17: 5-7 → test 8-14, complex 15-16
uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits 8 --max-digits 14 -o test-17.jsonl
uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits 15 --max-digits 16 -o test-complex-17.jsonl

# examples-18: 6-7 → test 8-14, complex 15-16
uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits 8 --max-digits 14 -o test-18.jsonl
uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits 15 --max-digits 16 -o test-complex-18.jsonl

# examples-19: 2-8 → test 9-16, complex 17-18
uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits 9 --max-digits 16 -o test-19.jsonl
uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits 17 --max-digits 18 -o test-complex-19.jsonl

# examples-20: 3-8 → test 9-16, complex 17-18
uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits 9 --max-digits 16 -o test-20.jsonl
uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits 17 --max-digits 18 -o test-complex-20.jsonl

# examples-21: 4-8 → test 9-16, complex 17-18
uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits 9 --max-digits 16 -o test-21.jsonl
uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits 17 --max-digits 18 -o test-complex-21.jsonl

# examples-22: 5-8 → test 9-16, complex 17-18
uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits 9 --max-digits 16 -o test-22.jsonl
uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits 17 --max-digits 18 -o test-complex-22.jsonl

# examples-23: 6-8 → test 9-16, complex 17-18
uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits 9 --max-digits 16 -o test-23.jsonl
uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits 17 --max-digits 18 -o test-complex-23.jsonl

# examples-24: 7-8 → test 9-16, complex 17-18
uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits 9 --max-digits 16 -o test-24.jsonl
uv run python example_generator.py $EXP_ARG -n 500 --weights "1:1,3:0" --min-digits 17 --max-digits 18 -o test-complex-24.jsonl
