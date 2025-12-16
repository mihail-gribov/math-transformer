#!/usr/bin/env python3
"""
Example generator for math training data.

Each generator function returns (context, reasoning, answer).
Type ID indicates difficulty (higher = harder).
"""

import argparse
import hashlib
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path


def make_id(context: str) -> str:
    """Generate ID as hash of context."""
    return hashlib.md5(context.encode()).hexdigest()[:8]


@dataclass
class Example:
    """Single training example."""

    id: str
    type_id: int
    type_name: str
    context: str
    reasoning: str
    answer: str


def generate_addition_no_carry(min_digits: int = 1, max_digits: int = 3) -> Example:
    """
    Generate addition example without carry.

    Type ID: 1
    Example: 123 + 432 = 555 (no digit sum > 9)
    """
    num_digits = random.randint(min_digits, max_digits)

    # Generate two numbers where each column sum <= 9
    digits_a = []
    digits_b = []

    for _ in range(num_digits):
        # Random digit for first number
        a = random.randint(0, 9)
        # Second digit such that a + b <= 9
        b = random.randint(0, 9 - a)
        digits_a.append(a)
        digits_b.append(b)

    # Ensure first digit is not 0 (for multi-digit numbers)
    if num_digits > 1:
        if digits_a[0] == 0:
            max_a = 8  # Leave room for b >= 1
            digits_a[0] = random.randint(1, max_a)
            digits_b[0] = random.randint(0, 9 - digits_a[0])
        if digits_b[0] == 0:
            max_b = 9 - digits_a[0]
            if max_b >= 1:
                digits_b[0] = random.randint(1, max_b)
            # If max_b < 1, keep digits_b[0] = 0 (e.g., 900 + 0xx)

    # Convert to numbers
    num_a = int("".join(map(str, digits_a)))
    num_b = int("".join(map(str, digits_b)))
    result = num_a + num_b

    # Build reasoning (step by step from right to left)
    reasoning_parts = []
    for i in range(num_digits - 1, -1, -1):
        a, b = digits_a[i], digits_b[i]
        s = a + b
        reasoning_parts.append(f"{a}+{b}={s}")

    reasoning = ", ".join(reasoning_parts)

    context = f"{num_a}+{num_b}"
    return Example(
        id=make_id(context),
        type_id=1,
        type_name="addition_no_carry",
        context=context,
        reasoning=reasoning,
        answer=str(result),
    )


def generate_addition_with_carry(min_digits: int = 1, max_digits: int = 3) -> Example:
    """
    Generate addition example that may include carries.

    Type ID: 2
    Example: 156 + 278 = 434
    Reasoning shows step-by-step with carries.
    """
    num_digits = random.randint(min_digits, max_digits)

    # Generate random numbers with given number of digits
    if num_digits == 1:
        num_a = random.randint(0, 9)
        num_b = random.randint(0, 9)
    else:
        min_val = 10 ** (num_digits - 1)
        max_val = 10**num_digits - 1
        num_a = random.randint(min_val, max_val)
        num_b = random.randint(min_val, max_val)

    result = num_a + num_b

    # Build reasoning (step by step from right to left)
    str_a = str(num_a).zfill(num_digits)
    str_b = str(num_b).zfill(num_digits)

    reasoning_parts = []
    carry = 0

    for i in range(num_digits - 1, -1, -1):
        a = int(str_a[i])
        b = int(str_b[i])
        s = a + b + carry

        if carry > 0:
            part = f"{a}+{b}+{carry}={s % 10}"
        else:
            part = f"{a}+{b}={s % 10}"

        if s >= 10:
            part += " c1"  # carry 1
            carry = 1
        else:
            carry = 0

        reasoning_parts.append(part)

    # If there's a final carry, add it
    if carry > 0:
        reasoning_parts.append("1")

    reasoning = ", ".join(reasoning_parts)

    context = f"{num_a}+{num_b}"
    return Example(
        id=make_id(context),
        type_id=2,
        type_name="addition_with_carry",
        context=context,
        reasoning=reasoning,
        answer=str(result),
    )


def generate_addition_binary_digits(min_digits: int = 1, max_digits: int = 5) -> Example:
    """
    Generate simple addition with digits 0 and 1 only (no carry possible).

    Type ID: 3
    Example: 11111 + 101 = 11212
    Since max digit sum is 1+1=2, no carry ever occurs.
    Operands can have different lengths.
    """
    num_digits_a = random.randint(min_digits, max_digits)
    num_digits_b = random.randint(min_digits, max_digits)

    # Generate digits (0 or 1 only)
    digits_a = [random.randint(0, 1) for _ in range(num_digits_a)]
    digits_b = [random.randint(0, 1) for _ in range(num_digits_b)]

    # Ensure first digit is 1 for multi-digit numbers
    if num_digits_a > 1:
        digits_a[0] = 1
    if num_digits_b > 1:
        digits_b[0] = 1

    # Convert to numbers
    num_a = int("".join(map(str, digits_a)))
    num_b = int("".join(map(str, digits_b)))
    result = num_a + num_b

    # Pad shorter number with leading zeros for reasoning
    max_len = max(num_digits_a, num_digits_b)
    padded_a = [0] * (max_len - num_digits_a) + digits_a
    padded_b = [0] * (max_len - num_digits_b) + digits_b

    # Build reasoning (step by step from right to left)
    reasoning_parts = []
    for i in range(max_len - 1, -1, -1):
        a, b = padded_a[i], padded_b[i]
        s = a + b
        reasoning_parts.append(f"{a}+{b}={s}")

    reasoning = ", ".join(reasoning_parts)

    context = f"{num_a}+{num_b}"
    return Example(
        id=make_id(context),
        type_id=3,
        type_name="addition_binary_digits",
        context=context,
        reasoning=reasoning,
        answer=str(result),
    )


# Registry of generators: type_id -> (name, function, count_multiplier, digits_multiplier)
GENERATORS = {
    1: ("addition_no_carry", generate_addition_no_carry, 1, 1),
    2: ("addition_with_carry", generate_addition_with_carry, 0, 1),  # disabled
    3: ("addition_binary_digits", generate_addition_binary_digits, 1, 3),
}


def generate_examples(
    generator_fn,
    count: int,
    seen_ids: set[str] | None = None,
    max_consecutive_duplicates: int = 1000,
    **kwargs,
) -> list[Example]:
    """Generate multiple unique examples using given generator."""
    if seen_ids is None:
        seen_ids = set()

    examples = []
    consecutive_duplicates = 0

    while len(examples) < count:
        example = generator_fn(**kwargs)
        if example.id in seen_ids:
            consecutive_duplicates += 1
            if consecutive_duplicates >= max_consecutive_duplicates:
                print(f"Warning: {consecutive_duplicates} consecutive duplicates, stopping ({len(examples)}/{count})")
                break
            continue
        consecutive_duplicates = 0
        seen_ids.add(example.id)
        examples.append(example)

    return examples


def save_examples(examples: list[Example], output_path: Path, append: bool = False):
    """Save examples to JSONL file."""
    mode = "a" if append else "w"
    with open(output_path, mode) as f:
        for ex in examples:
            data = asdict(ex)
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate math training examples")
    parser.add_argument("-o", "--output", type=Path, default=Path("data/examples.jsonl"), help="Output path")
    parser.add_argument("-n", "--count", type=int, default=100, help="Number of examples per type")
    parser.add_argument("--min-digits", type=int, default=1, help="Min digits for numbers")
    parser.add_argument("--max-digits", type=int, default=3, help="Max digits for numbers")
    parser.add_argument("--types", type=int, nargs="+", help="Type IDs to generate (default: all)")
    parser.add_argument("--weights", type=str, help="Override count multipliers, e.g. '1:2,3:5'")
    parser.add_argument("--append", action="store_true", help="Append to existing file")
    parser.add_argument("--show", action="store_true", help="Print examples to stdout")
    args = parser.parse_args()

    # Parse weight overrides
    weight_overrides = {}
    if args.weights:
        for part in args.weights.split(","):
            tid, mult = part.split(":")
            weight_overrides[int(tid)] = int(mult)

    # Select types to generate
    type_ids = args.types or list(GENERATORS.keys())

    all_examples = []
    seen_ids: set[str] = set()

    for type_id in type_ids:
        if type_id not in GENERATORS:
            print(f"Unknown type ID: {type_id}")
            continue

        type_name, generator_fn, count_mult, digits_mult = GENERATORS[type_id]
        count_mult = weight_overrides.get(type_id, count_mult)
        if count_mult == 0:
            continue
        count = args.count * count_mult
        min_digits = args.min_digits * digits_mult
        max_digits = args.max_digits * digits_mult
        print(f"Generating {count} examples of type {type_id} ({type_name}) [count x{count_mult}, digits {min_digits}-{max_digits}]...")

        examples = generate_examples(
            generator_fn,
            count=count,
            seen_ids=seen_ids,
            min_digits=min_digits,
            max_digits=max_digits,
        )
        all_examples.extend(examples)

    # Shuffle
    random.shuffle(all_examples)

    # Save
    save_examples(all_examples, args.output, append=args.append)
    print(f"Saved {len(all_examples)} examples to {args.output}")

    # Show samples
    if args.show:
        print("\nSamples:")
        for ex in all_examples[:5]:
            print(f"  [{ex.type_id}] {ex.context} | {ex.reasoning} | {ex.answer}")


if __name__ == "__main__":
    main()
