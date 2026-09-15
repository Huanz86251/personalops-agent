"""Small offline implementation used only by the live CODE handoff probe."""

import argparse
import csv
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import sys


def process(source, config_path, output):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or "minimum_amount" not in config:
        raise ValueError("minimum_amount is required")
    if set(config) - {"minimum_amount", "category"}:
        raise ValueError("unknown config field")
    if type(config["minimum_amount"]) not in (int, float):
        raise ValueError("minimum_amount must be a number")
    minimum = Decimal(str(config["minimum_amount"]))
    if not minimum.is_finite():
        raise ValueError("minimum_amount must be finite")
    if "category" in config and not isinstance(config["category"], str):
        raise ValueError("category must be a string")
    selected = []
    with Path(source).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["id", "category", "amount"]:
            raise ValueError("invalid CSV header")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("invalid CSV row")
            try:
                amount = Decimal(row["amount"])
            except InvalidOperation as error:
                raise ValueError("invalid amount") from error
            if not amount.is_finite():
                raise ValueError("invalid amount")
            if amount >= minimum and ("category" not in config or row["category"] == config["category"]):
                selected.append(row)
    with Path(output).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "category", "amount"])
        writer.writeheader()
        writer.writerows(selected)
    return {"written": len(selected)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        summary = process(args.input, args.config, args.output)
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
