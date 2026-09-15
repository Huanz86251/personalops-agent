"""Reviewer checks derived from contract.json before the implementation fixture.

Only the contract, public entry point and data fixtures determine assertions.
No Worker tests, transcripts, self-report or private execution state are used.
"""

import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest
import filter_report  # Candidate is NOT in reviewer's default /review import path.


def run_case(tmp_path, rows, config, existing=None):
    source = tmp_path / "中文 input.csv"
    settings = tmp_path / "settings.json"
    output = tmp_path / "filtered output.csv"
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "category", "amount"])
        writer.writerows(rows)
    settings.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    original = source.read_bytes()
    if existing is not None:
        output.write_bytes(existing)
    result = subprocess.run([sys.executable, filter_report.__file__, "--input", str(source),
        "--config", str(settings), "--output", str(output)], capture_output=True, text=True, timeout=10)
    assert source.read_bytes() == original
    return result, output


def read_result(result, output):
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    summary = json.loads(result.stdout)
    assert type(summary["written"]) is int
    with output.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["id", "category", "amount"]
        rows = list(reader)
    assert summary["written"] == len(rows)
    return rows


def test_normal_category_and_order(tmp_path):
    result, output = run_case(tmp_path, [("a", "餐饮", "12.50"), ("b", "交通", "99"),
        ("c", "餐饮", "3"), ("d", "餐饮", "18")], {"minimum_amount": 10, "category": "餐饮"})
    assert [r["id"] for r in read_result(result, output)] == ["a", "d"]


@pytest.mark.parametrize("minimum", [0, 10])
def test_inclusive_threshold_and_zero(tmp_path, minimum):
    result, output = run_case(tmp_path, [("equal", "餐饮", str(minimum)),
        ("above", "餐饮", str(minimum + 1))], {"minimum_amount": minimum})
    assert [r["id"] for r in read_result(result, output)] == ["equal", "above"]


@pytest.mark.parametrize("config", [{}, {"minimum_amount": None}, {"minimum_amount": True},
    {"minimum_amount": "10"}, {"minimum_amount": 0, "unexpected": 1}])
def test_schema_rejects_invalid_config(tmp_path, config):
    result, output = run_case(tmp_path, [("x", "餐饮", "15")], config)
    assert result.returncode == 2
    assert result.stderr
    assert result.stdout == ""
    assert not output.exists()


def test_bad_amount_preserves_existing_output(tmp_path):
    result, output = run_case(tmp_path, [("x", "餐饮", "oops")], {"minimum_amount": 0}, b"KEEP")
    assert result.returncode == 2
    assert "invalid amount" in result.stderr.lower()
    assert output.read_bytes() == b"KEEP"


def test_refuses_overwrite(tmp_path):
    result, output = run_case(tmp_path, [("x", "餐饮", "12")], {"minimum_amount": 0}, b"KEEP")
    assert result.returncode == 2
    assert "exists" in result.stderr.lower()
    assert output.read_bytes() == b"KEEP"


def test_quoted_unicode_fields_and_paths(tmp_path):
    result, output = run_case(tmp_path, [("中文,\n编号", "餐饮", "12.50")], {"minimum_amount": 1})
    assert read_result(result, output) == [{"id": "中文,\n编号", "category": "餐饮", "amount": "12.50"}]


def test_empty_csv_keeps_header(tmp_path):
    result, output = run_case(tmp_path, [], {"minimum_amount": 0})
    assert read_result(result, output) == []
