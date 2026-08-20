"""
Simple EDA for Rift Archetypes raw collection files.

Run from repo root:
    python src/eda_overview.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from statistics import median
from typing import Any, Iterable, Iterator


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight EDA over collected raw files.")
    parser.add_argument("--league-entries-path", default="data/raw/league_entries.json")
    parser.add_argument("--match-ids-path", default="data/raw/match_ids.ndjson")
    parser.add_argument("--matches-path", default="data/raw/matches.ndjson")
    parser.add_argument("--max-records", type=int, default=0, help="0 means all records.")
    parser.add_argument("--top-k-categories", type=int, default=8)
    parser.add_argument("--max-numeric-samples", type=int, default=50000)
    parser.add_argument("--max-sample-values", type=int, default=3)
    parser.add_argument("--report-path", default="data/raw/eda_report.md")
    return parser.parse_args()


def file_size_mb(path: str) -> float:
    if not os.path.exists(path):
        return 0.0
    return os.path.getsize(path) / (1024 * 1024)


def iter_json_records(path: str) -> Iterator[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as infile:
        payload = json.load(infile)

    if isinstance(payload, list):
        for row in payload:
            if isinstance(row, dict):
                yield row
    elif isinstance(payload, dict):
        yield payload


def iter_ndjson_records(path: str) -> Iterator[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as infile:
        for raw_line in infile:
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                yield row


def infer_type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return sorted_values[0]

    pos = (len(sorted_values) - 1) * q
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return sorted_values[low]

    weight = pos - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


@dataclass
class NumericAccumulator:
    values: list[float] = field(default_factory=list)
    max_samples: int = 50000

    def add(self, value: float) -> None:
        if len(self.values) < self.max_samples:
            self.values.append(float(value))

    def describe(self) -> dict[str, float | int | None]:
        if not self.values:
            return {
                "count": 0,
                "mean": None,
                "min": None,
                "p50": None,
                "p90": None,
                "max": None,
            }

        values = sorted(self.values)
        count = len(values)
        mean_value = sum(values) / count
        return {
            "count": count,
            "mean": mean_value,
            "min": values[0],
            "p50": median(values),
            "p90": quantile(values, 0.90),
            "max": values[-1],
        }


@dataclass
class FieldStats:
    present_count: int = 0
    null_count: int = 0
    types: Counter[str] = field(default_factory=Counter)
    sample_values: list[str] = field(default_factory=list)
    categorical: Counter[str] = field(default_factory=Counter)
    numeric: NumericAccumulator = field(default_factory=NumericAccumulator)

    def register(self, value: Any, max_sample_values: int, top_k_categories: int) -> None:
        self.present_count += 1
        type_name = infer_type_name(value)
        self.types[type_name] += 1

        if value is None:
            self.null_count += 1
            return

        if len(self.sample_values) < max_sample_values:
            self.sample_values.append(repr(value)[:120])

        if is_number(value):
            self.numeric.add(float(value))

        if isinstance(value, (str, bool, int, float)):
            if len(self.categorical) <= top_k_categories * 3:
                self.categorical[str(value)] += 1


def walk_leaf_fields(payload: dict[str, Any]) -> Iterator[tuple[str, Any]]:
    stack: list[tuple[str, Any]] = [("", payload)]

    while stack:
        prefix, value = stack.pop()

        if isinstance(value, dict):
            if not value and prefix:
                yield prefix, value
                continue
            for key, nested in value.items():
                key_path = f"{prefix}.{key}" if prefix else key
                stack.append((key_path, nested))
            continue

        if isinstance(value, list):
            list_key = f"{prefix}[]" if prefix else "[]"
            yield list_key, value
            for nested in value:
                stack.append((list_key, nested))
            continue

        yield prefix, value


def limit_records(records: Iterable[dict[str, Any]], max_records: int) -> Iterator[dict[str, Any]]:
    if max_records <= 0:
        yield from records
        return

    for idx, record in enumerate(records):
        if idx >= max_records:
            return
        yield record


def profile_dataset(
    dataset_name: str,
    path: str,
    record_iter: Iterable[dict[str, Any]],
    max_sample_values: int,
    top_k_categories: int,
    max_numeric_samples: int,
) -> dict[str, Any]:
    field_profiles: dict[str, FieldStats] = {}
    record_count = 0

    for record in record_iter:
        record_count += 1
        for field_path, value in walk_leaf_fields(record):
            if field_path not in field_profiles:
                field_profiles[field_path] = FieldStats(
                    numeric=NumericAccumulator(max_samples=max_numeric_samples)
                )
            field_profiles[field_path].register(
                value=value,
                max_sample_values=max_sample_values,
                top_k_categories=top_k_categories,
            )

    schema_rows: list[dict[str, Any]] = []
    for field_path in sorted(field_profiles.keys()):
        stats = field_profiles[field_path]
        missing_count: int | None
        missing_pct: float | None
        occurrences_per_record = (stats.present_count / record_count) if record_count else 0.0

        if stats.present_count <= record_count:
            missing_count = record_count - stats.present_count
            missing_pct = (missing_count / record_count * 100.0) if record_count else 0.0
        else:
            missing_count = None
            missing_pct = None

        numeric_summary = stats.numeric.describe()
        top_categories = stats.categorical.most_common(top_k_categories)

        schema_rows.append(
            {
                "field": field_path,
                "present": stats.present_count,
                "occurrences_per_record": occurrences_per_record,
                "missing": missing_count,
                "missing_pct": missing_pct,
                "nulls": stats.null_count,
                "types": dict(stats.types),
                "samples": stats.sample_values,
                "top_categories": top_categories,
                "numeric": numeric_summary,
            }
        )

    return {
        "dataset": dataset_name,
        "path": path,
        "file_size_mb": round(file_size_mb(path), 3),
        "records_profiled": record_count,
        "fields_profiled": len(schema_rows),
        "schema": schema_rows,
    }


def render_report(profiles: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# Rift Archetypes EDA Report")
    lines.append("")
    lines.append(f"Generated at: {utc_now_iso()}")
    lines.append("")

    for profile in profiles:
        lines.append(f"## {profile['dataset']}")
        lines.append("")
        lines.append(f"- Path: `{profile['path']}`")
        lines.append(f"- File size (MB): {profile['file_size_mb']}")
        lines.append(f"- Records profiled: {profile['records_profiled']}")
        lines.append(f"- Fields profiled: {profile['fields_profiled']}")
        lines.append("")
        lines.append("### Field Profile")
        lines.append("")
        lines.append("| field | present | avg occurrences/record | missing% | types | numeric summary | top categories | sample values |")
        lines.append("|---|---:|---:|---:|---|---|---|---|")

        for row in profile["schema"]:
            numeric = row["numeric"]
            numeric_text = ""
            if numeric["count"] > 0:
                numeric_text = (
                    f"n={numeric['count']}, mean={numeric['mean']:.3f}, "
                    f"min={numeric['min']:.3f}, p50={numeric['p50']:.3f}, "
                    f"p90={numeric['p90']:.3f}, max={numeric['max']:.3f}"
                )

            categories_text = ", ".join([f"{k}:{v}" for k, v in row["top_categories"]])
            samples_text = ", ".join(row["samples"])
            types_text = ", ".join([f"{k}:{v}" for k, v in sorted(row["types"].items())])
            missing_pct = "-" if row["missing_pct"] is None else f"{row['missing_pct']:.2f}"

            lines.append(
                "| "
                f"{row['field']} | {row['present']} | {row['occurrences_per_record']:.2f} | {missing_pct} | "
                f"{types_text} | "
                f"{numeric_text} | {categories_text} | {samples_text} |"
            )

        lines.append("")

    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    datasets: list[tuple[str, str, str]] = [
        ("league_entries", args.league_entries_path, "json"),
        ("match_ids", args.match_ids_path, "ndjson"),
        ("matches", args.matches_path, "ndjson"),
    ]

    profiles: list[dict[str, Any]] = []

    for dataset_name, path, file_type in datasets:
        if not os.path.exists(path):
            print(f"Skipping {dataset_name}: file not found at {path}")
            continue

        print(f"Profiling {dataset_name} from {path} ...")

        if file_type == "json":
            records = iter_json_records(path)
        else:
            records = iter_ndjson_records(path)

        profile = profile_dataset(
            dataset_name=dataset_name,
            path=path,
            record_iter=limit_records(records, args.max_records),
            max_sample_values=args.max_sample_values,
            top_k_categories=args.top_k_categories,
            max_numeric_samples=args.max_numeric_samples,
        )
        profiles.append(profile)

    report = render_report(profiles)

    os.makedirs(os.path.dirname(args.report_path), exist_ok=True)
    with open(args.report_path, "w", encoding="utf-8") as outfile:
        outfile.write(report)

    print("\nEDA complete.")
    print(f"Report written to: {args.report_path}")
    for profile in profiles:
        print(
            f"- {profile['dataset']}: records={profile['records_profiled']}, "
            f"fields={profile['fields_profiled']}, file_size_mb={profile['file_size_mb']}"
        )


if __name__ == "__main__":
    run(parse_args())