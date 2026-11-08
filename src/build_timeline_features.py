"""
Build the early-game feature table from stored timelines (see timeline_features.py).

Scans matches.ndjson once for every match that has a stored timeline (the match payload
supplies each participant's team and role), extracts the per-participant features and
writes one row per (matchId, puuid).

Run from repo root:
    python src/build_timeline_features.py
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from pipeline_io import iter_ndjson
from timeline_features import DEFAULT_TIMELINE_DIR, TIMELINE_FEATURE_COLUMNS, extract_timeline_features, read_timeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract early-game features from stored timelines.")
    parser.add_argument("--matches-path", default="data/raw/matches.ndjson")
    parser.add_argument("--timeline-dir", default=DEFAULT_TIMELINE_DIR)
    parser.add_argument("--output-path", default="data/processed/timeline_features.parquet")
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    if not os.path.isdir(args.timeline_dir):
        raise FileNotFoundError(f"No timelines at {args.timeline_dir}. Run fetch_match_timelines.py first.")
    stored = {name[: -len(".json.gz")] for name in os.listdir(args.timeline_dir) if name.endswith(".json.gz")}
    print(f"{len(stored)} stored timelines.")

    rows: list[dict] = []
    found = 0
    for match in iter_ndjson(args.matches_path):
        match_id = match.get("metadata", {}).get("matchId")
        if match_id not in stored:
            continue
        found += 1
        rows.extend(extract_timeline_features(read_timeline(match_id, args.timeline_dir), match))
        if found == len(stored):
            break

    df = pd.DataFrame(rows)
    print(f"Matched {found}/{len(stored)} timelines to match payloads -> {len(df)} participant rows.")
    if len(df):
        print("\nMissing values per feature (short games / ambiguous lanes / undefined shares):")
        print(df[TIMELINE_FEATURE_COLUMNS].isna().mean().round(3).to_string())

    parent = os.path.dirname(args.output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    df.to_parquet(args.output_path, index=False)
    print(f"\nTimeline feature table written to: {args.output_path}")


if __name__ == "__main__":
    run(parse_args())
