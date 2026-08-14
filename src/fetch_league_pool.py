"""
Step 1: Pull a pool of players at a target rank tier via league-v4.
Saves raw results to data/raw/league_entries.json.

Run from the repo root:
    python src/fetch_league_pool.py
"""

from __future__ import annotations

import argparse
import json
import os

from riot_client import RiotClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch league player entries by tier/division.")
    parser.add_argument("--platform", default="na1")
    parser.add_argument("--queue", default="RANKED_SOLO_5x5")
    parser.add_argument("--tier", default="DIAMOND")
    parser.add_argument("--division", default="II")
    parser.add_argument("--num-pages", type=int, default=1)
    parser.add_argument("--output-path", default="data/raw/league_entries.json")
    return parser.parse_args()


def fetch_player_pool(args: argparse.Namespace) -> None:
    client = RiotClient(platform=args.platform)
    all_entries: list[dict] = []

    for page in range(1, args.num_pages + 1):
        print(f"Fetching page {page}...")
        entries = client.get_league_entries(args.queue, args.tier, args.division, page=page)

        if not entries:
            print("No more entries. Stopping early.")
            break

        all_entries.extend(entries)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as outfile:
        json.dump(all_entries, outfile, indent=2)

    print(f"Saved {len(all_entries)} player entries to {args.output_path}")


if __name__ == "__main__":
    fetch_player_pool(parse_args())
