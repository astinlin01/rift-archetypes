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


TOP_TIERS = {"MASTER", "GRANDMASTER", "CHALLENGER"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch league player entries by tier/division.")
    parser.add_argument("--platform", default="na1")
    parser.add_argument("--queue", default="RANKED_SOLO_5x5")
    parser.add_argument("--tier", default="DIAMOND")
    parser.add_argument("--division", default="I")
    parser.add_argument("--num-pages", type=int, default=1)
    parser.add_argument(
        "--positive-winrate-only",
        action="store_true",
        help="Keep only entries where wins > losses.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append deduped players to existing output instead of overwriting.",
    )
    parser.add_argument("--output-path", default="data/raw/league_entries.json")
    return parser.parse_args()


def attach_missing_puuids(client: RiotClient, entries: list[dict]) -> None:
    """Top-tier league endpoints omit puuid, so enrich from summoner-v4."""
    for idx, row in enumerate(entries, start=1):
        if row.get("puuid"):
            continue
        summoner_id = row.get("summonerId")
        if not summoner_id:
            continue
        print(f"Resolving puuid for top-tier entry {idx}/{len(entries)}...")
        summoner_payload = client.get_summoner_by_id(summoner_id)
        puuid = summoner_payload.get("puuid")
        if puuid:
            row["puuid"] = puuid


def dedupe_by_puuid(entries: list[dict]) -> list[dict]:
    seen: set[str] = set()
    deduped: list[dict] = []
    for row in entries:
        puuid = row.get("puuid")
        if not puuid:
            continue
        if puuid in seen:
            continue
        seen.add(puuid)
        deduped.append(row)
    return deduped


def fetch_player_pool(args: argparse.Namespace) -> None:
    client = RiotClient(platform=args.platform)
    all_entries: list[dict]
    tier = args.tier.upper()

    if tier in TOP_TIERS:
        print(f"Fetching {tier} league for queue {args.queue}...")
        league_payload = client.get_top_league(queue=args.queue, tier=tier)
        all_entries = league_payload.get("entries", [])
        attach_missing_puuids(client, all_entries)
    else:
        all_entries = []
        for page in range(1, args.num_pages + 1):
            print(f"Fetching page {page}...")
            entries = client.get_league_entries(args.queue, tier, args.division, page=page)

            if not entries:
                print("No more entries. Stopping early.")
                break

            all_entries.extend(entries)

    all_entries = dedupe_by_puuid(all_entries)

    if args.positive_winrate_only:
        before = len(all_entries)
        all_entries = [row for row in all_entries if int(row.get("wins", 0)) > int(row.get("losses", 0))]
        print(f"Applied positive winrate filter: {before} -> {len(all_entries)} entries.")

    if args.append and os.path.exists(args.output_path):
        existing = []
        with open(args.output_path, "r", encoding="utf-8") as infile:
            existing = json.load(infile)
        combined = dedupe_by_puuid(existing + all_entries)
        print(f"Appending to existing pool: {len(existing)} existing + {len(all_entries)} new -> {len(combined)} total.")
        all_entries = combined

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as outfile:
        json.dump(all_entries, outfile, indent=2)

    print(f"Saved {len(all_entries)} player entries to {args.output_path}")


if __name__ == "__main__":
    fetch_player_pool(parse_args())
