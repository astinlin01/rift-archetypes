"""
Fetch match-v5 timelines and store them gzipped (one file per match, see timeline_features.py).

Default subset: your own games plus every game of each player with 10+ games in a role
(the players that make up the reference averages), restricted to the current season.
--all fetches every current-season match instead (backfill). Resumable: matches that
already have a file in --timeline-dir are skipped, so just re-run after an interruption.

Run from repo root (after build_features.py, so the selection sees all collected matches):
    python src/fetch_match_timelines.py
    python src/fetch_match_timelines.py --all
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
import requests

from accounts import ENV_VAR
from compare_self import resolve_puuids
from feature_config import current_season
from fetch_match_details import REPRESENTATIVE_PLATFORM_BY_REGION, region_for_match_id
from pipeline_io import append_ndjson, utc_now_iso
from riot_client import RiotClient
from timeline_features import DEFAULT_TIMELINE_DIR, select_timeline_match_ids, timeline_path, write_timeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and store gzipped match timelines.")
    parser.add_argument("--features-path", default="data/processed/participant_features.parquet")
    parser.add_argument("--timeline-dir", default=DEFAULT_TIMELINE_DIR)
    parser.add_argument("--failures-path", default="data/raw/failures/timeline_failures.ndjson")
    parser.add_argument("--platform", default="na1", help="Fallback platform for unrecognized match ID prefixes.")
    parser.add_argument("--puuid", action="append", default=[])
    parser.add_argument("--riot-id", action="append", default=[], help=f"Defaults to {ENV_VAR} in .env.")
    parser.add_argument("--min-player-games", type=int, default=10)
    parser.add_argument("--all", action="store_true", help="Fetch every current-season match, not just the subset.")
    parser.add_argument("--max-matches", type=int, default=0, help="Stop after this many fetches (0 = no cap).")
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    df = pd.read_parquet(args.features_path, columns=["matchId", "puuid", "teamPosition", "season", "tier"])
    season = current_season(df)
    df = df[df["season"] == season]

    if args.all:
        match_ids = sorted(df["matchId"].unique())
    else:
        my_puuids = resolve_puuids(args)
        match_ids = select_timeline_match_ids(df, my_puuids, args.min_player_games)

    todo = [m for m in match_ids if not os.path.exists(timeline_path(m, args.timeline_dir))]
    if args.max_matches > 0:
        todo = todo[: args.max_matches]
    print(f"Season {season}: {len(match_ids)} matches selected, {len(match_ids) - len(todo)} already stored, {len(todo)} to fetch.")

    clients: dict[str, RiotClient] = {}
    failures = 0
    for index, match_id in enumerate(todo, start=1):
        region = region_for_match_id(match_id, args.platform)
        if region not in clients:
            clients[region] = RiotClient(platform=REPRESENTATIVE_PLATFORM_BY_REGION[region])
        try:
            write_timeline(clients[region].get_match_timeline(match_id), args.timeline_dir)
        except requests.HTTPError as error:
            failures += 1
            status = error.response.status_code if error.response is not None else None
            append_ndjson(args.failures_path, {"matchId": match_id, "status": status, "at": utc_now_iso()})
            print(f"[{index}/{len(todo)}] {match_id} failed with HTTP {status}; logged and skipped.")
            continue
        if index % 25 == 0 or index == len(todo):
            print(f"[{index}/{len(todo)}] stored {match_id} ({failures} failures so far)")

    print(f"Done. {len(todo) - failures} fetched, {failures} failed (see {args.failures_path}).")


if __name__ == "__main__":
    run(parse_args())
