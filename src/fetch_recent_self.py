"""
Fetch your own ranked match history without scanning the full league pool.

Resolves one or more Riot IDs and paginates through match-v5 (100 IDs per
call, the API max) until it runs out of history or hits --max-games, then
appends any new match IDs (plus full match payloads) straight into the same
data/raw/match_ids.ndjson and data/raw/matches.ndjson files the main pipeline
uses, so build_features.py picks them up on the next run. Safe to re-run --
already-fetched match IDs/details are skipped.

Run from repo root:
    python src/fetch_recent_self.py                        # all available history
    python src/fetch_recent_self.py --riot-id "yourname#na1" --max-games 50
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import requests

from pipeline_io import (
    append_ndjson,
    collect_match_ids_from_matches_ndjson,
    collect_values_from_ndjson,
    ndjson_file_exists,
    utc_now_iso,
)
from accounts import require_riot_ids
from riot_client import RiotClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch your own recent matches only.")
    parser.add_argument("--platform", default="na1")
    parser.add_argument("--queue", type=int, default=420)
    parser.add_argument("--page-size", type=int, default=100, help="Match IDs per API call (100 is the Riot max).")
    parser.add_argument(
        "--max-games",
        type=int,
        default=0,
        help="Stop after this many match IDs per account. 0 (default) pulls all available history.",
    )
    parser.add_argument("--start", type=int, default=0, help="Riot match-history offset to start paginating from.")
    parser.add_argument(
        "--since",
        default=None,
        help="Only fetch games on/after this date (YYYY-MM-DD), e.g. the current season's start. "
        "Default: all available history.",
    )
    parser.add_argument(
        "--riot-id",
        action="append",
        default=[],
        help="Riot ID(s) as gameName#tagLine. Repeatable. Defaults to RIOT_IDS in .env.",
    )
    parser.add_argument("--match-ids-path", default="data/raw/match_ids.ndjson")
    parser.add_argument("--matches-path", default="data/raw/matches.ndjson")
    parser.add_argument("--failures-path", default="data/raw/failures/fetch_recent_self_failures.ndjson")
    return parser.parse_args()


def status_code_from_http_error(error: requests.HTTPError) -> int | None:
    response = getattr(error, "response", None)
    return response.status_code if response is not None else None


def run(args: argparse.Namespace) -> None:
    riot_ids = require_riot_ids(args.riot_id)
    client = RiotClient(platform=args.platform)
    start_time = None
    if args.since:
        start_time = int(datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())

    seen_match_ids: set[str] = set()
    if ndjson_file_exists(args.match_ids_path):
        seen_match_ids = collect_values_from_ndjson(args.match_ids_path, key="match_id")

    fetched_match_ids: set[str] = set()
    if ndjson_file_exists(args.matches_path):
        fetched_match_ids = collect_match_ids_from_matches_ndjson(args.matches_path)

    total_new_ids = 0
    total_new_matches = 0

    for riot_id in riot_ids:
        game_name, _, tag_line = riot_id.partition("#")
        if not game_name or not tag_line:
            print(f"Skipping invalid Riot ID format: {riot_id}")
            continue

        try:
            account = client.get_account_by_riot_id_any_region(
                game_name=game_name.strip(), tag_line=tag_line.strip()
            )
        except requests.HTTPError as http_error:
            print(f"Failed to resolve {riot_id}: {http_error}")
            continue

        puuid = account.get("puuid")
        if not puuid:
            print(f"Riot ID {riot_id} resolved without puuid. Skipping.")
            continue

        print(f"Fetching match history for {riot_id} (page size {args.page_size})...")
        match_ids: list[str] = []
        start = args.start
        while True:
            batch = client.get_match_ids(
                puuid=puuid, queue=args.queue, count=args.page_size, start=start, start_time=start_time
            )
            if not batch:
                break
            match_ids.extend(batch)
            start += len(batch)
            print(f"  ...{len(match_ids)} match ID(s) so far")
            if len(batch) < args.page_size:
                break  # fewer than a full page means we've reached the end of history
            if args.max_games > 0 and len(match_ids) >= args.max_games:
                match_ids = match_ids[: args.max_games]
                break

        new_ids = [match_id for match_id in match_ids if match_id not in seen_match_ids]
        for match_id in new_ids:
            append_ndjson(
                args.match_ids_path,
                {
                    "match_id": match_id,
                    "source_puuid": puuid,
                    "source_summoner_id": None,
                    "source_summoner_name": None,
                    "source_riot_id": riot_id,
                    "queue": args.queue,
                    "fetched_at": utc_now_iso(),
                },
            )
            seen_match_ids.add(match_id)
        total_new_ids += len(new_ids)
        print(f"  {len(new_ids)} new match ID(s) out of {len(match_ids)} total available.")

        ids_to_fetch_details = [match_id for match_id in match_ids if match_id not in fetched_match_ids]
        for match_id in ids_to_fetch_details:
            print(f"  Fetching match details for {match_id}...")
            try:
                match_payload = client.get_match(match_id)
            except requests.HTTPError as http_error:
                status_code = status_code_from_http_error(http_error)
                append_ndjson(
                    args.failures_path,
                    {
                        "stage": "fetch_recent_self",
                        "match_id": match_id,
                        "status_code": status_code,
                        "error": str(http_error),
                        "logged_at": utc_now_iso(),
                    },
                )
                print(f"  Failed match {match_id} with status {status_code}. Logged and continuing.")
                continue

            append_ndjson(args.matches_path, match_payload)
            fetched_match_ids.add(match_id)
            total_new_matches += 1

    print(
        "\nDone. "
        f"New match IDs added: {total_new_ids}. "
        f"New match payloads fetched: {total_new_matches}."
    )


if __name__ == "__main__":
    run(parse_args())
