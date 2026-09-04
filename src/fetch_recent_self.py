"""
Fetch your own most recent ranked matches without scanning the full league pool.

Resolves one or more Riot IDs, pulls their most recent match IDs, and appends
any new ones (plus full match payloads) straight into the same
data/raw/match_ids.ndjson and data/raw/matches.ndjson files the main pipeline
uses, so build_features.py picks them up on the next run.

Run from repo root:
    python src/fetch_recent_self.py
    python src/fetch_recent_self.py --riot-id "river3#na1" --count 20
"""

from __future__ import annotations

import argparse

import requests

from pipeline_io import (
    append_ndjson,
    collect_match_ids_from_matches_ndjson,
    collect_values_from_ndjson,
    ndjson_file_exists,
    utc_now_iso,
)
from riot_client import RiotClient

# Mirrors fetch_match_ids.py's always-include accounts.
DEFAULT_RIOT_IDS = [
    "river3#na1",
    "river#ell",
    "riverell#888",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch your own recent matches only.")
    parser.add_argument("--platform", default="na1")
    parser.add_argument("--queue", type=int, default=420)
    parser.add_argument("--count", type=int, default=20, help="Most recent matches to pull per account.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--riot-id",
        action="append",
        default=[],
        help="Riot ID(s) as gameName#tagLine. Repeatable. Defaults to the built-in accounts.",
    )
    parser.add_argument("--match-ids-path", default="data/raw/match_ids.ndjson")
    parser.add_argument("--matches-path", default="data/raw/matches.ndjson")
    parser.add_argument("--failures-path", default="data/raw/failures/fetch_recent_self_failures.ndjson")
    return parser.parse_args()


def status_code_from_http_error(error: requests.HTTPError) -> int | None:
    response = getattr(error, "response", None)
    return response.status_code if response is not None else None


def run(args: argparse.Namespace) -> None:
    riot_ids = args.riot_id if args.riot_id else DEFAULT_RIOT_IDS
    client = RiotClient(platform=args.platform)

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

        print(f"Fetching {args.count} most recent matches for {riot_id}...")
        match_ids = client.get_match_ids(
            puuid=puuid, queue=args.queue, count=args.count, start=args.start
        )

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
        print(f"  {len(new_ids)} new match ID(s) out of {len(match_ids)} fetched.")

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
