"""
Stage A: collect match IDs for a pool of players and dedupe globally.

Run from repo root:
    python src/fetch_match_ids.py
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import requests

from pipeline_io import (
    append_ndjson,
    collect_values_from_ndjson,
    load_json,
    ndjson_file_exists,
    save_json_atomic,
    utc_now_iso,
)
from riot_client import RiotClient


# Always include these personal accounts even if they are outside the ranked pool slice.
DEFAULT_ALWAYS_INCLUDE_RIOT_IDS = [
    "river3#na1",
    "river#ell",
    "riverell#888",
    "riverell#ttv",
    "xlsteemo#na1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and dedupe match IDs across a league entry pool.")
    parser.add_argument("--platform", default="na1")
    parser.add_argument("--queue", type=int, default=420)
    parser.add_argument("--matches-per-player", type=int, default=20)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-players", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--league-entries-path", default="data/raw/league_entries.json")
    parser.add_argument("--output-path", default="data/raw/match_ids.ndjson")
    parser.add_argument("--checkpoint-path", default="data/raw/checkpoints/match_ids_checkpoint.json")
    parser.add_argument("--failures-path", default="data/raw/failures/match_ids_failures.ndjson")
    parser.add_argument(
        "--include-puuid",
        action="append",
        default=[],
        help="Guarantee these puuids are included even if not in league_entries.",
    )
    parser.add_argument(
        "--include-puuid-file",
        default="",
        help="Path to a newline-delimited file of puuids to always include.",
    )
    parser.add_argument(
        "--include-riot-id",
        action="append",
        default=[],
        help="Additional Riot IDs (gameName#tagLine) to resolve and always include.",
    )
    parser.add_argument(
        "--skip-default-riot-ids",
        action="store_true",
        help="Disable built-in always-include Riot IDs.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_puuid_rows(league_entries_path: str) -> list[dict[str, str | None]]:
    entries = load_json(league_entries_path, default=[])
    if not entries:
        raise RuntimeError(f"No league entries found at {league_entries_path}")

    seen: set[str] = set()
    ordered_rows: list[dict[str, str | None]] = []
    for row in entries:
        puuid = row.get("puuid")
        if not puuid or puuid in seen:
            continue
        seen.add(puuid)
        ordered_rows.append(
            {
                "puuid": puuid,
                "summoner_id": row.get("summonerId"),
                "summoner_name": row.get("summonerName"),
            }
        )

    return ordered_rows


def load_extra_puuids(include_puuids: list[str], include_puuid_file: str) -> list[str]:
    extras = [puuid.strip() for puuid in include_puuids if puuid and puuid.strip()]

    if include_puuid_file:
        if not os.path.exists(include_puuid_file):
            raise RuntimeError(f"include-puuid-file not found: {include_puuid_file}")
        with open(include_puuid_file, "r", encoding="utf-8") as infile:
            for line in infile:
                candidate = line.strip()
                if not candidate or candidate.startswith("#"):
                    continue
                extras.append(candidate)

    return extras


def merge_with_extra_puuids(
    puuid_rows: list[dict[str, str | None]],
    extra_puuids: list[str],
) -> list[dict[str, str | None]]:
    seen = {row["puuid"] for row in puuid_rows if row.get("puuid")}
    merged = list(puuid_rows)

    for puuid in extra_puuids:
        if puuid in seen:
            continue
        merged.append(
            {
                "puuid": puuid,
                "summoner_id": None,
                "summoner_name": None,
            }
        )
        seen.add(puuid)

    return merged


def parse_riot_id(riot_id: str) -> tuple[str, str] | None:
    value = riot_id.strip()
    if not value or "#" not in value:
        return None
    game_name, tag_line = value.split("#", 1)
    game_name = game_name.strip()
    tag_line = tag_line.strip()
    if not game_name or not tag_line:
        return None
    return game_name, tag_line


def resolve_riot_ids_to_rows(client: RiotClient, riot_ids: list[str]) -> list[dict[str, str | None]]:
    resolved_rows: list[dict[str, str | None]] = []
    seen_puuids: set[str] = set()

    for riot_id in riot_ids:
        parsed = parse_riot_id(riot_id)
        if parsed is None:
            print(f"Skipping invalid Riot ID format: {riot_id}")
            continue

        game_name, tag_line = parsed
        try:
            account = client.get_account_by_riot_id_any_region(game_name=game_name, tag_line=tag_line)
        except Exception as exc:
            print(f"Failed to resolve Riot ID {riot_id}: {exc}")
            continue

        puuid = account.get("puuid")
        if not puuid:
            print(f"Riot ID {riot_id} resolved without puuid. Skipping.")
            continue
        if puuid in seen_puuids:
            continue

        seen_puuids.add(puuid)
        resolved_rows.append(
            {
                "puuid": puuid,
                "summoner_id": None,
                "summoner_name": None,
                "riot_id": f"{account.get('gameName', game_name)}#{account.get('tagLine', tag_line)}",
            }
        )

    return resolved_rows


def merge_with_extra_rows(
    puuid_rows: list[dict[str, str | None]],
    extra_rows: list[dict[str, str | None]],
) -> list[dict[str, str | None]]:
    seen = {row["puuid"] for row in puuid_rows if row.get("puuid")}
    merged = list(puuid_rows)

    for row in extra_rows:
        puuid = row.get("puuid")
        if not puuid or puuid in seen:
            continue
        merged.append(row)
        seen.add(puuid)

    return merged


def status_code_from_http_error(error: requests.HTTPError) -> int | None:
    response = getattr(error, "response", None)
    if response is None:
        return None
    return response.status_code


def save_checkpoint(
    checkpoint_path: str,
    next_puuid_index: int,
    total_puuids: int,
    processed_puuids: int,
    unique_match_ids: int,
    output_path: str,
) -> None:
    payload = {
        "next_puuid_index": next_puuid_index,
        "total_puuids": total_puuids,
        "processed_puuids": processed_puuids,
        "unique_match_ids": unique_match_ids,
        "output_path": output_path,
        "updated_at": utc_now_iso(),
    }
    save_json_atomic(checkpoint_path, payload)


def run(args: argparse.Namespace) -> None:
    puuid_rows = load_puuid_rows(args.league_entries_path)
    if args.max_players > 0:
        puuid_rows = puuid_rows[: args.max_players]

    checkpoint: dict[str, Any] = {}
    start_index = 0
    seen_match_ids: set[str] = set()

    if ndjson_file_exists(args.output_path):
        seen_match_ids = collect_values_from_ndjson(args.output_path, key="match_id")

    if args.resume:
        checkpoint = load_json(args.checkpoint_path, default={}) or {}
        start_index = int(checkpoint.get("next_puuid_index", 0))

    client = RiotClient(platform=args.platform)

    extra_puuids = load_extra_puuids(args.include_puuid, args.include_puuid_file)
    extra_puuid_rows = merge_with_extra_puuids([], extra_puuids)

    riot_ids = [] if args.skip_default_riot_ids else list(DEFAULT_ALWAYS_INCLUDE_RIOT_IDS)
    riot_ids.extend(args.include_riot_id)
    riot_id_rows = resolve_riot_ids_to_rows(client, riot_ids)

    puuid_rows = merge_with_extra_rows(puuid_rows, extra_puuid_rows)
    puuid_rows = merge_with_extra_rows(puuid_rows, riot_id_rows)

    if start_index > len(puuid_rows):
        print(
            "Resume checkpoint index is beyond the current pool size. "
            "Resetting to 0 for this run."
        )
        start_index = 0

    if extra_puuids:
        print(f"Added {len(extra_puuids)} requested puuid include(s) to the pool.")
    if riot_id_rows:
        print(f"Added {len(riot_id_rows)} Riot ID include(s) to the pool.")

    processed_puuids = start_index
    new_match_ids = 0

    for idx in range(start_index, len(puuid_rows)):
        row = puuid_rows[idx]
        puuid = row["puuid"]
        summoner_id = row.get("summoner_id")
        summoner_name = row.get("summoner_name")
        riot_id = row.get("riot_id")

        log_identity = summoner_id or riot_id or f"puuid:{puuid[:12]}"
        if summoner_name:
            log_identity = f"{log_identity} ({summoner_name})"
        print(f"[{idx + 1}/{len(puuid_rows)}] Fetching match IDs for {log_identity}...")

        try:
            match_ids = client.get_match_ids(
                puuid=puuid,
                queue=args.queue,
                count=args.matches_per_player,
                start=args.start,
            )
        except requests.HTTPError as http_error:
            status_code = status_code_from_http_error(http_error)
            append_ndjson(
                args.failures_path,
                {
                    "stage": "match_ids",
                    "puuid": puuid,
                    "summoner_id": summoner_id,
                    "summoner_name": summoner_name,
                    "riot_id": riot_id,
                    "status_code": status_code,
                    "error": str(http_error),
                    "logged_at": utc_now_iso(),
                },
            )
            print(f"Failed for {log_identity} with status {status_code}. Logged and continuing.")
            processed_puuids += 1
            continue

        for match_id in match_ids:
            if match_id in seen_match_ids:
                continue
            seen_match_ids.add(match_id)
            append_ndjson(
                args.output_path,
                {
                    "match_id": match_id,
                    "source_puuid": puuid,
                    "source_summoner_id": summoner_id,
                    "source_summoner_name": summoner_name,
                    "source_riot_id": riot_id,
                    "queue": args.queue,
                    "fetched_at": utc_now_iso(),
                },
            )
            new_match_ids += 1

        processed_puuids += 1

        if processed_puuids % args.checkpoint_every == 0:
            save_checkpoint(
                checkpoint_path=args.checkpoint_path,
                next_puuid_index=idx + 1,
                total_puuids=len(puuid_rows),
                processed_puuids=processed_puuids,
                unique_match_ids=len(seen_match_ids),
                output_path=args.output_path,
            )
            print(
                "Checkpoint saved "
                f"(processed_puuids={processed_puuids}, unique_match_ids={len(seen_match_ids)})."
            )

    save_checkpoint(
        checkpoint_path=args.checkpoint_path,
        next_puuid_index=len(puuid_rows),
        total_puuids=len(puuid_rows),
        processed_puuids=processed_puuids,
        unique_match_ids=len(seen_match_ids),
        output_path=args.output_path,
    )

    print(
        "Done. "
        f"Processed {processed_puuids}/{len(puuid_rows)} puuids. "
        f"New unique match IDs this run: {new_match_ids}. "
        f"Total unique match IDs on disk: {len(seen_match_ids)}."
    )


if __name__ == "__main__":
    run(parse_args())
