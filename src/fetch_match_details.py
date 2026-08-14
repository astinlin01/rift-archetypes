"""
Stage B: fetch full match payloads from deduped match IDs.

Run from repo root:
    python src/fetch_match_details.py
"""

from __future__ import annotations

import argparse
from typing import Any

import requests

from pipeline_io import (
    append_ndjson,
    collect_match_ids_from_matches_ndjson,
    iter_ndjson,
    load_json,
    ndjson_file_exists,
    save_json_atomic,
    utc_now_iso,
)
from riot_client import RiotClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch full Riot match payloads from match IDs.")
    parser.add_argument("--platform", default="na1")
    parser.add_argument("--match-ids-path", default="data/raw/match_ids.ndjson")
    parser.add_argument("--output-path", default="data/raw/matches.ndjson")
    parser.add_argument("--checkpoint-path", default="data/raw/checkpoints/match_details_checkpoint.json")
    parser.add_argument("--failures-path", default="data/raw/failures/match_details_failures.ndjson")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--max-errors", type=int, default=50)
    parser.add_argument("--max-match-ids", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_match_ids(path: str) -> list[str]:
    match_ids: list[str] = []
    seen: set[str] = set()

    for row in iter_ndjson(path):
        match_id = row.get("match_id")
        if not match_id or match_id in seen:
            continue
        seen.add(match_id)
        match_ids.append(match_id)

    if not match_ids:
        raise RuntimeError(f"No match IDs found at {path}")
    return match_ids


def status_code_from_http_error(error: requests.HTTPError) -> int | None:
    response = getattr(error, "response", None)
    if response is None:
        return None
    return response.status_code


def save_checkpoint(
    checkpoint_path: str,
    next_match_index: int,
    total_match_ids: int,
    fetched_matches: int,
    failed_matches: int,
    output_path: str,
) -> None:
    payload = {
        "next_match_index": next_match_index,
        "total_match_ids": total_match_ids,
        "fetched_matches": fetched_matches,
        "failed_matches": failed_matches,
        "output_path": output_path,
        "updated_at": utc_now_iso(),
    }
    save_json_atomic(checkpoint_path, payload)


def run(args: argparse.Namespace) -> None:
    match_ids = load_match_ids(args.match_ids_path)
    if args.max_match_ids > 0:
        match_ids = match_ids[: args.max_match_ids]

    checkpoint: dict[str, Any] = {}
    start_index = 0
    fetched_set: set[str] = set()

    if args.resume:
        checkpoint = load_json(args.checkpoint_path, default={}) or {}
        start_index = int(checkpoint.get("next_match_index", 0))
        if ndjson_file_exists(args.output_path):
            fetched_set = collect_match_ids_from_matches_ndjson(args.output_path)

    client = RiotClient(platform=args.platform)

    fetched_count = 0
    failed_count = 0

    for idx in range(start_index, len(match_ids)):
        match_id = match_ids[idx]

        if match_id in fetched_set:
            continue

        print(f"[{idx + 1}/{len(match_ids)}] Fetching match {match_id}...")

        try:
            match_payload = client.get_match(match_id)
        except requests.HTTPError as http_error:
            failed_count += 1
            status_code = status_code_from_http_error(http_error)
            append_ndjson(
                args.failures_path,
                {
                    "stage": "match_details",
                    "match_id": match_id,
                    "status_code": status_code,
                    "error": str(http_error),
                    "logged_at": utc_now_iso(),
                },
            )
            print(f"Failed match {match_id} with status {status_code}. Logged and continuing.")

            if failed_count >= args.max_errors:
                print(f"Reached max_errors={args.max_errors}. Stopping early.")
                save_checkpoint(
                    checkpoint_path=args.checkpoint_path,
                    next_match_index=idx,
                    total_match_ids=len(match_ids),
                    fetched_matches=len(fetched_set),
                    failed_matches=failed_count,
                    output_path=args.output_path,
                )
                return

            continue

        append_ndjson(args.output_path, match_payload)
        fetched_set.add(match_id)
        fetched_count += 1

        if fetched_count % args.checkpoint_every == 0:
            save_checkpoint(
                checkpoint_path=args.checkpoint_path,
                next_match_index=idx + 1,
                total_match_ids=len(match_ids),
                fetched_matches=len(fetched_set),
                failed_matches=failed_count,
                output_path=args.output_path,
            )
            print(
                "Checkpoint saved "
                f"(fetched_matches={len(fetched_set)}, failed_matches={failed_count})."
            )

    save_checkpoint(
        checkpoint_path=args.checkpoint_path,
        next_match_index=len(match_ids),
        total_match_ids=len(match_ids),
        fetched_matches=len(fetched_set),
        failed_matches=failed_count,
        output_path=args.output_path,
    )

    print(
        "Done. "
        f"Fetched new matches this run: {fetched_count}. "
        f"Total fetched matches on disk: {len(fetched_set)}. "
        f"Failures this run: {failed_count}."
    )


if __name__ == "__main__":
    run(parse_args())
