"""
Build a per-participant feature table for playstyle clustering.

Reads data/raw/matches.ndjson (via pipeline_io) plus any data/raw/league_entries*.json
pool files (to attach a source tier per puuid where known), applies the curated
field list from feature_config.py, and writes a flat feature table.

Run from repo root:
    python src/build_features.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd

from feature_config import (
    ALL_FEATURE_FIELDS,
    FEATURE_COLUMN_NAMES,
    VALID_ROLES,
    add_derived_features,
    get_nested,
    season_from_patch,
)
from pipeline_io import iter_ndjson
from playstyle_space import add_game_state_columns

TIER_FROM_FILENAME = {
    "challenger": "CHALLENGER",
    "grandmaster": "GRANDMASTER",
    "master": "MASTER",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the per-participant feature table.")
    parser.add_argument("--matches-path", default="data/raw/matches.ndjson")
    parser.add_argument(
        "--league-entries-glob",
        default="data/raw/league_entries*.json",
        help="Glob for league pool files used to attach a source tier per puuid.",
    )
    parser.add_argument("--output-path", default="data/processed/participant_features.parquet")
    parser.add_argument(
        "--min-game-length-seconds",
        type=int,
        default=300,
        help="Drop remakes/early-surrender games shorter than this (default 5 minutes).",
    )
    return parser.parse_args()


def infer_tier_from_filename(path: str) -> str:
    lowered = os.path.basename(path).lower()
    for keyword, tier in TIER_FROM_FILENAME.items():
        if keyword in lowered:
            return tier
    return "UNKNOWN"


def infer_tier_for_row(path_fallback_tier: str, row: dict) -> str:
    # Non-apex tiers (Diamond and below) carry "tier" per row from league-v4.
    # Apex tiers (Master/GM/Challenger) come from a top-league endpoint whose
    # entries omit "tier" entirely, so those rows fall back to the filename --
    # this matters even for merged/appended pool files, where entries[0]'s
    # tier is not representative of every row in the file.
    if row.get("tier"):
        return str(row["tier"]).upper()
    return path_fallback_tier


def infer_region_for_row(row: dict, path: str) -> str:
    # fetch_league_pool.py tags "platform" on every row it saves going forward.
    # Older files predate that (all NA1-only at the time), and filenames with no
    # region marker are assumed NA1 too.
    if row.get("platform"):
        return str(row["platform"]).upper()
    if "_kr_" in os.path.basename(path).lower():
        return "KR"
    return "NA1"


def build_tier_lookup(league_entries_glob: str) -> dict[str, dict[str, str]]:
    """Map puuid -> {"tier": ..., "region": ...}, scanning all matching league pool files.

    Rows from later files win on conflict (arbitrary but rare in practice --
    a puuid should only appear in one tier/region pool at collection time).
    """
    info_by_puuid: dict[str, dict[str, str]] = {}
    for path in sorted(glob.glob(league_entries_glob)):
        with open(path, "r", encoding="utf-8") as infile:
            entries = json.load(infile)
        if not entries:
            continue
        path_fallback_tier = infer_tier_from_filename(path)
        for row in entries:
            puuid = row.get("puuid")
            if not puuid:
                continue
            info_by_puuid[puuid] = {
                "tier": infer_tier_for_row(path_fallback_tier, row),
                "region": infer_region_for_row(row, path),
            }
    return info_by_puuid


def extract_participant_row(
    match_id: str,
    queue_id: int,
    match_context: dict,
    participant: dict,
    info_by_puuid: dict[str, dict[str, str]],
    min_game_length_seconds: int,
) -> dict | None:
    team_position = participant.get("teamPosition")
    if team_position not in VALID_ROLES:
        return None

    game_length_seconds = get_nested(participant, "challenges.gameLength")
    if not game_length_seconds:
        game_length_seconds = participant.get("timePlayed")
    if not game_length_seconds or game_length_seconds < min_game_length_seconds:
        return None
    minutes = game_length_seconds / 60.0

    puuid = participant.get("puuid")
    puuid_info = info_by_puuid.get(puuid, {})

    row: dict = {
        "matchId": match_id,
        "puuid": puuid,
        "championName": participant.get("championName"),
        "teamPosition": team_position,
        "win": bool(participant.get("win")),
        "queueId": queue_id,
        "gameLength": game_length_seconds,
        "tier": puuid_info.get("tier", "UNKNOWN"),
        "region": puuid_info.get("region", "UNKNOWN"),
        **match_context,
    }

    for field in ALL_FEATURE_FIELDS:
        value = get_nested(participant, field.path)
        if value is not None and field.normalize == "per_minute":
            value = value / minutes
        row[field.name] = value

    return row


def build_feature_rows(
    matches_path: str,
    info_by_puuid: dict[str, dict[str, str]],
    min_game_length_seconds: int,
) -> list[dict]:
    rows: list[dict] = []
    match_count = 0

    for match in iter_ndjson(matches_path):
        match_count += 1
        match_id = match.get("metadata", {}).get("matchId")
        info = match.get("info", {})
        queue_id = info.get("queueId")
        participants = info.get("participants", [])
        # "16.17.712.3911" -> patch "16.17", season 16. Comparisons are season-matched
        # downstream, since stat baselines shift a lot between seasons.
        patch = ".".join(str(info.get("gameVersion", "")).split(".")[:2]) or None
        match_context = {
            "gameCreation": info.get("gameCreation"),
            "patch": patch,
            "season": season_from_patch(patch),
        }

        for participant in participants:
            row = extract_participant_row(
                match_id, queue_id, match_context, participant, info_by_puuid, min_game_length_seconds
            )
            if row is not None:
                rows.append(row)

    print(f"Matches read: {match_count}")
    return rows


def print_sanity_report(df: pd.DataFrame) -> None:
    print(f"\nTotal feature rows: {len(df)}")
    print("\nRows by role:")
    print(df["teamPosition"].value_counts())
    print("\nRows by tier:")
    print(df["tier"].value_counts())
    print("\nRows by region:")
    print(df["region"].value_counts())
    print("\nRows by season:")
    print(df["season"].value_counts().sort_index())

    feature_columns = FEATURE_COLUMN_NAMES
    nan_counts = df[feature_columns].isna().sum()
    nonzero_nan = nan_counts[nan_counts > 0]
    if len(nonzero_nan) > 0:
        print("\nColumns with missing values:")
        print(nonzero_nan.sort_values(ascending=False))
    else:
        print("\nNo missing values in feature columns.")


def run(args: argparse.Namespace) -> None:
    if not os.path.exists(args.matches_path):
        raise FileNotFoundError(f"Matches file not found: {args.matches_path}")

    info_by_puuid = build_tier_lookup(args.league_entries_glob)
    print(f"Loaded tier/region lookup for {len(info_by_puuid)} puuids.")

    rows = build_feature_rows(args.matches_path, info_by_puuid, args.min_game_length_seconds)
    # Team kill/gold differentials per minute -- used to outcome-adjust features before
    # clustering (see playstyle_space.py).
    df = add_game_state_columns(add_derived_features(pd.DataFrame(rows)))

    print_sanity_report(df)

    parent = os.path.dirname(args.output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    df.to_parquet(args.output_path, index=False)
    print(f"\nFeature table written to: {args.output_path}")


if __name__ == "__main__":
    run(parse_args())
