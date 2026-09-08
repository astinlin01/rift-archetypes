"""
Build a per-player-per-game feature table from Oracle's Elixir's professional
match data CSV, for comparing your own stats against LCK/LPL pros.

Oracle's Elixir download: https://oracleselixir.com/tools/downloads (Google Drive-hosted;
their download quota is sometimes exhausted -- if `gdown` fails, download manually in a
browser and place the file at --input-path).

Only a subset of our own solo-queue features have a clean equivalent in Oracle's Elixir's
schema (no challenges-style derived stats, no gameLength-based per-minute normalization
for a few fields) -- see PRO_FEATURE_MAP for exactly which features are compared and how
they're derived from OE's raw columns.

Run from repo root:
    python src/build_pro_features.py
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from feature_config import VALID_ROLES

DEFAULT_LEAGUES = ["LCK", "LPL"]
MIN_GAME_LENGTH_SECONDS = 300

POSITION_TO_ROLE = {
    "top": "TOP",
    "jng": "JUNGLE",
    "mid": "MIDDLE",
    "bot": "BOTTOM",
    "sup": "UTILITY",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the pro player feature table from Oracle's Elixir data.")
    parser.add_argument("--input-path", default="data/raw/oracles_elixir/2026_match_data.csv")
    parser.add_argument(
        "--leagues",
        nargs="+",
        default=DEFAULT_LEAGUES,
        help="Oracle's Elixir league codes to include (e.g. LCK LPL). See the 'league' column for all codes.",
    )
    parser.add_argument("--output-path", default="data/processed/pro_player_features.parquet")
    return parser.parse_args()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    minutes = df["gamelength"] / 60.0

    features = pd.DataFrame(
        {
            "gameid": df["gameid"],
            "league": df["league"],
            "playername": df["playername"],
            "teamname": df["teamname"],
            "champion": df["champion"],
            "role": df["position"].map(POSITION_TO_ROLE),
            "win": df["result"] == 1,
            "gameLength": df["gamelength"],
            # Direct equivalents of our own HEADLINE_FEATURES:
            "damage_per_minute": df["dpm"],
            "cs_per_min": df["cspm"],
            "vision_score_per_minute": df["vspm"],
            "team_damage_pct": df["damageshare"],
            "damage_taken_per_min": df["damagetakenperminute"],
            "deaths_per_min": df["deaths"] / minutes,
            "kda": (df["kills"] + df["assists"]) / df["deaths"].clip(lower=1),
            "kill_participation": (df["kills"] + df["assists"]) / df["teamkills"].clip(lower=1),
            "control_wards_per_min": df["controlwardsbought"] / minutes,
            # gold_earned_per_min is in HEADLINE_FEATURES but has a clean OE equivalent too:
            "gold_earned_per_min": df["earned gpm"],
        }
    )
    return features


def run(args: argparse.Namespace) -> None:
    if not os.path.exists(args.input_path):
        raise FileNotFoundError(
            f"Oracle's Elixir CSV not found at {args.input_path}. "
            "Download it from https://oracleselixir.com/tools/downloads (or via gdown -- "
            "their Drive quota is sometimes exhausted, in which case download manually in a browser)."
        )

    print(f"Reading {args.input_path}...")
    df = pd.read_csv(args.input_path, low_memory=False)
    print(f"Total rows: {len(df)}")

    df = df[df["position"] != "team"]
    df = df[df["datacompleteness"] == "complete"]
    df = df[df["league"].isin(args.leagues)]
    df = df[df["gamelength"] >= MIN_GAME_LENGTH_SECONDS]
    df = df[df["position"].isin(POSITION_TO_ROLE)]
    print(f"Rows after filtering to leagues {args.leagues}, complete data, valid role, >= {MIN_GAME_LENGTH_SECONDS}s: {len(df)}")

    features = build_features(df)
    features = features.dropna(subset=["role", "kda", "damage_per_minute"])
    print(f"Final feature rows: {len(features)}")

    print("\nRows by league:")
    print(features["league"].value_counts())
    print("\nRows by role:")
    print(features["role"].value_counts())

    parent = os.path.dirname(args.output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    features.to_parquet(args.output_path, index=False)
    print(f"\nPro feature table written to: {args.output_path}")


if __name__ == "__main__":
    run(parse_args())
