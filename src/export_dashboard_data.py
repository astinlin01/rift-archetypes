"""
Export a static JSON payload for the self-analysis dashboard (an Artifact,
not part of this repo). Pulls from the already-built feature table, fitted
cluster models, and self_comparison.csv -- run build_features.py,
build_clusters.py, and compare_self.py first.

Run from repo root:
    python src/export_dashboard_data.py --riot-id "river3#na1"
"""

from __future__ import annotations

import argparse
import json
import os
import random

import joblib
import pandas as pd

from feature_config import ALL_FEATURE_FIELDS
from pipeline_io import iter_ndjson

FEATURE_COLUMNS = [f.name for f in ALL_FEATURE_FIELDS]

# Curated subset + display labels for the headline percentile comparison.
# invert=True means a lower raw value is typically the better outcome for this role.
HEADLINE_FEATURES = [
    ("damage_per_minute", "Damage / min", False),
    ("gold_earned_per_min", "Gold / min", False),
    ("vision_score_per_minute", "Vision score / min", False),
    ("kda", "KDA", False),
    ("kill_participation", "Kill participation", False),
    ("team_damage_pct", "Team damage share", False),
    ("cs_per_min", "CS / min", False),
    ("cc_time_per_min", "CC time on enemies / min", False),
    ("damage_taken_per_min", "Damage taken / min", True),
    ("control_wards_per_min", "Control wards / min", False),
    ("deaths_per_min", "Deaths / min", True),
    ("max_cs_advantage_lane_opp", "Max CS lead on opponent", False),
]

CLUSTER_NAME_HINTS = {
    # Fallback label if we can't auto-name from win rate + damage rank.
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export dashboard JSON payload.")
    parser.add_argument("--features-path", default="data/processed/participant_features.parquet")
    parser.add_argument("--clusters-path", default="data/processed/participant_clusters.parquet")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--matches-path", default="data/raw/matches.ndjson")
    parser.add_argument("--riot-id", default="river3#na1")
    parser.add_argument("--reference-tier", default="CHALLENGER")
    parser.add_argument("--focus-role", default="BOTTOM")
    parser.add_argument(
        "--recent-games",
        type=int,
        default=0,
        help="Most recent games to include. 0 (default) includes all games found for this puuid.",
    )
    parser.add_argument("--population-sample", type=int, default=700)
    parser.add_argument("--output-path", default="data/processed/dashboard_data.json")
    return parser.parse_args()


def resolve_puuid_from_match_ids(match_ids_path: str, riot_id: str) -> str:
    for row in iter_ndjson(match_ids_path):
        if row.get("source_riot_id") == riot_id:
            return row["source_puuid"]
    raise RuntimeError(f"No match_ids.ndjson rows found for riot id {riot_id}")


def load_game_creation_by_match_id(matches_path: str, match_ids: set[str]) -> dict[str, int]:
    creation_by_id: dict[str, int] = {}
    for match in iter_ndjson(matches_path):
        match_id = match.get("metadata", {}).get("matchId")
        if match_id in match_ids:
            creation_by_id[match_id] = match.get("info", {}).get("gameCreation")
    return creation_by_id


CLUSTER_NAMES_BY_WIN_RATE_RANK = {
    # rank 0 = lowest win rate cluster ... rank (k-1) = highest.
    0: "Passive & Behind",
    1: "Sustain Skirmisher",
    2: "Snowballing Carry",
    3: "High-Variance Carry",
}


def build_cluster_summary(clusters_df: pd.DataFrame, role: str) -> list[dict]:
    role_df = clusters_df[clusters_df["teamPosition"] == role]
    summary = (
        role_df.groupby("cluster")
        .agg(n=("cluster", "size"), win_rate=("win", "mean"), damage_per_minute=("damage_per_minute", "mean"))
        .reset_index()
        .sort_values("win_rate")
        .reset_index(drop=True)
    )
    summary["name"] = [
        CLUSTER_NAMES_BY_WIN_RATE_RANK.get(rank, f"Archetype {rank + 1}") for rank in range(len(summary))
    ]
    return summary.to_dict("records")


def build_scatter(role: str, role_df: pd.DataFrame, my_role_df: pd.DataFrame, models_dir: str, sample_size: int) -> dict:
    artifact = joblib.load(os.path.join(models_dir, f"{role.lower()}_cluster.joblib"))
    matrix = role_df[FEATURE_COLUMNS].fillna(role_df[FEATURE_COLUMNS].median(numeric_only=True))
    scaled = artifact["scaler"].transform(matrix)
    reduced = artifact["pca"].transform(scaled)

    population = list(zip(reduced[:, 0].tolist(), reduced[:, 1].tolist(), role_df["cluster"].tolist()))
    random.Random(42).shuffle(population)
    population = population[:sample_size]

    my_matrix = my_role_df[FEATURE_COLUMNS].fillna(role_df[FEATURE_COLUMNS].median(numeric_only=True))
    my_scaled = artifact["scaler"].transform(my_matrix)
    my_reduced = artifact["pca"].transform(my_scaled)
    my_clusters = artifact["kmeans"].predict(my_reduced)

    mine = []
    for point, cluster, (_, game_row) in zip(my_reduced.tolist(), my_clusters.tolist(), my_role_df.iterrows()):
        mine.append(
            {
                "x": point[0],
                "y": point[1],
                "cluster": int(cluster),
                "champion": game_row["championName"],
                "win": bool(game_row["win"]),
            }
        )

    return {
        "population": [{"x": x, "y": y, "cluster": int(c)} for x, y, c in population],
        "mine": mine,
    }


def run(args: argparse.Namespace) -> None:
    df = pd.read_parquet(args.features_path)
    clusters_df = pd.read_parquet(args.clusters_path)

    puuid = resolve_puuid_from_match_ids("data/raw/match_ids.ndjson", args.riot_id)
    my_df = df[df["puuid"] == puuid].copy()
    if my_df.empty:
        raise RuntimeError(f"No feature rows found for puuid {puuid}. Run build_features.py after fetching matches.")

    creation_by_id = load_game_creation_by_match_id(args.matches_path, set(my_df["matchId"]))
    my_df["gameCreation"] = my_df["matchId"].map(creation_by_id)
    my_df = my_df.sort_values("gameCreation", ascending=False)
    if args.recent_games > 0:
        my_df = my_df.head(args.recent_games)

    record = {
        "wins": int(my_df["win"].sum()),
        "losses": int((~my_df["win"]).sum()),
        "win_rate": round(float(my_df["win"].mean()), 3),
        "total_games": int(len(my_df)),
    }
    role_counts = my_df["teamPosition"].value_counts().to_dict()
    champion_counts = my_df["championName"].value_counts().to_dict()

    focus_role = args.focus_role
    reference_df = df[(df["tier"] == args.reference_tier.upper()) & (df["teamPosition"] == focus_role)]
    my_focus_df = my_df[my_df["teamPosition"] == focus_role]

    headline_rows = []
    for column, label, invert in HEADLINE_FEATURES:
        your_avg = float(my_focus_df[column].mean())
        ref_series = reference_df[column].dropna()
        ref_avg = float(ref_series.mean())
        if ref_series.empty or pd.isna(your_avg):
            percentile = 50.0
        else:
            from scipy.stats import percentileofscore

            percentile = float(percentileofscore(ref_series, your_avg, kind="mean"))

        # For "lower raw value is the better outcome" stats (deaths, damage taken),
        # flip the percentile so the dashboard's polarity always means the same
        # thing: right/positive = performing well, left/negative = performing poorly.
        performance_percentile = (100.0 - percentile) if invert else percentile

        headline_rows.append(
            {
                "feature": column,
                "label": label,
                "your_avg": round(your_avg, 3),
                "ref_avg": round(ref_avg, 3),
                "percentile": round(performance_percentile, 1),
                "invert": invert,
            }
        )

    role_df_full = clusters_df[clusters_df["teamPosition"] == focus_role].reset_index(drop=True)
    cluster_summary = build_cluster_summary(clusters_df, focus_role)

    my_cluster_assignments = joblib.load(os.path.join(args.models_dir, f"{focus_role.lower()}_cluster.joblib"))
    my_matrix = my_focus_df[FEATURE_COLUMNS].fillna(role_df_full[FEATURE_COLUMNS].median(numeric_only=True))
    my_scaled = my_cluster_assignments["scaler"].transform(my_matrix)
    my_reduced_full = my_cluster_assignments["pca"].transform(my_scaled)
    my_focus_df = my_focus_df.copy()
    my_focus_df["cluster"] = my_cluster_assignments["kmeans"].predict(my_reduced_full)

    your_cluster_counts = my_focus_df["cluster"].value_counts().to_dict()
    for row in cluster_summary:
        row["your_count"] = int(your_cluster_counts.get(row["cluster"], 0))
        row["win_rate"] = round(float(row["win_rate"]), 3)
        row["damage_per_minute"] = round(float(row["damage_per_minute"]), 1)

    scatter = build_scatter(focus_role, role_df_full, my_focus_df, args.models_dir, args.population_sample)

    games = []
    for _, row in my_df.iterrows():
        role = row["teamPosition"]
        cluster_id = None
        if role == focus_role:
            match_cluster_row = my_focus_df[my_focus_df["matchId"] == row["matchId"]]
            if not match_cluster_row.empty:
                cluster_id = int(match_cluster_row.iloc[0]["cluster"])
        games.append(
            {
                "matchId": row["matchId"],
                "champion": row["championName"],
                "role": role,
                "win": bool(row["win"]),
                "kda": round(float(row["kda"]), 2) if pd.notna(row["kda"]) else None,
                "cs_per_min": round(float(row["cs_per_min"]), 1) if pd.notna(row["cs_per_min"]) else None,
                "damage_per_minute": round(float(row["damage_per_minute"]), 0) if pd.notna(row["damage_per_minute"]) else None,
                "vision_score_per_minute": round(float(row["vision_score_per_minute"]), 2) if pd.notna(row["vision_score_per_minute"]) else None,
                "cluster": cluster_id,
                "gameCreation": int(row["gameCreation"]) if pd.notna(row["gameCreation"]) else None,
            }
        )

    payload = {
        "riot_id": args.riot_id,
        "reference_tier": args.reference_tier,
        "focus_role": focus_role,
        "record": record,
        "role_counts": role_counts,
        "champion_counts": champion_counts,
        "headline_features": headline_rows,
        "clusters": cluster_summary,
        "scatter": scatter,
        "games": games,
    }

    parent = os.path.dirname(args.output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as outfile:
        json.dump(payload, outfile, indent=2)
    print(f"Dashboard data written to: {args.output_path}")


if __name__ == "__main__":
    run(parse_args())
