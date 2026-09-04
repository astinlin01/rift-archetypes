"""
Export a static JSON payload for the interactive self-analysis dashboard
(published as a Claude Artifact, not part of this repo). Pulls from the
already-built feature table and per-role fitted scaler/PCA -- run
build_features.py and build_clusters.py first.

Unlike build_clusters.py (which picks one "official" k per role via
silhouette score for reuse elsewhere, e.g. compare_self.py), this script
fits several k values per role on the fly purely for the dashboard's
cluster-count slider. It also precomputes percentile reference curves per
(role, tier, feature) so the published dashboard can switch game window,
role, reference tier, and cluster count entirely client-side with no
server/recompute step.

Run from repo root:
    python src/export_dashboard_data.py --riot-id "river3#na1"
"""

from __future__ import annotations

import argparse
import json
import os
import random

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from feature_config import ALL_FEATURE_FIELDS, VALID_ROLES
from pipeline_io import iter_ndjson

FEATURE_COLUMNS = [f.name for f in ALL_FEATURE_FIELDS]
RANDOM_STATE = 42
K_OPTIONS = [3, 4, 5, 6]
MIN_REFERENCE_SAMPLE = 20  # skip a (role, tier) reference if fewer rows than this
BREAKPOINT_COUNT = 51  # percentile grid: 0, 2, 4, ..., 100

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

# Ordered from "worst win rate" to "best win rate" archetype flavor. For a given k,
# ranks are spread evenly across this spectrum so the two extremes always carry the
# same name regardless of how many clusters are chosen.
ARCHETYPE_NAME_SPECTRUM = [
    "Passive & Behind",
    "Under Pressure",
    "Sustain Skirmisher",
    "Balanced Contributor",
    "Snowballing Carry",
    "Dominant Carry",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export interactive dashboard JSON payload.")
    parser.add_argument("--features-path", default="data/processed/participant_features.parquet")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--matches-path", default="data/raw/matches.ndjson")
    parser.add_argument("--match-ids-path", default="data/raw/match_ids.ndjson")
    parser.add_argument("--riot-id", default="river3#na1")
    parser.add_argument("--population-sample", type=int, default=500, help="Scatter background points per role.")
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


def name_for_rank(rank: int, k: int) -> str:
    if k <= 1:
        return ARCHETYPE_NAME_SPECTRUM[len(ARCHETYPE_NAME_SPECTRUM) // 2]
    idx = round(rank / (k - 1) * (len(ARCHETYPE_NAME_SPECTRUM) - 1))
    return ARCHETYPE_NAME_SPECTRUM[idx]


def fillna_with_median(matrix: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    return matrix.fillna(reference.median(numeric_only=True))


def build_role_payload(
    role: str,
    df: pd.DataFrame,
    my_role_df: pd.DataFrame,
    models_dir: str,
    population_sample: int,
) -> dict:
    role_df = df[df["teamPosition"] == role].reset_index(drop=True)
    artifact = joblib.load(os.path.join(models_dir, f"{role.lower()}_cluster.joblib"))

    matrix = fillna_with_median(role_df[FEATURE_COLUMNS], role_df[FEATURE_COLUMNS])
    scaled = artifact["scaler"].transform(matrix)
    reduced_full = artifact["pca"].transform(scaled)  # (n, pca_components)

    rng = random.Random(RANDOM_STATE)
    sample_idx = list(range(len(role_df)))
    rng.shuffle(sample_idx)
    sample_idx = sample_idx[: min(population_sample, len(sample_idx))]

    scatter = {
        "x": [round(float(v), 4) for v in reduced_full[sample_idx, 0]],
        "y": [round(float(v), 4) for v in reduced_full[sample_idx, 1]],
    }

    my_matrix = fillna_with_median(my_role_df[FEATURE_COLUMNS], role_df[FEATURE_COLUMNS])
    my_scaled = artifact["scaler"].transform(my_matrix)
    my_reduced_full = artifact["pca"].transform(my_scaled) if len(my_role_df) else np.empty((0, reduced_full.shape[1]))

    clusters_by_k: dict[str, dict] = {}
    my_cluster_by_k: dict[str, list[int]] = {}
    for k in K_OPTIONS:
        if k >= len(role_df):
            continue
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(reduced_full)
        labels_full = kmeans.labels_

        win_by_cluster = pd.Series(role_df["win"].to_numpy()).groupby(labels_full).mean()
        size_by_cluster = pd.Series(labels_full).value_counts()
        ranked = win_by_cluster.sort_values().index.tolist()
        name_by_cluster = {cluster_id: name_for_rank(rank, k) for rank, cluster_id in enumerate(ranked)}

        clusters_by_k[str(k)] = {
            "sample_labels": [int(labels_full[i]) for i in sample_idx],
            "summary": [
                {
                    "cluster": int(cluster_id),
                    "n": int(size_by_cluster[cluster_id]),
                    "win_rate": round(float(win_by_cluster[cluster_id]), 3),
                    "name": name_by_cluster[cluster_id],
                }
                for cluster_id in sorted(win_by_cluster.index)
            ],
        }

        if len(my_reduced_full):
            my_labels = kmeans.predict(my_reduced_full)
        else:
            my_labels = np.array([], dtype=int)
        my_cluster_by_k[str(k)] = [int(v) for v in my_labels]

    my_scatter_xy = [[round(float(x), 4), round(float(y), 4)] for x, y in my_reduced_full[:, :2].tolist()] if len(my_reduced_full) else []

    return {
        "population_scatter": scatter,
        "clusters_by_k": clusters_by_k,
        "my_cluster_by_k": my_cluster_by_k,
        "my_scatter_xy": my_scatter_xy,
    }


def build_reference_curves(df: pd.DataFrame, tiers: list[str]) -> dict:
    reference: dict[str, dict] = {}
    for role in VALID_ROLES:
        reference[role] = {}
        for tier in tiers:
            tier_role_df = df[(df["tier"] == tier) & (df["teamPosition"] == role)]
            if len(tier_role_df) < MIN_REFERENCE_SAMPLE:
                continue
            feature_curves = {}
            quantiles = np.linspace(0, 100, BREAKPOINT_COUNT)
            for column, _, _ in HEADLINE_FEATURES:
                series = tier_role_df[column].dropna()
                if series.empty:
                    continue
                feature_curves[column] = {
                    "ref_avg": round(float(series.mean()), 3),
                    "n": int(len(series)),
                    "breakpoints": [round(float(v), 4) for v in np.percentile(series, quantiles)],
                }
            if feature_curves:
                reference[role][tier] = feature_curves
    return reference


def run(args: argparse.Namespace) -> None:
    df = pd.read_parquet(args.features_path)

    puuid = resolve_puuid_from_match_ids(args.match_ids_path, args.riot_id)
    my_df = df[df["puuid"] == puuid].copy()
    if my_df.empty:
        raise RuntimeError(f"No feature rows found for puuid {puuid}. Run build_features.py after fetching matches.")

    creation_by_id = load_game_creation_by_match_id(args.matches_path, set(my_df["matchId"]))
    my_df["gameCreation"] = my_df["matchId"].map(creation_by_id)
    my_df = my_df.sort_values("gameCreation", ascending=False).reset_index(drop=True)

    tiers = sorted(t for t in df["tier"].unique() if t != "UNKNOWN")
    print(f"Reference tiers with data: {tiers}")

    print("Building reference percentile curves...")
    reference = build_reference_curves(df, tiers)

    roles_payload = {}
    games_by_role_index: dict[str, list[int]] = {role: [] for role in VALID_ROLES}
    for role in VALID_ROLES:
        my_role_df = my_df[my_df["teamPosition"] == role].reset_index(drop=True)
        games_by_role_index[role] = my_df.index[my_df["teamPosition"] == role].tolist()
        print(f"Building cluster/scatter payload for {role} ({len(my_role_df)} of your games)...")
        roles_payload[role] = build_role_payload(role, df, my_role_df, args.models_dir, args.population_sample)

    # Attach each of your games' scatter coords + cluster-by-k, matched back to my_df order.
    scatter_xy_by_row: dict[int, list[float]] = {}
    cluster_by_k_by_row: dict[int, dict[str, int]] = {}
    for role in VALID_ROLES:
        role_indices = games_by_role_index[role]
        role_payload = roles_payload[role]
        for local_i, global_i in enumerate(role_indices):
            if local_i < len(role_payload["my_scatter_xy"]):
                scatter_xy_by_row[global_i] = role_payload["my_scatter_xy"][local_i]
            cluster_by_k_by_row[global_i] = {
                k: labels[local_i] for k, labels in role_payload["my_cluster_by_k"].items() if local_i < len(labels)
            }

    games = []
    for i, row in my_df.iterrows():
        feature_values = {}
        for column, _, _ in HEADLINE_FEATURES:
            value = row[column]
            feature_values[column] = round(float(value), 4) if pd.notna(value) else None

        games.append(
            {
                "matchId": row["matchId"],
                "champion": row["championName"],
                "role": row["teamPosition"],
                "win": bool(row["win"]),
                "gameCreation": int(row["gameCreation"]) if pd.notna(row["gameCreation"]) else None,
                "features": feature_values,
                "scatterXY": scatter_xy_by_row.get(i),
                "clusterByK": cluster_by_k_by_row.get(i, {}),
            }
        )

    record = {
        "wins": int(my_df["win"].sum()),
        "losses": int((~my_df["win"]).sum()),
        "win_rate": round(float(my_df["win"].mean()), 3),
        "total_games": int(len(my_df)),
    }
    role_counts = my_df["teamPosition"].value_counts().to_dict()
    champion_counts = my_df["championName"].value_counts().to_dict()

    payload = {
        "riot_id": args.riot_id,
        "k_options": K_OPTIONS,
        "tiers": tiers,
        "headline_features": [{"feature": c, "label": l, "invert": inv} for c, l, inv in HEADLINE_FEATURES],
        "record": record,
        "role_counts": role_counts,
        "champion_counts": champion_counts,
        "reference": reference,
        "roles": {
            role: {
                "population_scatter": roles_payload[role]["population_scatter"],
                "clusters_by_k": roles_payload[role]["clusters_by_k"],
            }
            for role in VALID_ROLES
        },
        "games": games,
    }

    parent = os.path.dirname(args.output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as outfile:
        json.dump(payload, outfile)
    size_kb = os.path.getsize(args.output_path) / 1024
    print(f"Dashboard data written to: {args.output_path} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    run(parse_args())
