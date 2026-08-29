"""
Assign your own matches to the fitted per-role clusters and compare your
per-feature averages against a reference tier (Challenger by default).

Run from repo root:
    python src/compare_self.py --puuid <your puuid>
    python src/compare_self.py --riot-id "river3#na1"
"""

from __future__ import annotations

import argparse
import os

import joblib
import pandas as pd
from scipy.stats import percentileofscore

from feature_config import ALL_FEATURE_FIELDS, VALID_ROLES
from riot_client import RiotClient

FEATURE_COLUMNS = [field.name for field in ALL_FEATURE_FIELDS]

# Mirrors fetch_match_ids.py's always-include accounts.
DEFAULT_RIOT_IDS = [
    "river3#na1",
    "river#ell",
    "riverell#888",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare your matches against learned clusters.")
    parser.add_argument("--features-path", default="data/processed/participant_features.parquet")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--reference-tier", default="CHALLENGER")
    parser.add_argument("--platform", default="na1")
    parser.add_argument(
        "--puuid",
        action="append",
        default=[],
        help="Your puuid(s). Repeatable. If omitted, resolves --riot-id / defaults instead.",
    )
    parser.add_argument(
        "--riot-id",
        action="append",
        default=[],
        help="Your Riot ID(s) as gameName#tagLine. Repeatable.",
    )
    parser.add_argument("--output-csv", default="data/processed/self_comparison.csv")
    return parser.parse_args()


def resolve_puuids(args: argparse.Namespace) -> list[str]:
    if args.puuid:
        return args.puuid

    riot_ids = args.riot_id if args.riot_id else DEFAULT_RIOT_IDS
    client = RiotClient(platform=args.platform)
    puuids: list[str] = []
    for riot_id in riot_ids:
        game_name, tag_line = riot_id.split("#", 1)
        try:
            account = client.get_account_by_riot_id_any_region(
                game_name=game_name.strip(), tag_line=tag_line.strip()
            )
        except Exception as exc:
            print(f"Failed to resolve {riot_id}: {exc}")
            continue
        puuids.append(account["puuid"])
    return puuids


def load_role_artifact(models_dir: str, role: str) -> dict | None:
    path = os.path.join(models_dir, f"{role.lower()}_cluster.joblib")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def assign_clusters(rows: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    matrix = rows[FEATURE_COLUMNS].copy()
    matrix = matrix.fillna(matrix.median(numeric_only=True))
    scaled = artifact["scaler"].transform(matrix)
    reduced = artifact["pca"].transform(scaled)
    labels = artifact["kmeans"].predict(reduced)
    result = rows.copy()
    result["cluster"] = labels
    return result


def build_comparison_rows(
    my_role_df: pd.DataFrame,
    reference_role_df: pd.DataFrame,
    role: str,
) -> list[dict]:
    rows: list[dict] = []
    my_means = my_role_df[FEATURE_COLUMNS].mean()
    ref_means = reference_role_df[FEATURE_COLUMNS].mean()

    for column in FEATURE_COLUMNS:
        my_value = my_means[column]
        ref_series = reference_role_df[column].dropna()
        if ref_series.empty or pd.isna(my_value):
            percentile = None
        else:
            percentile = percentileofscore(ref_series, my_value, kind="mean")

        rows.append(
            {
                "role": role,
                "feature": column,
                "your_avg": round(float(my_value), 3) if pd.notna(my_value) else None,
                "reference_avg": round(float(ref_means[column]), 3) if pd.notna(ref_means[column]) else None,
                "your_percentile_in_reference": round(percentile, 1) if percentile is not None else None,
            }
        )
    return rows


def run(args: argparse.Namespace) -> None:
    if not os.path.exists(args.features_path):
        raise FileNotFoundError(
            f"Feature table not found at {args.features_path}. Run build_features.py first."
        )

    puuids = resolve_puuids(args)
    if not puuids:
        raise RuntimeError("No puuids resolved. Pass --puuid or --riot-id.")
    print(f"Comparing {len(puuids)} account(s).")

    df = pd.read_parquet(args.features_path)
    my_df = df[df["puuid"].isin(puuids)]
    reference_df = df[df["tier"] == args.reference_tier.upper()]

    if my_df.empty:
        print("No matches found for the given puuid(s) in the feature table.")
        return

    all_comparison_rows: list[dict] = []
    for role in VALID_ROLES:
        my_role_df = my_df[my_df["teamPosition"] == role]
        if my_role_df.empty:
            continue

        reference_role_df = reference_df[reference_df["teamPosition"] == role]
        if reference_role_df.empty:
            print(f"[{role}] No reference-tier rows available, skipping comparison.")
            continue

        artifact = load_role_artifact(args.models_dir, role)
        if artifact is None:
            print(f"[{role}] No fitted cluster model found, skipping cluster assignment.")
        else:
            assigned = assign_clusters(my_role_df, artifact)
            cluster_counts = assigned["cluster"].value_counts().to_dict()
            print(f"[{role}] games={len(my_role_df)}, your cluster distribution: {cluster_counts}")

        comparison_rows = build_comparison_rows(my_role_df, reference_role_df, role)
        all_comparison_rows.extend(comparison_rows)

        print(f"\n=== {role} vs {args.reference_tier} (n_you={len(my_role_df)}, n_ref={len(reference_role_df)}) ===")
        for row in comparison_rows:
            print(
                f"  {row['feature']:<32} you={row['your_avg']!s:<10} "
                f"ref={row['reference_avg']!s:<10} pctile={row['your_percentile_in_reference']}"
            )

    comparison_df = pd.DataFrame(all_comparison_rows)
    parent = os.path.dirname(args.output_csv)
    if parent:
        os.makedirs(parent, exist_ok=True)
    comparison_df.to_csv(args.output_csv, index=False)
    print(f"\nComparison table written to: {args.output_csv}")


if __name__ == "__main__":
    run(parse_args())
