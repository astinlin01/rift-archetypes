"""
Fit per-role KMeans playstyle clusters on the participant feature table.

Run from repo root:
    python src/build_clusters.py
"""

from __future__ import annotations

import argparse
import os

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from feature_config import ALL_FEATURE_FIELDS, VALID_ROLES

FEATURE_COLUMNS = [field.name for field in ALL_FEATURE_FIELDS]
RANDOM_STATE = 42
# silhouette_score is O(n^2) (full pairwise distances) -- exact is intractable once a
# role's population reaches tens of thousands of rows, so cap it to a random sample.
SILHOUETTE_SAMPLE_SIZE = 3000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit per-role clustering models.")
    parser.add_argument("--features-path", default="data/processed/participant_features.parquet")
    parser.add_argument("--output-path", default="data/processed/participant_clusters.parquet")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--centroid-report-path", default="data/processed/cluster_centroid_report.md")
    parser.add_argument("--min-k", type=int, default=3)
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument(
        "--k-override",
        action="append",
        default=[],
        help="Force a specific k for a role, e.g. --k-override JUNGLE=5. Repeatable.",
    )
    parser.add_argument("--pca-components", type=int, default=8)
    return parser.parse_args()


def parse_k_overrides(raw_overrides: list[str]) -> dict[str, int]:
    overrides: dict[str, int] = {}
    for item in raw_overrides:
        if "=" not in item:
            raise ValueError(f"--k-override must look like ROLE=K, got: {item}")
        role, k_str = item.split("=", 1)
        overrides[role.strip().upper()] = int(k_str.strip())
    return overrides


def prepare_role_matrix(role_df: pd.DataFrame) -> pd.DataFrame:
    matrix = role_df[FEATURE_COLUMNS].copy()
    return matrix.fillna(matrix.median(numeric_only=True))


def pick_best_k(scaled_matrix, min_k: int, max_k: int) -> tuple[int, dict[int, float]]:
    scores: dict[int, float] = {}
    n_samples = scaled_matrix.shape[0]
    upper_k = min(max_k, n_samples - 1)

    for k in range(min_k, upper_k + 1):
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(scaled_matrix)
        score = silhouette_score(
            scaled_matrix, labels,
            sample_size=min(SILHOUETTE_SAMPLE_SIZE, n_samples),
            random_state=RANDOM_STATE,
        )
        scores[k] = score

    best_k = max(scores, key=scores.get)
    return best_k, scores


def fit_role_cluster(
    role: str,
    role_df: pd.DataFrame,
    min_k: int,
    max_k: int,
    pca_components: int,
    k_override: int | None,
) -> tuple[pd.DataFrame, dict]:
    matrix = prepare_role_matrix(role_df)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)

    n_components = min(pca_components, scaled.shape[1], scaled.shape[0])
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    reduced = pca.fit_transform(scaled)

    if k_override is not None:
        best_k = k_override
        model = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(reduced)
        scores = {
            best_k: silhouette_score(
                reduced, labels,
                sample_size=min(SILHOUETTE_SAMPLE_SIZE, reduced.shape[0]),
                random_state=RANDOM_STATE,
            )
        }
    else:
        best_k, scores = pick_best_k(reduced, min_k, max_k)
        model = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(reduced)

    print(
        f"[{role}] n={len(role_df)}, chosen k={best_k}, "
        f"silhouette={scores[best_k]:.3f}, explained_var={pca.explained_variance_ratio_.sum():.3f}"
    )

    result_df = role_df.copy()
    result_df["cluster"] = labels

    artifact = {
        "role": role,
        "scaler": scaler,
        "pca": pca,
        "kmeans": model,
        "feature_columns": FEATURE_COLUMNS,
        "silhouette_by_k": scores,
        "chosen_k": best_k,
    }
    return result_df, artifact


def render_centroid_report(role_results: dict[str, pd.DataFrame]) -> str:
    lines = ["# Cluster Centroid Report", ""]
    for role, df in role_results.items():
        lines.append(f"## {role}")
        lines.append("")
        lines.append(f"- rows: {len(df)}, clusters: {df['cluster'].nunique()}")
        lines.append("")

        centroids = df.groupby("cluster")[FEATURE_COLUMNS].mean().round(2)
        cluster_sizes = df.groupby("cluster").size()
        win_rate = df.groupby("cluster")["win"].mean().round(3)

        lines.append("| cluster | n | win_rate | " + " | ".join(FEATURE_COLUMNS) + " |")
        lines.append("|---|---:|---:|" + "---:|" * len(FEATURE_COLUMNS))
        for cluster_id in sorted(centroids.index):
            row_values = " | ".join(str(v) for v in centroids.loc[cluster_id])
            lines.append(
                f"| {cluster_id} | {cluster_sizes[cluster_id]} | {win_rate[cluster_id]} | {row_values} |"
            )
        lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    if not os.path.exists(args.features_path):
        raise FileNotFoundError(
            f"Feature table not found at {args.features_path}. Run build_features.py first."
        )

    df = pd.read_parquet(args.features_path)
    k_overrides = parse_k_overrides(args.k_override)

    os.makedirs(args.models_dir, exist_ok=True)

    role_results: dict[str, pd.DataFrame] = {}
    for role in VALID_ROLES:
        role_df = df[df["teamPosition"] == role].reset_index(drop=True)
        if role_df.empty:
            print(f"[{role}] no rows, skipping.")
            continue

        result_df, artifact = fit_role_cluster(
            role=role,
            role_df=role_df,
            min_k=args.min_k,
            max_k=args.max_k,
            pca_components=args.pca_components,
            k_override=k_overrides.get(role),
        )
        role_results[role] = result_df

        model_path = os.path.join(args.models_dir, f"{role.lower()}_cluster.joblib")
        joblib.dump(artifact, model_path)

    combined = pd.concat(role_results.values(), ignore_index=True)
    parent = os.path.dirname(args.output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    combined.to_parquet(args.output_path, index=False)
    print(f"\nCluster assignments written to: {args.output_path}")

    report = render_centroid_report(role_results)
    with open(args.centroid_report_path, "w", encoding="utf-8") as outfile:
        outfile.write(report)
    print(f"Centroid report written to: {args.centroid_report_path}")


if __name__ == "__main__":
    run(parse_args())
