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
    python src/export_dashboard_data.py --riot-id "river3#na1" --riot-id "river#ell"
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.cluster import KMeans

from feature_config import ALL_FEATURE_FIELDS, VALID_ROLES, add_derived_features, current_season
from pipeline_io import iter_ndjson
from playstyle_space import (
    add_game_state_columns,
    apply_outcome_adjuster,
    name_clusters,
    to_playstyle_space,
)

FEATURE_COLUMNS = [f.name for f in ALL_FEATURE_FIELDS]
RANDOM_STATE = 42
K_OPTIONS = [3, 4, 5, 6]
MIN_REFERENCE_SAMPLE = 20  # skip a (role, tier) reference if fewer *players* than this
# Reference curves are built from per-player averages, not individual games: the
# dashboard compares *your average* over a window, and an average over many games has
# far less spread than single games -- ranking it against single games would squash a
# genuinely top-10% average down to ~70th percentile.
MIN_PLAYER_GAMES_FOR_REFERENCE = 10
MIN_PRO_GAMES_FOR_REFERENCE = 5
# A pro league only has ~10-16 players per role, so the solo-queue player minimum
# (MIN_REFERENCE_SAMPLE) would rule every league out.
MIN_PRO_REFERENCE_SAMPLE = 8
BREAKPOINT_COUNT = 51  # percentile grid: 0, 2, 4, ..., 100
DEFAULT_RIOT_IDS = ["river3#na1", "river#ell", "riverell#888", "riverell#ttv", "xlsteemo#na1"]
PCA_AXIS_TOP_N = 3
ALL_ROLES_KEY = "ALL"

# Highest-to-lowest, so the "Compare vs." dropdown reads Challenger > Grandmaster >
# Master > Diamond instead of alphabetically.
TIER_RANK = {"CHALLENGER": 0, "GRANDMASTER": 1, "MASTER": 2, "DIAMOND": 3}
REGION_RANK = {"NA1": 0, "KR": 1}
REGION_DISPLAY_NAMES = {"NA1": "NA"}

DDRAGON_VERSIONS_URL = "https://ddragon.leagueoflegends.com/api/versions.json"
DDRAGON_ICON_URL = "https://ddragon.leagueoflegends.com/cdn/{version}/img/champion/{champion}.png"
CHAMPION_ICON_CACHE_DIR = "data/raw/champion_icon_cache"

PRO_LEAGUE_RANK = {"LCK": 0, "LPL": 1}
MIN_PRO_PLAYER_GAMES = 5  # skip a pro player from the "which pro do you play like" list below this

# Subset of HEADLINE_FEATURES that Oracle's Elixir's schema has a clean equivalent for
# (see build_pro_features.py's PRO_FEATURE_MAP-equivalent construction) -- no CC-time or
# lane-opponent-diff data in pro match data, so those two are dropped here.
PRO_HEADLINE_FEATURES = [
    ("damage_per_minute", "Damage / min", False),
    ("gold_earned_per_min", "Gold / min", False),
    ("vision_score_per_minute", "Vision score / min", False),
    ("kda", "KDA", False),
    ("kill_participation", "Kill participation", False),
    ("team_damage_pct", "Team damage share", False),
    ("cs_total_per_min", "CS / min", False),
    ("damage_taken_per_min", "Damage taken / min", True),
    ("control_wards_per_min", "Control wards / min", False),
    ("deaths_per_min", "Deaths / min", True),
]

# Curated subset + display labels for the headline percentile comparison.
# invert=True means a lower raw value is typically the better outcome for this role.
HEADLINE_FEATURES = [
    ("damage_per_minute", "Damage / min", False),
    ("gold_earned_per_min", "Gold / min", False),
    ("vision_score_per_minute", "Vision score / min", False),
    ("kda", "KDA", False),
    ("kill_participation", "Kill participation", False),
    ("team_damage_pct", "Team damage share", False),
    ("cs_total_per_min", "CS / min", False),
    ("cc_time_per_min", "CC time on enemies / min", False),
    ("damage_taken_per_min", "Damage taken / min", True),
    ("control_wards_per_min", "Control wards / min", False),
    ("deaths_per_min", "Deaths / min", True),
    ("max_cs_advantage_lane_opp", "Max CS lead on opponent", False),
]

# Archetype names come from playstyle_space.name_clusters (dominant playstyle traits),
# not from win-rate rank -- clusters are fit on outcome-adjusted features.
TOP_CHAMPIONS_PER_CLUSTER = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export interactive dashboard JSON payload.")
    parser.add_argument("--features-path", default="data/processed/participant_features.parquet")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--matches-path", default="data/raw/matches.ndjson")
    parser.add_argument("--match-ids-path", default="data/raw/match_ids.ndjson")
    parser.add_argument(
        "--riot-id",
        action="append",
        default=None,
        help="Riot ID to include, e.g. 'river3#na1'. Repeatable. Defaults to both configured accounts.",
    )
    parser.add_argument("--population-sample", type=int, default=500, help="Scatter background points per role.")
    parser.add_argument("--pro-features-path", default="data/processed/pro_player_features.parquet")
    parser.add_argument("--output-path", default="data/processed/dashboard_data.json")
    return parser.parse_args()


def resolve_puuid_from_match_ids(match_ids_path: str, riot_id: str) -> str:
    # Riot IDs are case-insensitive ("river3#NA1" and "river3#na1" are one account).
    wanted = riot_id.strip().lower()
    for row in iter_ndjson(match_ids_path):
        if str(row.get("source_riot_id", "")).strip().lower() == wanted:
            return row["source_puuid"]
    raise RuntimeError(f"No match_ids.ndjson rows found for riot id {riot_id}")


def load_game_creation_by_match_id(matches_path: str, match_ids: set[str]) -> dict[str, int]:
    creation_by_id: dict[str, int] = {}
    for match in iter_ndjson(matches_path):
        match_id = match.get("metadata", {}).get("matchId")
        if match_id in match_ids:
            creation_by_id[match_id] = match.get("info", {}).get("gameCreation")
    return creation_by_id


HEADLINE_LABEL_BY_FEATURE = {column: label for column, label, _ in HEADLINE_FEATURES}


def prettify_feature_name(name: str) -> str:
    if name in HEADLINE_LABEL_BY_FEATURE:
        return HEADLINE_LABEL_BY_FEATURE[name]
    label = name
    for suffix, replacement in (("_per_minute", "/min"), ("_per_min", "/min"), ("_pct", " %")):
        if label.endswith(suffix):
            label = label[: -len(suffix)] + replacement
            break
    label = label.replace("_", " ")
    return label[:1].upper() + label[1:]


def describe_pca_axis(component: np.ndarray, feature_columns: list[str]) -> str:
    # Walk loadings in descending magnitude order, but skip a feature whose label
    # we've already used -- a few feature pairs (e.g. our own gold_earned_per_min vs.
    # Riot's already-normalized gold_per_minute) are highly correlated and would
    # otherwise show up as a confusing near-duplicate in the same caption.
    order = np.argsort(-np.abs(component))
    seen_labels: set[str] = set()
    parts = []
    for idx in order:
        label = prettify_feature_name(feature_columns[idx])
        normalized = re.sub(r"[^a-z0-9]", "", label.lower())
        if normalized in seen_labels:
            continue
        seen_labels.add(normalized)
        sign = "higher" if component[idx] >= 0 else "lower"
        parts.append(f"{sign} {label}")
        if len(parts) >= PCA_AXIS_TOP_N:
            break
    return ", ".join(parts)


def build_role_payload(
    role: str,
    df: pd.DataFrame,
    my_role_df: pd.DataFrame,
    models_dir: str,
    population_sample: int,
) -> dict:
    role_df = df[df["teamPosition"] == role].reset_index(drop=True)
    artifact = joblib.load(os.path.join(models_dir, f"{role.lower()}_cluster.joblib"))

    # Outcome-adjusted + standardized features (for naming) and their PCA projection.
    adjusted_scaled = artifact["scaler"].transform(apply_outcome_adjuster(role_df, artifact["outcome_adjuster"]))
    reduced_full = artifact["pca"].transform(adjusted_scaled)  # (n, pca_components)

    rng = random.Random(RANDOM_STATE)
    sample_idx = list(range(len(role_df)))
    rng.shuffle(sample_idx)
    sample_idx = sample_idx[: min(population_sample, len(sample_idx))]

    scatter = {
        "x": [round(float(v), 4) for v in reduced_full[sample_idx, 0]],
        "y": [round(float(v), 4) for v in reduced_full[sample_idx, 1]],
    }

    my_reduced_full = (
        to_playstyle_space(my_role_df, artifact) if len(my_role_df) else np.empty((0, reduced_full.shape[1]))
    )

    clusters_by_k: dict[str, dict] = {}
    my_cluster_by_k: dict[str, list[int]] = {}
    for k in K_OPTIONS:
        if k >= len(role_df):
            continue
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(reduced_full)
        labels_full = kmeans.labels_

        win_by_cluster = pd.Series(role_df["win"].to_numpy()).groupby(labels_full).mean()
        size_by_cluster = pd.Series(labels_full).value_counts()
        naming = name_clusters(adjusted_scaled, labels_full, artifact["feature_columns"], role)
        top_champions = (
            pd.Series(role_df["championName"].to_numpy())
            .groupby(labels_full)
            .agg(lambda s: s.value_counts().index[:TOP_CHAMPIONS_PER_CLUSTER].tolist())
        )

        clusters_by_k[str(k)] = {
            "sample_labels": [int(labels_full[i]) for i in sample_idx],
            "summary": [
                {
                    "cluster": int(cluster_id),
                    "n": int(size_by_cluster[cluster_id]),
                    "win_rate": round(float(win_by_cluster[cluster_id]), 3),
                    "name": naming[int(cluster_id)]["name"],
                    "traits": naming[int(cluster_id)]["traits"],
                    "low_traits": naming[int(cluster_id)]["low_traits"],
                    "top_champions": top_champions[cluster_id],
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

    # PCA components are expressed in *scaled feature* space -- component_[i] is the
    # loading of each standardized feature onto axis i, so the top absolute loadings
    # tell us what that axis mostly separates on.
    pca_axis_labels = {
        "x": describe_pca_axis(artifact["pca"].components_[0], artifact["feature_columns"]),
        "y": describe_pca_axis(artifact["pca"].components_[1], artifact["feature_columns"]),
    }

    return {
        "population_scatter": scatter,
        "clusters_by_k": clusters_by_k,
        "my_cluster_by_k": my_cluster_by_k,
        "my_scatter_xy": my_scatter_xy,
        "pca_axis_labels": pca_axis_labels,
    }


def player_averages(rows: pd.DataFrame, player_column: str, min_games: int, feature_columns: list[str]) -> pd.DataFrame:
    counts = rows.groupby(player_column).size()
    regulars = counts[counts >= min_games].index
    return rows[rows[player_column].isin(regulars)].groupby(player_column)[feature_columns].mean()


def build_curves_for_role_df(
    role_df: pd.DataFrame,
    headline_features=HEADLINE_FEATURES,
    player_column: str = "puuid",
    min_games: int = MIN_PLAYER_GAMES_FOR_REFERENCE,
    min_players: int = MIN_REFERENCE_SAMPLE,
) -> dict:
    feature_columns = [c for c, _, _ in headline_features]
    averages = player_averages(role_df, player_column, min_games, feature_columns)
    if len(averages) < min_players:
        return {}
    feature_curves = {}
    quantiles = np.linspace(0, 100, BREAKPOINT_COUNT)
    for column in feature_columns:
        series = averages[column].dropna()
        if series.empty:
            continue
        feature_curves[column] = {
            "ref_avg": round(float(series.mean()), 3),
            # Spread of player averages -- the dashboard uses it to place you relative to
            # your solo-queue peers when matching against pros (see renderProMatches).
            "std": round(float(series.std(ddof=0)) or 1.0, 4),
            "n": int(len(series)),  # number of players, not games
            "breakpoints": [round(float(v), 4) for v in np.percentile(series, quantiles)],
        }
    return feature_curves


def load_pro_features(pro_features_path: str) -> pd.DataFrame | None:
    if not os.path.exists(pro_features_path):
        print(f"No pro feature table found at {pro_features_path}; skipping pro comparison. Run build_pro_features.py to add it.")
        return None
    return pd.read_parquet(pro_features_path)


def build_pro_reference(pro_df: pd.DataFrame) -> tuple[dict, list[str]]:
    """Percentile reference curves per (role, league), built from Oracle's Elixir data.

    Mirrors build_reference_curves' shape but keyed by league instead of region/tier,
    and uses PRO_HEADLINE_FEATURES (a subset of our own features that has a clean
    equivalent in Oracle's Elixir's schema -- see build_pro_features.py).
    """
    leagues = sorted(pro_df["league"].unique(), key=lambda l: PRO_LEAGUE_RANK.get(l, len(PRO_LEAGUE_RANK)))

    reference: dict[str, dict] = {}
    for role in VALID_ROLES:
        reference[role] = {}
        role_df = pro_df[pro_df["role"] == role]
        for league in leagues:
            league_role_df = role_df[role_df["league"] == league]
            feature_curves = build_curves_for_role_df(
                league_role_df,
                PRO_HEADLINE_FEATURES,
                player_column="playername",
                min_games=MIN_PRO_GAMES_FOR_REFERENCE,
                min_players=MIN_PRO_REFERENCE_SAMPLE,
            )
            if feature_curves:
                reference[role][league] = feature_curves
    return reference, leagues


def build_pro_player_profiles(pro_df: pd.DataFrame) -> dict:
    """Per-pro-player season-average stat profiles per role, standardized against the
    pro population's own scale, for a client-side "which pro do you play like" nearest-
    neighbor match. A player's team/league label uses whichever they played the most
    games under this season (handles a mid-split team change without splitting them
    into two entries).
    """
    feature_cols = [c for c, _, _ in PRO_HEADLINE_FEATURES]
    profiles: dict[str, dict] = {}

    for role in VALID_ROLES:
        role_df = pro_df[pro_df["role"] == role]
        if role_df.empty:
            continue

        # Standardize against the spread of *player averages* (not single pro games),
        # matching how the dashboard standardizes you against your solo-queue peers'
        # averages -- so both z-scores mean "how far from typical for your population".
        qualifying = player_averages(role_df, "playername", MIN_PRO_PLAYER_GAMES, feature_cols)
        if qualifying.empty:
            continue
        mean = qualifying.mean()
        std = qualifying.std(ddof=0).replace(0, 1.0)

        players = []
        for player_name, player_df in role_df.groupby("playername"):
            if len(player_df) < MIN_PRO_PLAYER_GAMES:
                continue
            representative_team = player_df["teamname"].value_counts().idxmax()
            representative_league = player_df["league"].value_counts().idxmax()
            raw_avg = player_df[feature_cols].mean()
            z = (raw_avg - mean) / std
            top_champs = player_df["champion"].value_counts().head(3).index.tolist()

            players.append(
                {
                    "name": player_name,
                    "team": representative_team,
                    "league": representative_league,
                    "games": int(len(player_df)),
                    "win_rate": round(float(player_df["win"].mean()), 3),
                    "top_champs": top_champs,
                    "z": {col: round(float(z[col]), 4) for col in feature_cols},
                }
            )

        if players:
            profiles[role] = {
                "mean": {col: round(float(mean[col]), 4) for col in feature_cols},
                "std": {col: round(float(std[col]), 4) for col in feature_cols},
                "players": players,
            }

    return profiles


def build_reference_curves(df: pd.DataFrame, regions: list[str], tiers: list[str]) -> dict:
    reference: dict[str, dict] = {}
    for role in list(VALID_ROLES) + [ALL_ROLES_KEY]:
        reference[role] = {}
        role_df = df if role == ALL_ROLES_KEY else df[df["teamPosition"] == role]
        for region in regions:
            region_role_df = role_df[role_df["region"] == region]
            region_curves = {}
            for tier in tiers:
                tier_role_df = region_role_df[region_role_df["tier"] == tier]
                feature_curves = build_curves_for_role_df(tier_role_df)
                if feature_curves:
                    region_curves[tier] = feature_curves
            if region_curves:
                reference[role][region] = region_curves
    return reference


def fetch_champion_icons(champion_names: set[str]) -> dict[str, str]:
    """Base64 data-URI icons for the given champion names, fetched from Data Dragon and
    cached on disk. Embedding as data URIs is required because the dashboard, once
    published as an Artifact, runs under a CSP that blocks image requests to external
    hosts -- only same-document data: URIs work.
    """
    icons: dict[str, str] = {}
    if not champion_names:
        return icons

    os.makedirs(CHAMPION_ICON_CACHE_DIR, exist_ok=True)

    version = None
    try:
        response = requests.get(DDRAGON_VERSIONS_URL, timeout=10)
        response.raise_for_status()
        version = response.json()[0]
    except Exception as exc:
        print(f"Warning: couldn't fetch Data Dragon version ({exc}); using any cached icons only.")

    for champion in sorted(champion_names):
        cache_path = os.path.join(CHAMPION_ICON_CACHE_DIR, f"{champion}.png")
        if not os.path.exists(cache_path) and version:
            url = DDRAGON_ICON_URL.format(version=version, champion=champion)
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    with open(cache_path, "wb") as outfile:
                        outfile.write(response.content)
                else:
                    print(f"  no Data Dragon icon for '{champion}' (status {response.status_code})")
            except Exception as exc:
                print(f"  failed to fetch icon for '{champion}': {exc}")

        if os.path.exists(cache_path):
            with open(cache_path, "rb") as infile:
                encoded = base64.b64encode(infile.read()).decode("ascii")
            icons[champion] = f"data:image/png;base64,{encoded}"

    return icons


def run(args: argparse.Namespace) -> None:
    df = add_game_state_columns(add_derived_features(pd.read_parquet(args.features_path)))
    riot_ids = args.riot_id if args.riot_id else DEFAULT_RIOT_IDS

    # Everything -- reference curves, clustering population, and your own games -- is
    # restricted to one season so you're never compared across a stat-baseline shift.
    season = current_season(df)
    if season is not None:
        before = len(df)
        df = df[df["season"] == season].reset_index(drop=True)
        print(f"Season-matching to season {season}: kept {len(df)} of {before} rows.")
    else:
        print("Warning: feature table has no season column -- rebuild with build_features.py to season-match.")

    account_frames = []
    for riot_id in riot_ids:
        try:
            puuid = resolve_puuid_from_match_ids(args.match_ids_path, riot_id)
        except RuntimeError as error:
            print(f"Warning: {error}. Skipping (run fetch_recent_self.py for this account first).")
            continue
        account_df = df[df["puuid"] == puuid].copy()
        if account_df.empty:
            print(f"Warning: no feature rows found for {riot_id} (puuid {puuid}). Skipping.")
            continue
        account_df["account"] = riot_id
        account_frames.append(account_df)

    if not account_frames:
        raise RuntimeError("No feature rows found for any requested Riot ID. Run build_features.py after fetching matches.")

    my_df = pd.concat(account_frames, ignore_index=True)

    if "gameCreation" not in my_df.columns:
        creation_by_id = load_game_creation_by_match_id(args.matches_path, set(my_df["matchId"]))
        my_df["gameCreation"] = my_df["matchId"].map(creation_by_id)
    my_df = my_df.sort_values("gameCreation", ascending=False).reset_index(drop=True)

    tiers = sorted(
        (t for t in df["tier"].unique() if t != "UNKNOWN"),
        key=lambda t: TIER_RANK.get(t, len(TIER_RANK)),
    )
    regions = sorted(
        (r for r in df["region"].unique() if r != "UNKNOWN"),
        key=lambda r: REGION_RANK.get(r, len(REGION_RANK)),
    )
    print(f"Reference tiers with data: {tiers}")
    print(f"Reference regions with data: {regions}")

    print("Building reference percentile curves...")
    reference = build_reference_curves(df, regions, tiers)

    pro_df = load_pro_features(args.pro_features_path)
    if pro_df is not None:
        print("Building pro reference percentile curves...")
        pro_reference, pro_leagues = build_pro_reference(pro_df)
        print("Building pro player profiles for nearest-match comparison...")
        pro_player_profiles = build_pro_player_profiles(pro_df)
    else:
        pro_reference, pro_leagues, pro_player_profiles = {}, [], {}

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
                "account": row["account"],
                "champion": row["championName"],
                "role": row["teamPosition"],
                "win": bool(row["win"]),
                "gameCreation": int(row["gameCreation"]) if pd.notna(row["gameCreation"]) else None,
                "features": feature_values,
                "scatterXY": scatter_xy_by_row.get(i),
                "clusterByK": cluster_by_k_by_row.get(i, {}),
            }
        )

    print("Fetching champion icons...")
    champion_icons = fetch_champion_icons(set(my_df["championName"].unique()))
    print(f"  {len(champion_icons)} icon(s) available.")

    # Per-account summaries, for notebook/debug introspection only -- the published
    # dashboard recomputes everything it shows from the filtered `games` list directly.
    accounts_summary = {}
    for riot_id in riot_ids:
        account_df = my_df[my_df["account"] == riot_id]
        if account_df.empty:
            continue
        accounts_summary[riot_id] = {
            "wins": int(account_df["win"].sum()),
            "losses": int((~account_df["win"]).sum()),
            "win_rate": round(float(account_df["win"].mean()), 3),
            "total_games": int(len(account_df)),
            "role_counts": account_df["teamPosition"].value_counts().to_dict(),
            "champion_counts": account_df["championName"].value_counts().to_dict(),
        }

    payload = {
        "season": season,
        "accounts": list(accounts_summary.keys()),
        "accounts_summary": accounts_summary,
        "k_options": K_OPTIONS,
        "tiers": tiers,
        "regions": regions,
        "region_display_names": REGION_DISPLAY_NAMES,
        "headline_features": [{"feature": c, "label": l, "invert": inv} for c, l, inv in HEADLINE_FEATURES],
        "pro_leagues": pro_leagues,
        "pro_headline_features": [{"feature": c, "label": l, "invert": inv} for c, l, inv in PRO_HEADLINE_FEATURES],
        "champion_icons": champion_icons,
        "reference": reference,
        "pro_reference": pro_reference,
        "pro_player_profiles": pro_player_profiles,
        "roles": {
            role: {
                "population_scatter": roles_payload[role]["population_scatter"],
                "clusters_by_k": roles_payload[role]["clusters_by_k"],
                "pca_axis_labels": roles_payload[role]["pca_axis_labels"],
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
