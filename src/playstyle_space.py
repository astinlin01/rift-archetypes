"""
Shared "playstyle space" transform: outcome-adjusted features -> scaler -> PCA.

Why outcome adjustment: almost every per-minute stat (damage, gold, KDA, kills,
objective damage, ...) scales with *how the game went*, not just *how the player
plays*. Clustering those raw stats sorts games into "got stomped / normal / stomping"
(cluster win rates of ~15% / ~60% / ~90%) rather than into playstyles.

Fix: per role, regress every clustering feature on game-context covariates (win flag,
team kill differential per minute, team gold differential per minute, and a spline of
game length) and cluster the residuals -- i.e. "how this game's stats differ from what
a game with this outcome and length typically looks like for this role". The fitted regression is stored in the role's
model artifact so every consumer (build_clusters, compare_self, export_dashboard_data)
projects rows identically.

The raw feature columns are untouched in the feature table -- percentile comparisons
against a reference tier still use raw values, which is what you want there (being
ahead *is* the point of e.g. gold/min percentiles).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import SplineTransformer

from feature_config import FEATURE_COLUMN_NAMES

GAME_STATE_COLUMNS = ("team_kill_diff_per_min", "team_gold_diff_per_min")
OUTCOME_COVARIATES = ("win",) + GAME_STATE_COLUMNS
# Game length enters the adjustment as a cubic spline (see fit_outcome_adjuster).
LENGTH_SPLINE_KNOTS = 6

# Exact or near-exact duplicates of another clustering feature (Riot's precomputed
# per-minute value vs. our own division of the same total). Kept in the feature table
# for the percentile charts, but excluded from clustering so they don't double-weight
# one dimension.
CLUSTERING_EXCLUDED_DUPLICATES = frozenset(
    {
        "gold_per_minute",  # == gold_earned_per_min
        "damage_to_champions_per_min",  # == damage_per_minute
        "vision_score_per_min_raw",  # == vision_score_per_minute
        "cs_total_per_min",  # == cs_per_min + neutral_cs_per_min
    }
)

CLUSTER_FEATURE_COLUMNS = [c for c in FEATURE_COLUMN_NAMES if c not in CLUSTERING_EXCLUDED_DUPLICATES]


def add_game_state_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Attach team kill/gold differential per minute to every row (no-op if present).

    Must be called on the full feature table, not a subset of one player's rows.

    Derived from the feature table itself: within a match, the 5 rows sharing a `win`
    value are one team. Rows whose opposing team is missing from the table (shouldn't
    happen after build_features' remake filter, but possible for partial data) get 0,
    i.e. "even game".
    """
    if all(column in df.columns for column in GAME_STATE_COLUMNS):
        return df

    team = (
        df.groupby(["matchId", "win"])[["kills_per_min", "gold_earned_per_min"]]
        .sum()
        .reset_index()
    )
    opponent = team.copy()
    opponent["win"] = ~opponent["win"].astype(bool)
    merged = team.merge(opponent, on=["matchId", "win"], suffixes=("_team", "_opp"))
    merged["team_kill_diff_per_min"] = merged["kills_per_min_team"] - merged["kills_per_min_opp"]
    merged["team_gold_diff_per_min"] = merged["gold_earned_per_min_team"] - merged["gold_earned_per_min_opp"]

    out = df.drop(columns=[c for c in GAME_STATE_COLUMNS if c in df.columns]).merge(
        merged[["matchId", "win", *GAME_STATE_COLUMNS]],
        on=["matchId", "win"],
        how="left",
    )
    for column in GAME_STATE_COLUMNS:
        out[column] = out[column].fillna(0.0)
    return out


def _game_minutes(df: pd.DataFrame) -> np.ndarray:
    return (df["gameLength"].astype(float).to_numpy() / 60.0).reshape(-1, 1)


def _covariate_matrix(
    win_and_state: np.ndarray, minutes: np.ndarray, length_spline: SplineTransformer
) -> np.ndarray:
    return np.hstack([win_and_state, length_spline.transform(minutes)])


def _row_covariates(df: pd.DataFrame, length_spline: SplineTransformer) -> np.ndarray:
    return _covariate_matrix(
        df[list(OUTCOME_COVARIATES)].astype(float).to_numpy(), _game_minutes(df), length_spline
    )


def fit_outcome_adjuster(role_df: pd.DataFrame) -> dict:
    """Fit the per-role feature-on-context regression. role_df must have game-state columns.

    Covariates: win, team kill/gold differential per minute, and a smooth (cubic spline)
    function of game length. Game length matters on its own: champion level, killing
    sprees, turret plates, max lane leads etc. are per-game totals/maxima that grow with
    length -- non-linearly (champion level follows the XP curve and caps at 18) -- and
    even per-minute rates like gold/min drift with it. Without this, the shortest games
    formed their own "low everything" cluster.
    """
    medians = role_df[CLUSTER_FEATURE_COLUMNS].median(numeric_only=True)
    features = role_df[CLUSTER_FEATURE_COLUMNS].fillna(medians)
    minutes = _game_minutes(role_df)
    # Knots at length quantiles so the curve has resolution where games actually are;
    # constant extrapolation keeps unusually short/long games from blowing up.
    length_spline = SplineTransformer(
        n_knots=LENGTH_SPLINE_KNOTS, degree=3, knots="quantile", extrapolation="constant"
    ).fit(minutes)
    regression = LinearRegression().fit(_row_covariates(role_df, length_spline), features)
    return {
        "regression": regression,
        "length_spline": length_spline,
        "median_game_minutes": float(np.median(minutes)),
        "feature_medians": medians,
        "feature_columns": list(CLUSTER_FEATURE_COLUMNS),
        "covariates": list(OUTCOME_COVARIATES) + ["game_minutes_spline"],
    }


def apply_outcome_adjuster(rows: pd.DataFrame, adjuster: dict) -> pd.DataFrame:
    """Residualize rows' features against game context, keeping each feature's scale.

    Residual + the prediction for a "typical game" (win=0.5, even teams, median game
    length for the role) keeps adjusted values in readable units, e.g. "champion level
    this game would have had in an even, median-length game".
    """
    columns = adjuster["feature_columns"]
    features = rows[columns].fillna(adjuster["feature_medians"])
    regression: LinearRegression = adjuster["regression"]
    length_spline: SplineTransformer = adjuster["length_spline"]
    predicted = regression.predict(_row_covariates(rows, length_spline))
    typical = _covariate_matrix(
        np.array([[0.5] + [0.0] * len(GAME_STATE_COLUMNS)]),
        np.array([[adjuster["median_game_minutes"]]]),
        length_spline,
    )
    adjusted = features.to_numpy() - predicted + regression.predict(typical)
    return pd.DataFrame(adjusted, columns=columns, index=rows.index)


def to_playstyle_space(rows: pd.DataFrame, artifact: dict) -> np.ndarray:
    """rows -> outcome-adjusted -> scaled -> PCA coords, using a role's fitted artifact."""
    # Game-state columns must be computed on the *full* feature table (they need all
    # 10 participants of a match), so a subset like "just my games" can't derive them.
    missing = [c for c in GAME_STATE_COLUMNS if c not in rows.columns]
    if missing:
        raise ValueError(
            f"Rows are missing game-state columns {missing}; call add_game_state_columns() "
            "on the full feature table before selecting a subset."
        )
    adjusted = apply_outcome_adjuster(rows, artifact["outcome_adjuster"])
    scaled = artifact["scaler"].transform(adjusted)
    return artifact["pca"].transform(scaled)


# ---------- Cluster naming ----------
#
# Clusters used to be named by win-rate rank ("Passive & Behind" ... "Dominant Carry").
# With outcome removed they're named by what they actually do: each trait is the mean
# standardized (outcome-adjusted) centroid value over a small group of features, and a
# cluster is named after its two strongest traits.

TRAITS: list[tuple[str, tuple[str, ...]]] = [
    ("Frontline", ("damage_taken_per_min", "damage_self_mitigated_per_min", "damage_taken_on_team_pct")),
    ("Crowd Control", ("cc_time_per_min",)),
    ("Damage Carry", ("damage_per_minute", "team_damage_pct")),
    ("Enchanter", ("heal_on_teammates_per_min", "shield_on_teammates_per_min")),
    ("Vision Control", ("vision_score_per_minute", "wards_placed_per_min", "wards_killed_per_min", "control_wards_per_min")),
    ("Farm-Focused", ("cs_per_min", "lane_minions_first_10", "gold_earned_per_min")),
    ("Lane Bully", ("max_cs_advantage_lane_opp", "max_level_lead_lane_opp", "turret_plates_taken")),
    ("Duelist", ("solo_kills", "kills_per_min")),
    ("Teamfighter", ("kill_participation", "assists_per_min")),
    ("Split-Pusher", ("damage_to_buildings_per_min", "turret_takedowns_per_min")),
    ("Objective Control", ("damage_to_objectives_per_min", "dragon_takedowns_per_min", "baron_takedowns_per_min")),
    ("Jungle Farmer", ("neutral_cs_per_min",)),
    ("Skillshot Mage", ("skillshots_hit_per_min", "ability_uses_per_min")),
    ("High-Risk", ("deaths_per_min",)),
]

# Traits that read as nonsense for a role (e.g. "Jungle Farmer" for a bot laner whose
# adjusted neutral CS is merely above the near-zero role average).
ROLE_EXCLUDED_TRAITS: dict[str, frozenset[str]] = {
    "TOP": frozenset({"Jungle Farmer"}),
    "JUNGLE": frozenset({"Farm-Focused", "Lane Bully", "Enchanter"}),
    "MIDDLE": frozenset({"Jungle Farmer"}),
    "BOTTOM": frozenset({"Jungle Farmer"}),
    "UTILITY": frozenset({"Jungle Farmer", "Farm-Focused", "Lane Bully"}),
}

# Below this, a cluster's strongest trait isn't really distinctive.
MIN_TRAIT_SCORE = 0.15
LOW_PROFILE_NAME = "Low-Profile"


def trait_scores(centroid_z: pd.Series, role: str | None = None) -> pd.Series:
    excluded = ROLE_EXCLUDED_TRAITS.get(role, frozenset()) if role else frozenset()
    scores = {}
    for trait, features in TRAITS:
        if trait in excluded:
            continue
        present = [f for f in features if f in centroid_z.index]
        if present:
            scores[trait] = float(centroid_z[present].mean())
    return pd.Series(scores).sort_values(ascending=False)


def name_clusters(
    adjusted_scaled: np.ndarray,
    labels: np.ndarray,
    feature_columns: list[str],
    role: str | None = None,
) -> dict[int, dict]:
    """Return {cluster_id: {"name", "traits", "low_traits"}} from standardized adjusted features.

    Names are made unique within a role: if two clusters' top-two traits collide, the
    later one swaps in its next-best trait.
    """
    frame = pd.DataFrame(adjusted_scaled, columns=feature_columns)
    centroids = frame.groupby(labels).mean()

    result: dict[int, dict] = {}
    used_names: set[str] = set()
    # Name the most distinctive clusters first so they get first pick of trait pairs.
    order = sorted(centroids.index, key=lambda c: -trait_scores(centroids.loc[c], role).iloc[0])
    for cluster_id in order:
        scores = trait_scores(centroids.loc[cluster_id], role)
        positive = [t for t, s in scores.items() if s >= MIN_TRAIT_SCORE]
        low = [t for t, s in scores.sort_values().items() if s <= -MIN_TRAIT_SCORE][:2]

        name = None
        if not positive:
            name = LOW_PROFILE_NAME
        elif len(positive) == 1:
            name = positive[0]
        else:
            for i in range(1, len(positive)):
                candidate = f"{positive[0]} · {positive[i]}"
                if candidate not in used_names:
                    name = candidate
                    break
            name = name or positive[0]
        base, suffix = name, 2
        while name in used_names:
            name = f"{base} ({suffix})"
            suffix += 1
        used_names.add(name)
        result[int(cluster_id)] = {"name": name, "traits": positive[:3], "low_traits": low}
    return result
