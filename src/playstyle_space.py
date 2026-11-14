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

from champion_classes import champion_class
from feature_config import FEATURE_COLUMN_NAMES, add_team_context_features, lane_opponent_values

# Game state is split into what the player did and what their teammates did:
#     team lead = teammates' lead + own lane lead
# where "teammates' lead" compares your 4 teammates against the 4 enemies other than your
# lane opponent. The adjustment controls only for the teammates' part (outside your
# control); your own lane lead -- and any win your style causes -- stays in as style.
TEAM_STATE_COLUMNS = ("team_kill_diff_per_min", "team_gold_diff_per_min")
LANE_STATE_COLUMNS = ("lane_kill_diff_per_min", "lane_gold_diff_per_min")
TEAMMATE_STATE_COLUMNS = ("teammates_kill_diff_per_min", "teammates_gold_diff_per_min")
GAME_STATE_COLUMNS = TEAM_STATE_COLUMNS + LANE_STATE_COLUMNS + TEAMMATE_STATE_COLUMNS
STATE_COLUMNS = TEAMMATE_STATE_COLUMNS
# Covariates used by models fitted before the teammates-only change (win flag + full team
# lead). Kept so older saved artifacts still apply.
LEGACY_STATE_COLUMNS = ("win",) + TEAM_STATE_COLUMNS
# Game length enters the adjustment as a cubic spline (see fit_outcome_adjuster).
LENGTH_SPLINE_KNOTS = 6
# Champions with fewer games than this in a role share one adjustment with the rest of
# their class (see champion_groups) -- too few games to estimate their own typical stats.
MIN_CHAMPION_GAMES = 30

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

# Kill-count features that mostly restate "got kills" (and depend heavily on how the game
# went). kills_per_min and solo_kills stay as the style profile's kill dimension; these
# remain in the feature table for percentile comparisons.
STYLE_EXCLUDED_REDUNDANT = frozenset(
    {"kda", "killing_sprees", "largest_killing_spree", "largest_multikill", "multikills"}
)

CLUSTER_FEATURE_COLUMNS = [
    c for c in FEATURE_COLUMN_NAMES if c not in CLUSTERING_EXCLUDED_DUPLICATES | STYLE_EXCLUDED_REDUNDANT
]


def add_game_state_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Attach team / own-lane / teammates' kill and gold differentials per minute, plus
    the team-context features (team_gold_share, lane_gold_diff_per_min). No-op if present.

    Must be called on the full feature table, not a subset of one player's rows.

    Derived from the feature table itself: within a match, the 5 rows sharing a `win`
    value are one team. Rows whose opposing team or lane opponent is missing from the
    table (shouldn't happen after build_features' remake filter, but possible for partial
    data or lane swaps) get 0 for that part, i.e. "even".
    """
    df = add_team_context_features(df)
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

    out = df.drop(columns=[c for c in GAME_STATE_COLUMNS if c in df.columns and c not in LANE_STATE_COLUMNS]).merge(
        merged[["matchId", "win", *TEAM_STATE_COLUMNS]],
        on=["matchId", "win"],
        how="left",
    )
    for column in TEAM_STATE_COLUMNS:
        out[column] = out[column].fillna(0.0)

    # lane_gold_diff_per_min is also a style feature (computed in add_team_context_features);
    # it's reused here as-is. Missing lane opponent -> 0 for the decomposition only.
    out["lane_kill_diff_per_min"] = out["kills_per_min"] - lane_opponent_values(out, "kills_per_min")
    for stat in ("kill", "gold"):
        own = out[f"lane_{stat}_diff_per_min"].fillna(0.0)
        out[f"teammates_{stat}_diff_per_min"] = out[f"team_{stat}_diff_per_min"] - own
    out["lane_kill_diff_per_min"] = out["lane_kill_diff_per_min"].fillna(0.0)
    return out


def _game_minutes(df: pd.DataFrame) -> np.ndarray:
    return (df["gameLength"].astype(float).to_numpy() / 60.0).reshape(-1, 1)


def _covariate_matrix(
    state: np.ndarray, minutes: np.ndarray, length_spline: SplineTransformer
) -> np.ndarray:
    return np.hstack([state, length_spline.transform(minutes)])


def _row_covariates(
    df: pd.DataFrame, length_spline: SplineTransformer, state_columns=STATE_COLUMNS
) -> np.ndarray:
    return _covariate_matrix(df[list(state_columns)].astype(float).to_numpy(), _game_minutes(df), length_spline)


def _typical_state(state_columns) -> np.ndarray:
    """The "typical game" state: even teams (0 lead); win=0.5 for legacy artifacts."""
    return np.array([[0.5 if column == "win" else 0.0 for column in state_columns]])


def champion_groups(champions: pd.Series, classes: dict[str, str], min_games: int) -> pd.Series:
    """Champion name if it has >= min_games rows, else its class pool ("class:Mage")."""
    counts = champions.value_counts()
    common = set(counts[counts >= min_games].index)
    return champions.map(lambda name: name if name in common else f"class:{champion_class(name, classes)}")


def _champion_onehot(champions: pd.Series, adjuster: dict) -> np.ndarray:
    """One-hot champion-group matrix for rows, using the groups seen at fit time.

    A champion not seen at fit time falls back to its class pool; if the class pool wasn't
    seen either, the row gets the population's champion mix, i.e. no champion adjustment.
    """
    levels: list[str] = adjuster["champion_group_levels"]
    shares: np.ndarray = adjuster["champion_group_shares"]
    index = {level: i for i, level in enumerate(levels)}
    onehot = np.tile(shares, (len(champions), 1))
    for row, name in enumerate(champions.to_numpy()):
        group = name if name in index else f"class:{champion_class(name, adjuster['champion_classes'])}"
        if group in index:
            onehot[row] = 0.0
            onehot[row, index[group]] = 1.0
    return onehot


def fit_outcome_adjuster(
    role_df: pd.DataFrame,
    champion_classes: dict[str, str] | None = None,
    min_champion_games: int = MIN_CHAMPION_GAMES,
    feature_columns: list[str] | None = None,
) -> dict:
    """Fit the per-role feature-on-context regression. role_df must have game-state columns.

    Covariates: the teammates' kill/gold differential per minute (your 4 teammates vs the
    4 enemies other than your lane opponent -- deliberately *not* the win flag or the full
    team lead, so a style that wins games isn't subtracted out), and a smooth (cubic
    spline) function of game length. Game length matters on its own: champion level, killing
    sprees, turret plates, max lane leads etc. are per-game totals/maxima that grow with
    length -- non-linearly (champion level follows the XP curve and caps at 18) -- and
    even per-minute rates like gold/min drift with it. Without this, the shortest games
    formed their own "low everything" cluster.

    With champion_classes, the champion played is a covariate too (one indicator per
    champion, rare champions pooled by class), so residuals mean "how this game differs
    from a typical game *on the same champion*" -- otherwise playstyle mostly reduces to
    which champion was picked (Malphite always "tanks more" than Jinx).

    feature_columns defaults to the clustering features; the performance view passes its own.
    """
    columns = list(feature_columns) if feature_columns is not None else list(CLUSTER_FEATURE_COLUMNS)
    medians = role_df[columns].astype(float).median()
    features = role_df[columns].astype(float).fillna(medians)
    minutes = _game_minutes(role_df)
    # Knots at length quantiles so the curve has resolution where games actually are;
    # constant extrapolation keeps unusually short/long games from blowing up.
    length_spline = SplineTransformer(
        n_knots=LENGTH_SPLINE_KNOTS, degree=3, knots="quantile", extrapolation="constant"
    ).fit(minutes)
    covariates = _row_covariates(role_df, length_spline, STATE_COLUMNS)
    covariate_names = list(STATE_COLUMNS) + ["game_minutes_spline"]

    champion_terms: dict = {}
    if champion_classes is not None:
        groups = champion_groups(role_df["championName"], champion_classes, min_champion_games)
        levels = sorted(groups.unique())
        onehot = (groups.to_numpy()[:, None] == np.array(levels)[None, :]).astype(float)
        champion_terms = {
            "champion_group_levels": levels,
            # The population's champion mix -- the "typical game" champion (see apply).
            "champion_group_shares": onehot.mean(axis=0),
            "champion_classes": dict(champion_classes),
            "champion_min_games": min_champion_games,
        }
        covariates = np.hstack([covariates, onehot])
        covariate_names.append("champion_group")

    regression = LinearRegression().fit(covariates, features)
    return {
        "regression": regression,
        "length_spline": length_spline,
        "median_game_minutes": float(np.median(minutes)),
        "feature_medians": medians,
        "feature_columns": columns,
        "state_columns": list(STATE_COLUMNS),
        "covariates": covariate_names,
        **champion_terms,
    }


def apply_outcome_adjuster(rows: pd.DataFrame, adjuster: dict) -> pd.DataFrame:
    """Residualize rows' features against game context, keeping each feature's scale.

    Residual + the prediction for a "typical game" (even teammates, median game length for
    the role, and -- if champion-adjusted -- the role's average champion mix) keeps
    adjusted values in readable units, e.g. "champion level this game would have had in
    an even, median-length game".
    """
    columns = adjuster["feature_columns"]
    # Artifacts saved before the teammates-only change have no "state_columns".
    state_columns = adjuster.get("state_columns", LEGACY_STATE_COLUMNS)
    features = rows[columns].fillna(adjuster["feature_medians"])
    regression: LinearRegression = adjuster["regression"]
    length_spline: SplineTransformer = adjuster["length_spline"]
    covariates = _row_covariates(rows, length_spline, state_columns)
    typical = _covariate_matrix(
        _typical_state(state_columns),
        np.array([[adjuster["median_game_minutes"]]]),
        length_spline,
    )
    if "champion_group_levels" in adjuster:  # models fitted before champion adjustment lack this
        covariates = np.hstack([covariates, _champion_onehot(rows["championName"], adjuster)])
        typical = np.hstack([typical, adjuster["champion_group_shares"].reshape(1, -1)])
    predicted = regression.predict(covariates)
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
    ("Vision Control", (
        "vision_score_per_minute", "wards_placed_per_min", "wards_killed_per_min", "control_wards_per_min",
        "early_wards_placed", "early_wards_killed", "early_control_wards",
    )),
    ("Farm-Focused", ("cs_per_min", "lane_minions_first_10", "gold_earned_per_min")),
    ("Lane Bully", (
        "max_cs_advantage_lane_opp", "max_level_lead_lane_opp", "turret_plates_taken", "lane_gold_diff_per_min",
        "cs_diff_at_10", "gold_diff_at_14", "xp_diff_at_14",
    )),
    ("Duelist", ("solo_kills", "kills_per_min", "early_solo_kills")),
    ("Teamfighter", ("kill_participation", "assists_per_min", "early_kill_participation")),
    # Own building damage only: turret takedown counts mostly track whether the team won.
    ("Split-Pusher", ("damage_to_buildings_per_min",)),
    # No "Objective Control" trait: objective damage duplicates building damage for laners
    # (it includes towers), the epic-objective share is mostly game-to-game noise, and
    # junglers sit at 100% of it in most games. Objective participation is a performance
    # gap stat instead (performance.GAP_FEATURES).
    ("Jungle Farmer", ("neutral_cs_per_min",)),
    ("Skillshot Mage", ("skillshots_hit_per_min", "ability_uses_per_min")),
    # Named for what it measures: deaths carry no built-in payoff, so "High-Risk" overstated it.
    ("Death-Prone", ("deaths_per_min", "early_deaths")),
    # Trading in lane before 14:00 (damage dealt and absorbed), vs. farming passively.
    ("Early Trading", ("damage_to_champions_at_14", "damage_taken_at_14")),
    # Share of the team's gold: "the team plays through me" vs. "I play off the team".
    ("Resource Priority", ("team_gold_share",)),
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


# Plain-language phrasing of each trait for card descriptions ("More X; less Y").
TRAIT_PHRASES: dict[str, str] = {
    "Frontline": "damage absorbed",
    "Crowd Control": "crowd control",
    "Damage Carry": "damage output",
    "Enchanter": "heals & shields",
    "Vision Control": "vision",
    "Farm-Focused": "farming",
    "Lane Bully": "lane leads",
    "Duelist": "kills & solo kills",
    "Teamfighter": "kill participation",
    "Split-Pusher": "tower damage",
    "Jungle Farmer": "jungle camps",
    "Skillshot Mage": "ability usage",
    "Death-Prone": "deaths",
    "Early Trading": "early trading",
    "Resource Priority": "gold share",
}

# Role-specific named archetypes, each defined by a trait signature (+ = more of the
# trait, - = less). A cluster takes the archetype whose signature best matches its trait
# profile; generic trait-pair names are only a fallback. This keeps names meaningful
# for the role (an ADC who absorbs damage and dies more is a "skirmisher", not a
# "frontline") and stable across the dashboard's k = 3-6 refits.
ROLE_ARCHETYPES: dict[str, dict[str, dict[str, float]]] = {
    "BOTTOM": {
        "Hyper-Carry": {"Damage Carry": 1, "Duelist": 1, "Split-Pusher": 0.5},
        "Safe Farmer": {"Farm-Focused": 1, "Lane Bully": 0.7, "Death-Prone": -1, "Frontline": -0.7},
        "Aggressive Skirmisher": {"Death-Prone": 1, "Frontline": 1, "Duelist": 0.4, "Farm-Focused": -0.5},
        "Vision-Focused Carry": {"Vision Control": 1, "Farm-Focused": 0.4},
        "Low-Resource": {"Damage Carry": -1, "Farm-Focused": -1, "Duelist": -0.5},
        "Utility Mage": {"Enchanter": 1, "Crowd Control": 1, "Teamfighter": 0.5},
        "Ability-Heavy Carry": {"Skillshot Mage": 1.5},
    },
    "TOP": {
        "Carry Duelist": {"Duelist": 1, "Damage Carry": 1, "Lane Bully": 0.5},
        "Tank / Frontline": {"Frontline": 1, "Crowd Control": 0.8, "Teamfighter": 0.5, "Damage Carry": -0.5},
        "Lane Farmer": {"Farm-Focused": 1, "Lane Bully": 0.7, "Death-Prone": -0.7},
        "Split-Pusher": {"Split-Pusher": 1, "Teamfighter": -0.7},
        "Low-Resource": {"Damage Carry": -1, "Farm-Focused": -1},
        "Vision-Focused": {"Vision Control": 1},
    },
    "JUNGLE": {
        "Carry Jungler": {"Damage Carry": 1, "Duelist": 1},
        "Tank / Engage": {"Frontline": 1, "Crowd Control": 0.8, "Teamfighter": 0.6},
        "Farming Jungler": {"Jungle Farmer": 1, "Death-Prone": -0.6, "Frontline": -0.4},
        "Vision / Enabler": {"Vision Control": 1, "Teamfighter": 0.6},
        "Ranged / Poke Jungler": {"Skillshot Mage": 1, "Vision Control": 0.6},
        "Low-Resource": {"Damage Carry": -1, "Jungle Farmer": -1},
    },
    "MIDDLE": {
        "Assassin / Carry": {"Duelist": 1, "Damage Carry": 1},
        "Control Mage": {"Vision Control": 0.8, "Crowd Control": 0.8, "Farm-Focused": 0.6, "Death-Prone": -0.7},
        "Lane-Dominant Carry": {"Lane Bully": 1, "Farm-Focused": 0.7, "Split-Pusher": 0.5},
        "Frontline Skirmisher": {"Frontline": 1, "Death-Prone": 0.7},
        "Roaming Playmaker": {"Teamfighter": 1, "Crowd Control": 0.5, "Farm-Focused": -0.5},
        "Scaling Farmer": {"Farm-Focused": 1, "Lane Bully": 0.5, "Teamfighter": -0.5},
        "Low-Resource": {"Damage Carry": -1, "Farm-Focused": -1},
    },
    "UTILITY": {
        "Damage Support": {"Damage Carry": 1, "Duelist": 0.8},
        "Enchanter": {"Enchanter": 1, "Frontline": -0.6},
        "Engage Tank": {"Frontline": 1, "Crowd Control": 0.8, "Vision Control": 0.4},
        "Vision Support": {"Vision Control": 1},
        "Mage / Poke Support": {"Damage Carry": 0.8, "Split-Pusher": 0.6, "Skillshot Mage": 0.6, "Duelist": -0.3},
        "Low-Resource": {"Vision Control": -1, "Damage Carry": -0.5, "Enchanter": -0.5},
    },
}
# Minimum signature match (cosine similarity) for an archetype name to apply.
MIN_ARCHETYPE_MATCH = 0.45


def _signature_match(scores: pd.Series, signature: dict[str, float]) -> float:
    traits = [t for t in signature if t in scores.index]
    if not traits:
        return -1.0
    sig = np.array([signature[t] for t in traits])
    vec = scores.reindex(traits).to_numpy()
    # Cosine against the full trait vector, so a cluster that's extreme on traits the
    # signature doesn't mention matches less well.
    full_norm = np.linalg.norm(scores.to_numpy())
    if full_norm == 0:
        return -1.0
    return float(sig @ vec / (np.linalg.norm(sig) * full_norm))


def _fallback_name(scores: pd.Series) -> str:
    positive = [t for t, s in scores.items() if s >= MIN_TRAIT_SCORE]
    if not positive:
        return LOW_PROFILE_NAME
    return " · ".join(positive[:2])


def name_clusters(
    adjusted_scaled: np.ndarray,
    labels: np.ndarray,
    feature_columns: list[str],
    role: str | None = None,
) -> dict[int, dict]:
    """Return {cluster_id: {"name", "traits", "low_traits"}} from standardized adjusted features.

    Clusters are matched to the role's named archetypes greedily, best (cluster,
    archetype) match first, each archetype used at most once; clusters left without a
    good match fall back to their top-two trait names. "traits"/"low_traits" are plain
    phrases (TRAIT_PHRASES) for the card description.
    """
    frame = pd.DataFrame(adjusted_scaled, columns=feature_columns)
    centroids = frame.groupby(labels).mean()
    scores_by_cluster = {int(c): trait_scores(centroids.loc[c], role) for c in centroids.index}

    archetypes = ROLE_ARCHETYPES.get(role, {}) if role else {}
    candidates = sorted(
        (
            (_signature_match(scores, signature), cluster_id, name)
            for cluster_id, scores in scores_by_cluster.items()
            for name, signature in archetypes.items()
        ),
        reverse=True,
    )
    names: dict[int, str] = {}
    used: set[str] = set()
    for match, cluster_id, name in candidates:
        if match < MIN_ARCHETYPE_MATCH:
            break
        if cluster_id in names or name in used:
            continue
        names[cluster_id] = name
        used.add(name)

    result: dict[int, dict] = {}
    for cluster_id, scores in scores_by_cluster.items():
        name = names.get(cluster_id) or _fallback_name(scores)
        base, suffix = name, 2
        while name in used and names.get(cluster_id) != name:
            name = f"{base} ({suffix})"
            suffix += 1
        used.add(name)
        positive = [TRAIT_PHRASES[t] for t, s in scores.items() if s >= MIN_TRAIT_SCORE][:3]
        low = [TRAIT_PHRASES[t] for t, s in scores.sort_values().items() if s <= -MIN_TRAIT_SCORE][:2]
        result[cluster_id] = {"name": name, "traits": positive, "low_traits": low}
    return result
