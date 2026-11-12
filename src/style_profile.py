"""
Trait-axis style profiles with soft archetype labels and a confidence level (Phase 2).

Replaces KMeans as the main "what kind of player am I" view. Per role:

    game-level adjusted features            (playstyle_space.apply_outcome_adjuster for
                                             end-of-game features, apply_early_adjuster
                                             for timeline features)
    -> standardized (z, role population)    fit_standardizer / standardize
    -> per-game trait scores                game_trait_scores: mean z of the trait's
                                             *available* features
    -> player profile                       player_profiles: mean over games, plus how
                                             many games fed each trait
    -> percentile vs other players          trait_percentiles (reference = profiles of
                                             players with 10+ games in the role)
    -> soft archetype labels                soft_labels: cosine match to ROLE_ARCHETYPES
    -> confidence                           single_game_reliability (ICC) +
                                             profile_reliability (Spearman-Brown)

Why traits instead of clusters: silhouette scores of 0.12-0.15 said the data is a
continuum, not groups, so KMeans drew arbitrary lines through it. A trait axis ("78th
percentile Lane Bully") describes the continuum directly; archetype names become
labels ("72% Safe Farmer"), not boxes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

from playstyle_space import MIN_CHAMPION_GAMES, _signature_match, champion_groups
from champion_classes import champion_class

# Spearman-Brown reliability of a profile -> label. 0.7 is the conventional "good enough
# to compare individuals" bar; below 0.5 more than half the profile's variance is noise.
CONFIDENCE_THRESHOLDS = ((0.7, "high"), (0.5, "medium"))


# ---------- Early (timeline) feature adjustment ----------
#
# Early features are measured at/before 14:00, so the end-of-game context the main
# adjuster uses (teammates' lead, game length) is *after* them in time -- adjusting for
# it would subtract out effects the early game caused. They're adjusted for the champion
# played only. With champion as the only covariate, the regression reduces to group
# means: adjusted = value - mean(value | champion group) + mean(value), i.e. "how this
# game compares to a typical game on the same champion, in the role's average units".

def fit_early_adjuster(
    role_df: pd.DataFrame,
    early_columns: list[str],
    champion_classes: dict[str, str],
    min_champion_games: int = MIN_CHAMPION_GAMES,
) -> dict:
    """Per-champion-group means of each early column (NaN rows = no timeline, skipped).

    Groups are decided on all rows (as in the main adjuster). A group with no timeline
    rows for a column simply gets no adjustment for it.
    """
    groups = champion_groups(role_df["championName"], champion_classes, min_champion_games)
    group_means = {}
    overall = {}
    for column in early_columns:
        values = role_df[column].astype(float)
        group_means[column] = values.groupby(groups).mean().dropna().to_dict()
        overall[column] = float(values.mean())
    return {
        "early_columns": list(early_columns),
        "group_means": group_means,
        "overall_means": overall,
        "groups": sorted(set(groups)),
        "champion_classes": dict(champion_classes),
    }


def _row_groups(champions: pd.Series, adjuster: dict) -> pd.Series:
    """Champion's own group if it had one at fit time, else its class pool."""
    known = set(adjuster["groups"])
    classes = adjuster["champion_classes"]
    return champions.map(lambda name: name if name in known else f"class:{champion_class(name, classes)}")


def apply_early_adjuster(rows: pd.DataFrame, adjuster: dict) -> pd.DataFrame:
    """Champion-adjusted early columns. NaN (no timeline / undefined) stays NaN."""
    groups = _row_groups(rows["championName"], adjuster)
    out = {}
    for column in adjuster["early_columns"]:
        overall = adjuster["overall_means"][column]
        # Unseen group -> its mean is the overall mean, i.e. no adjustment.
        group_mean = groups.map(adjuster["group_means"][column]).astype(float).fillna(overall)
        out[column] = rows[column].astype(float) - group_mean + overall
    return pd.DataFrame(out, index=rows.index)


# ---------- Standardizing and trait scores ----------

def fit_standardizer(df: pd.DataFrame) -> dict:
    """Population mean and std (ddof=0) per column, ignoring NaN."""
    std = df.std(ddof=0)
    # A constant column carries no style information: scale 1 maps it to all zeros
    # instead of dividing by zero.
    return {"mean": df.mean(), "std": std.where(std > 0, 1.0)}


def standardize(df: pd.DataFrame, standardizer: dict) -> pd.DataFrame:
    columns = [c for c in df.columns if c in standardizer["mean"].index]
    return (df[columns] - standardizer["mean"][columns]) / standardizer["std"][columns]


def game_trait_scores(z: pd.DataFrame, traits: list[tuple[str, tuple[str, ...]]]) -> pd.DataFrame:
    """Per game: each trait = mean z of its features that are present and non-NaN.

    "Available" matters because games without a timeline have NaN early features; those
    games are scored on the trait's end-of-game features alone instead of imputing a
    fake average early game. A trait with nothing available for a game is NaN.
    """
    scores = {}
    for trait, features in traits:
        present = [f for f in features if f in z.columns]
        if present:
            scores[trait] = z[present].mean(axis=1, skipna=True)
    return pd.DataFrame(scores, index=z.index)


# ---------- Player profiles and percentiles ----------

def player_profiles(scores: pd.DataFrame, puuids: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(mean trait score per player, number of non-NaN games behind each mean)."""
    grouped = scores.groupby(puuids.to_numpy())
    return grouped.mean(), grouped.count()


def trait_percentiles(profile: pd.Series, reference: pd.DataFrame) -> pd.Series:
    """Percentile of each trait vs the reference players' profiles (ties count half)."""
    result = {}
    for trait, value in profile.items():
        ref = reference[trait].dropna() if trait in reference.columns else pd.Series(dtype=float)
        result[trait] = float(percentileofscore(ref, value, kind="mean")) if len(ref) and pd.notna(value) else np.nan
    return pd.Series(result)


# ---------- Soft archetype labels ----------

def soft_labels(
    profile: pd.Series,
    role: str,
    archetypes: dict[str, dict[str, dict[str, float]]],
    top: int = 2,
) -> list[tuple[str, float]]:
    """Top-N (archetype, match %) for a profile, best first.

    Match % = cosine similarity between the profile and the archetype's trait signature
    (playstyle_space._signature_match, which also penalizes being extreme on traits the
    signature doesn't mention), clamped to 0-100: an opposite profile is a 0% match, not
    a negative one.
    """
    profile = profile.dropna()
    matches = [
        (name, float(np.clip(_signature_match(profile, signature) * 100.0, 0.0, 100.0)))
        for name, signature in archetypes.get(role, {}).items()
    ]
    matches.sort(key=lambda item: item[1], reverse=True)
    return matches[:top]


# ---------- Confidence ----------

def single_game_reliability(scores: pd.DataFrame, puuids: pd.Series) -> pd.Series:
    """ICC(1) per trait: the share of game-to-game variance that's between players.

    One-way ANOVA with players as groups (unbalanced form):
        ICC = (MSB - MSW) / (MSB + (n0 - 1) * MSW)
    where n0 is the effective games-per-player. Read it as "how much one game tells you
    about a player's typical value". Only players with 2+ games for the trait count
    (a single game has no within-player spread). Clamped at 0 (sampling noise can push
    the estimate slightly negative when there's no real player effect).
    """
    result = {}
    for trait in scores.columns:
        frame = pd.DataFrame({"x": scores[trait].to_numpy(), "p": puuids.to_numpy()}).dropna()
        sizes = frame.groupby("p")["x"].transform("size")
        frame = frame[sizes >= 2]
        groups = frame.groupby("p")["x"]
        n_i = groups.size().to_numpy(dtype=float)
        k, n = len(n_i), n_i.sum()
        if k < 2:
            result[trait] = np.nan
            continue
        means = groups.mean()
        grand = frame["x"].mean()
        msb = float((n_i * (means.to_numpy() - grand) ** 2).sum() / (k - 1))
        msw = float(((frame["x"] - frame["p"].map(means)) ** 2).sum() / (n - k))
        n0 = (n - (n_i ** 2).sum() / n) / (k - 1)
        denominator = msb + (n0 - 1) * msw
        result[trait] = max(0.0, (msb - msw) / denominator) if denominator > 0 else np.nan
    return pd.Series(result)


def profile_reliability(single_game: float, games: int) -> float:
    """Spearman-Brown: reliability of an average over `games` games."""
    if games <= 0 or not np.isfinite(single_game):
        return 0.0
    return games * single_game / (1 + (games - 1) * single_game)


def confidence_label(reliability: float) -> str:
    for threshold, label in CONFIDENCE_THRESHOLDS:
        if reliability >= threshold:
            return label
    return "low"
