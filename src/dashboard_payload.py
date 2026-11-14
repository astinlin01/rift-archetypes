"""
Building blocks for the dashboard export (export_dashboard_data.py): everything the static
page needs to recompute profiles, pool stats, percentiles and gaps in the browser
(dashboard/profile_math.js).

The page never sees other players' games, only *reference curves*: for each stat, the
51 percentile cutoffs (0th, 2nd, ..., 100th) of the reference players' averages, plus
their mean, median and count. Reference players = players with 10+ games in the role and
tier, as everywhere else; a tier with fewer than the minimum number of such players
gets no curves, only its player count, so the page can say "insufficient data (n=X)".
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from champion_pool import effective_champions, reference_class_mix
from playstyle_space import ROLE_EXCLUDED_TRAITS, TRAITS, apply_outcome_adjuster, fit_outcome_adjuster
from style_profile import (
    apply_early_adjuster,
    fit_early_adjuster,
    fit_standardizer,
    game_trait_scores,
    single_game_reliability,
    standardize,
)
from timeline_features import TIMELINE_FEATURE_COLUMNS

BREAKPOINT_COUNT = 51  # percentile grid: 0, 2, 4, ..., 100
MIN_PLAYER_GAMES = 10
MIN_REFERENCE_PLAYERS = 20


def json_number(value, digits: int = 4):
    """JSON-safe number: NaN/None -> None, numpy scalars -> Python, floats rounded."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    value = float(value)
    return None if math.isnan(value) else round(value, digits)


def _curve(series: pd.Series) -> dict:
    quantiles = np.linspace(0, 100, BREAKPOINT_COUNT)
    return {
        "ref_avg": json_number(series.mean()),
        "std": json_number(series.std(ddof=0) or 1.0),
        "n": int(len(series)),
        "median": json_number(series.median()),
        "breakpoints": [json_number(v) for v in np.percentile(series, quantiles)],
    }


def _eligible(rows: pd.DataFrame, min_games: int) -> pd.Series:
    """Boolean mask: rows of players with >= min_games rows here."""
    return rows.groupby("puuid")["puuid"].transform("size") >= min_games


def player_average_curves(
    rows: pd.DataFrame,
    values: pd.DataFrame,
    min_games: int = MIN_PLAYER_GAMES,
    min_players: int = MIN_REFERENCE_PLAYERS,
) -> dict:
    """{stat: curve} over the averages of players with >= min_games rows. A player's
    average skips their missing values (e.g. games without a timeline). {} if fewer
    than min_players players qualify."""
    keep = _eligible(rows, min_games)
    averages = values[keep.to_numpy()].groupby(rows.loc[keep, "puuid"].to_numpy()).mean()
    if len(averages) < min_players:
        return {}
    curves = {}
    for column in averages.columns:
        series = averages[column].dropna()
        if len(series) >= min_players:
            curves[column] = _curve(series)
    return curves


def reference_by_tier(
    rows: pd.DataFrame,
    values: pd.DataFrame,
    regions: list[str],
    tiers: list[str],
    min_games: int = MIN_PLAYER_GAMES,
    min_players: int = MIN_REFERENCE_PLAYERS,
) -> dict:
    """{region: {tier: curves}} for tiers with enough players, plus
    "_player_counts": {region: {tier: qualifying players}} for every tier with any."""
    reference: dict = {"_player_counts": {}}
    for region in regions:
        for tier in tiers:
            mask = ((rows["region"] == region) & (rows["tier"] == tier)).to_numpy()
            if not mask.any():
                continue
            subset = rows[mask]
            players = int((subset.groupby("puuid").size() >= min_games).sum())
            reference["_player_counts"].setdefault(region, {})[tier] = players
            curves = player_average_curves(subset, values[mask], min_games, min_players)
            if curves:
                reference.setdefault(region, {})[tier] = curves
    return reference


def pool_reference(rows: pd.DataFrame, min_games: int = MIN_PLAYER_GAMES, min_players: int = MIN_REFERENCE_PLAYERS) -> dict:
    """The tier's champion-pool baseline: average class mix (every player weighs the
    same) and the spread of effective champion counts. rows: one role/tier's games with
    championName and champion_class (support subclasses already applied)."""
    eligible = rows[_eligible(rows, min_games)]
    players = eligible["puuid"].nunique()
    if players < min_players:
        return {"n": int(players)}
    mixes = pd.crosstab(eligible["puuid"], eligible["champion_class"], normalize="index")
    effective = eligible.groupby("puuid")["championName"].apply(lambda s: effective_champions(s.value_counts()))
    return {
        "n": int(players),
        "class_mix": {cls: json_number(share) for cls, share in reference_class_mix(mixes).items()},
        "effective_champions": _curve(effective.dropna()),
    }


def role_traits(role: str) -> list[tuple[str, tuple[str, ...]]]:
    excluded = ROLE_EXCLUDED_TRAITS.get(role, frozenset())
    return [(trait, features) for trait, features in TRAITS if trait not in excluded]


def style_scores(
    role_df: pd.DataFrame,
    role: str,
    champion_classes: dict[str, str],
    min_reliability_games: int = MIN_PLAYER_GAMES,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Per-game trait scores for one role's games, each trait's single-game reliability,
    and the role's trait names (in TRAITS order).

    End-of-game features: adjusted for teammates' lead, game length and champion.
    Timeline features (if the table has them): adjusted for champion only, NaN kept.
    Both are standardized against the role's whole population, then averaged per trait.
    """
    adjusted = apply_outcome_adjuster(role_df, fit_outcome_adjuster(role_df, champion_classes=champion_classes))
    early = [c for c in TIMELINE_FEATURE_COLUMNS if c in role_df.columns]
    if early:
        early_adjuster = fit_early_adjuster(role_df, early, champion_classes)
        adjusted = pd.concat([adjusted, apply_early_adjuster(role_df, early_adjuster)], axis=1)
    z = standardize(adjusted, fit_standardizer(adjusted))

    traits = role_traits(role)
    names = [trait for trait, _ in traits]
    # Traits with no available feature at all (Early Trading before timelines exist)
    # come back as all-missing columns rather than disappearing.
    scores = game_trait_scores(z, traits).reindex(columns=names)

    keep = _eligible(role_df, min_reliability_games).to_numpy()
    icc = single_game_reliability(scores[keep], role_df.loc[keep, "puuid"]).reindex(names)
    return scores, icc, names


# ---------- Style map ----------

def player_trait_points(
    rows: pd.DataFrame,
    scores: pd.DataFrame,
    traits: list[str],
    regions: list[str],
    tiers: list[str],
    min_games: int = MIN_PLAYER_GAMES,
    min_players: int = MIN_REFERENCE_PLAYERS,
) -> dict:
    """{region: {tier: [[trait averages in `traits` order], ...]}}: one point per reference
    player with >= min_games games, for the style map. Anonymous on purpose (no puuid or
    name). 3 decimals is far finer than a dot on screen; null = no data for that trait.
    Tiers with fewer than min_players players are left out (the page says so)."""
    points: dict = {}
    for region in regions:
        for tier in tiers:
            mask = ((rows["region"] == region) & (rows["tier"] == tier)).to_numpy()
            if not mask.any():
                continue
            subset = rows[mask]
            keep = _eligible(subset, min_games).to_numpy()
            if not keep.any():
                continue
            averages = scores.loc[mask, traits][keep].groupby(subset.loc[keep, "puuid"].to_numpy()).mean()
            if len(averages) < min_players:
                continue
            points.setdefault(region, {})[tier] = [
                [json_number(v, 3) for v in row] for row in averages.to_numpy()
            ]
    return points


# ---------- Player card splash art ----------

def splash_champions(games: pd.DataFrame, min_games: int) -> set[str]:
    """Champions that get splash art on the player card: each account's most-played
    champion per role (roles with >= min_games games, where the page shows a profile), plus
    each account's overall most-played one for the all-roles view. Ties -> alphabetical."""
    def top(series: pd.Series) -> str:
        counts = series.value_counts()
        return sorted(counts.index, key=lambda name: (-counts[name], name))[0]

    champions = set()
    for _, account_games in games.groupby("account"):
        champions.add(top(account_games["championName"]))
        for _, role_games in account_games.groupby("teamPosition"):
            if len(role_games) >= min_games:
                champions.add(top(role_games["championName"]))
    return champions


def crop_splash(image_bytes: bytes, width: int = 1200, height: int = 480, quality: int = 72) -> bytes:
    """Crop a Data Dragon splash (1215x717) to the card's wide shape and re-encode small.

    Keeps the full width and takes the band starting 3% down: splash art puts the
    champion's head near the top (12% cut Kai'Sa's off), and the card overlays text on
    the left, so the lower edge matters least."""
    from io import BytesIO

    from PIL import Image

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    source_w, source_h = image.size
    target_ratio = width / height
    if source_w / source_h > target_ratio:  # too wide: trim the sides evenly
        crop_w = round(source_h * target_ratio)
        left = (source_w - crop_w) // 2
        box = (left, 0, left + crop_w, source_h)
    else:  # too tall (the usual case): keep a band from 12% down
        crop_h = round(source_w / target_ratio)
        top = min(round(source_h * 0.03), source_h - crop_h)
        box = (0, top, source_w, top + crop_h)
    out = BytesIO()
    image.crop(box).resize((width, height), Image.LANCZOS).save(out, "JPEG", quality=quality, optimize=True, progressive=True)
    return out.getvalue()
