"""
Performance comparison (Phase 4): how your stats rank against a reference tier.

Two views per stat, side by side:
- raw: your average vs the tier's player averages (being ahead *is* the point here, so
  no outcome adjustment).
- on your champions: every game's stat minus what's typical on that champion (champion
  only -- the game state is deliberately left in, so winning still counts). The gap
  between the two views is informative on its own: "35th raw -> 62nd on your champions"
  means your low damage comes from what you pick, not how you play it.

Champion adjustment instead of class matching: it handles mixed pools without splitting
your games into per-class panels, and it's finer than class (Jinx and Ezreal are both
Marksmen with very different damage profiles).

Biggest gaps = constrained percentile shortfall (see biggest_gaps).
"""

from __future__ import annotations

import math

import pandas as pd
from scipy.stats import percentileofscore

from playstyle_space import MIN_CHAMPION_GAMES
from style_profile import apply_early_adjuster, fit_early_adjuster, trait_percentiles

MIN_REFERENCE_PLAYER_GAMES = 10
# Your own games in a role: below MIN_OWN_GAMES -> "insufficient data (n=X)";
# MIN_OWN_GAMES..FULL_OWN_GAMES-1 -> shown with a low-confidence marker.
MIN_OWN_GAMES = 5
FULL_OWN_GAMES = 10
# 80% two-sided normal interval for "is this gap distinguishable from typical?".
GAP_INTERVAL_Z = 1.2816


# Stats that can count as a "gap": only ones where better is clear. {stat: (theme, direction)},
# +1 = higher is better, -1 = lower is better. Reviewed 2026-11-13. Left out on purpose
# (style, not skill): damage taken/mitigated, heals/shields, CC time, ability uses and
# skillshots (mostly the champion), team damage/gold share (how resources are split),
# wards placed (covered by vision score), and objective takedown *counts* (mostly whether
# the team was winning -- the participation shares replace them). Timeline stats are
# skipped automatically until those columns exist.
GAP_FEATURES: dict[str, tuple[str, int]] = {
    "cs_total_per_min": ("Farming", +1),
    "gold_earned_per_min": ("Farming", +1),
    "lane_minions_first_10": ("Farming", +1),
    "lane_gold_diff_per_min": ("Laning", +1),
    "max_cs_advantage_lane_opp": ("Laning", +1),
    "turret_plates_taken": ("Laning", +1),
    "cs_diff_at_10": ("Laning", +1),
    "gold_diff_at_14": ("Laning", +1),
    "xp_diff_at_14": ("Laning", +1),
    "damage_per_minute": ("Combat", +1),
    "kill_participation": ("Combat", +1),
    "kda": ("Combat", +1),
    "deaths_per_min": ("Survival", -1),
    "early_deaths": ("Survival", -1),
    "vision_score_per_minute": ("Vision", +1),
    "control_wards_per_min": ("Vision", +1),
    "wards_killed_per_min": ("Vision", +1),
    "damage_to_objectives_per_min": ("Objectives", +1),
    "epic_objective_participation": ("Objectives", +1),
    "turret_participation": ("Objectives", +1),
    "early_objective_participation": ("Objectives", +1),
}
# Solo kills only where the lane really is 1v1 (bot lane is 2v2, jungle and support roam).
SOLO_LANE_GAP_FEATURES: dict[str, tuple[str, int]] = {"solo_kills": ("Laning", +1)}
SOLO_LANES = frozenset({"TOP", "MIDDLE"})
# Support: taking more CS isn't better, it's taking it from the ADC. Jungle: no lane minions.
ROLE_EXCLUDED_GAP_THEMES = {"UTILITY": frozenset({"Farming"})}
ROLE_EXCLUDED_GAP_FEATURES = {"JUNGLE": frozenset({"lane_minions_first_10"})}


def gap_features_for_role(role: str) -> dict[str, tuple[str, int]]:
    features = dict(GAP_FEATURES)
    if role in SOLO_LANES:
        features.update(SOLO_LANE_GAP_FEATURES)
    excluded_themes = ROLE_EXCLUDED_GAP_THEMES.get(role, frozenset())
    excluded = ROLE_EXCLUDED_GAP_FEATURES.get(role, frozenset())
    return {f: spec for f, spec in features.items() if spec[0] not in excluded_themes and f not in excluded}


def fit_champion_adjuster(
    role_df: pd.DataFrame,
    columns: list[str],
    champion_classes: dict[str, str],
    min_champion_games: int = MIN_CHAMPION_GAMES,
) -> dict:
    """Champion-only adjustment of `columns`. Same math as the early-feature adjuster
    (value - mean on that champion group + role mean), so it's reused rather than
    duplicated. Fit on the whole role population so you and the reference share one
    set of champion baselines."""
    return fit_early_adjuster(role_df, columns, champion_classes, min_champion_games)


def own_sample_status(games: int) -> str:
    if games < MIN_OWN_GAMES:
        return "insufficient"
    return "low" if games < FULL_OWN_GAMES else "ok"


def _player_averages(rows: pd.DataFrame, values: pd.DataFrame, min_games: int) -> pd.DataFrame:
    sizes = rows.groupby("puuid")["puuid"].transform("size")
    keep = sizes >= min_games
    return values[keep].groupby(rows.loc[keep, "puuid"].to_numpy()).mean()


def compare_features(
    my_rows: pd.DataFrame,
    reference_rows: pd.DataFrame,
    columns: list[str],
    adjuster: dict | None = None,
    min_games: int = MIN_REFERENCE_PLAYER_GAMES,
) -> pd.DataFrame:
    """Per stat: your average, the reference players' mean, your percentile among the
    reference players' averages (players with >= min_games), and how many players.
    With an adjuster, both sides are champion-adjusted first."""
    if adjuster is not None:
        mine = apply_early_adjuster(my_rows, adjuster)[columns]
        reference = apply_early_adjuster(reference_rows, adjuster)[columns]
    else:
        mine = my_rows[columns].astype(float)
        reference = reference_rows[columns].astype(float)
    reference_avgs = _player_averages(reference_rows, reference, min_games)
    my_avg = mine.mean()
    return pd.DataFrame({
        "your_avg": my_avg,
        "reference_avg": reference_avgs.mean(),
        "percentile": trait_percentiles(my_avg, reference_avgs),
        "n_players": reference_avgs.count(),
    })


def biggest_gaps(
    my_games: pd.DataFrame,
    reference_avgs: pd.DataFrame,
    gap_features: dict[str, tuple[str, int]],
    top: int = 3,
) -> list[dict]:
    """Your weakest stats vs the reference, at most one per theme, worst first.

    gap_features: {stat: (theme, direction)}, direction +1 = higher is better, -1 =
    lower is better. Stats without a clear direction (damage taken, wards placed) are
    simply not listed there, so they can never show up as a "weakness".

    A stat counts as a gap when its direction-corrected percentile is below 50 *and*
    the tier median lies outside the 80% range of your own average (mean +/- 1.28 SE,
    SE from your game-to-game spread). Otherwise it's indistinguishable from typical at
    your sample size and isn't ranked. "percentile" in the output is direction-
    corrected: lower always means worse.
    """
    best_per_theme: dict[str, dict] = {}
    for feature, (theme, direction) in gap_features.items():
        if feature not in my_games.columns or feature not in reference_avgs.columns:
            continue
        values = my_games[feature].dropna()
        reference = reference_avgs[feature].dropna()
        if values.empty or reference.empty:
            continue
        mean = float(values.mean())
        raw_pct = float(percentileofscore(reference, mean, kind="mean"))
        percentile = raw_pct if direction > 0 else 100.0 - raw_pct
        if percentile >= 50:
            continue
        median = float(reference.median())
        se = float(values.std(ddof=1)) / math.sqrt(len(values)) if len(values) > 1 else math.inf
        if abs(mean - median) <= GAP_INTERVAL_Z * se:
            continue  # within noise
        gap = {
            "feature": feature,
            "theme": theme,
            "percentile": percentile,
            "your_avg": mean,
            "reference_median": median,
            "n_games": int(len(values)),
        }
        if theme not in best_per_theme or percentile < best_per_theme[theme]["percentile"]:
            best_per_theme[theme] = gap
    return sorted(best_per_theme.values(), key=lambda g: g["percentile"])[:top]
