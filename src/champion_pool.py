"""
Champion-pool layer (Phase 3): which champions and classes a player plays, per role.

This is the identity layer that champion adjustment removed from the style profile:
"I play tanks" lives here, "I play this champion more aggressively than most" lives in
the trait axes (style_profile.py).

- effective_champions: concentration as "you play like someone with N champions played
  equally" = 1 / HHI, with HHI estimated without small-sample bias (see below).
- top_champion: most-played champion and its share of games.
- class_mix / player_class_mixes: share of games per class (support subclasses in
  UTILITY, see champion_classes.role_champion_class).
- reference_class_mix / class_share_percentiles: compare against the reference tier's
  players in the same role.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

from champion_classes import SUPPORT_SUBCLASSES, role_champion_class


def effective_champions(counts: pd.Series) -> float:
    """1 / HHI from per-champion game counts, capped at the number of champions played.

    The plain HHI (sum of squared shares) is biased toward concentration in small
    samples: 10 games can't show more than 10 champions. The unbiased version,
        sum n_i (n_i - 1) / (N (N - 1)),
    is the chance that two *different* games are on the same champion. It can
    overshoot the other way (10 games on 10 champions -> 0, i.e. infinitely many), so
    the result is capped at the champions actually played. NaN below 2 games.
    """
    counts = counts[counts > 0].astype(float)
    total = counts.sum()
    if total < 2:
        return np.nan
    same_pairs = float((counts * (counts - 1)).sum())
    distinct = float(len(counts))
    if same_pairs == 0:
        return distinct
    return min(total * (total - 1) / same_pairs, distinct)


def top_champion(counts: pd.Series) -> tuple[str, float]:
    """(most-played champion, its share of games)."""
    return str(counts.idxmax()), float(counts.max() / counts.sum())


def class_mix(
    champions: pd.Series,
    role: str,
    classes: dict[str, str],
    support_overrides: dict[str, str] = SUPPORT_SUBCLASSES,
) -> pd.Series:
    """Share of games per class (sums to 1)."""
    labels = champions.map(lambda name: role_champion_class(name, role, classes, support_overrides))
    return labels.value_counts(normalize=True)


def player_class_mixes(
    df: pd.DataFrame,
    role: str,
    classes: dict[str, str],
    support_overrides: dict[str, str] = SUPPORT_SUBCLASSES,
    min_games: int = 10,
) -> pd.DataFrame:
    """Players (index) x classes: each player's class mix; classes they never play = 0.
    Only players with >= min_games rows in df (df = one role's rows)."""
    sizes = df.groupby("puuid").size()
    eligible = df[df["puuid"].isin(sizes[sizes >= min_games].index)]
    labels = eligible["championName"].map(lambda name: role_champion_class(name, role, classes, support_overrides))
    return pd.crosstab(eligible["puuid"], labels, normalize="index").rename_axis(index=None, columns=None)


def reference_class_mix(player_mixes: pd.DataFrame) -> pd.Series:
    """Mean of the players' mixes: every player weighs the same, so a few players with
    hundreds of games don't define the tier's mix (pooling games would)."""
    return player_mixes.mean()


def class_share_percentiles(my_mix: pd.Series, player_mixes: pd.DataFrame) -> pd.Series:
    """Percentile of my share of each reference class (ties count half). A class I never
    play is a 0% share, not missing -- most players have 0% in several classes, so ties
    at zero are common and matter."""
    return pd.Series({
        cls: float(percentileofscore(player_mixes[cls], float(my_mix.get(cls, 0.0)), kind="mean"))
        for cls in player_mixes.columns
    })
