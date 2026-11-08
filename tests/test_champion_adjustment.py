"""
Phase 1 tests: champion-relative outcome adjustment (playstyle_space.py) and champion
class lookup (champion_classes.py).

Synthetic data, built so the right answer is known by construction:
- champion "Malphite" plays with a much higher damage_taken_per_min than "Jinx" (a
  champion effect that must be removed), and
- one player ("styled") takes +300 damage/min more than others *on the same champion*
  (a playstyle signal that must survive).
"""

import numpy as np
import pandas as pd
import pytest

from champion_classes import champion_class
from playstyle_space import (
    CLUSTER_FEATURE_COLUMNS,
    apply_outcome_adjuster,
    champion_groups,
    fit_outcome_adjuster,
)

FEATURE = "damage_taken_per_min"
CLASSES = {
    "malphite": "Tank",
    "jinx": "Marksman",
    "orianna": "Mage",
    "syndra": "Mage",
    "fiddlesticks": "Mage",
}
CHAMPION_BASE = {"Malphite": 1400.0, "Jinx": 600.0}
STYLE_BONUS = 300.0
MIN_GAMES = 30


def synthetic_role_df(seed: int = 0, games_per_champion: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for champion, base in CHAMPION_BASE.items():
        for i in range(games_per_champion):
            styled = i % 10 == 0  # 10% of each champion's games belong to the styled player
            win = bool(rng.integers(0, 2))
            row = {
                "championName": champion,
                "puuid": "styled" if styled else f"p{i % 37}",
                "win": win,
                "gameLength": float(rng.uniform(20, 38) * 60),
                "team_kill_diff_per_min": float(rng.normal(0.2 if win else -0.2, 0.1)),
                "team_gold_diff_per_min": float(rng.normal(80 if win else -80, 30)),
            }
            for column in CLUSTER_FEATURE_COLUMNS:
                row[column] = float(rng.normal(1.0, 0.1))
            row[FEATURE] = base + (STYLE_BONUS if styled else 0.0) + (50.0 if win else 0.0) + rng.normal(0, 40)
            rows.append(row)
    return pd.DataFrame(rows)


def adjusted(df: pd.DataFrame) -> pd.DataFrame:
    adjuster = fit_outcome_adjuster(df, champion_classes=CLASSES, min_champion_games=MIN_GAMES)
    out = apply_outcome_adjuster(df, adjuster)
    out["championName"] = df["championName"].to_numpy()
    out["puuid"] = df["puuid"].to_numpy()
    out["win"] = df["win"].to_numpy()
    return out


# --- Champion effect removed, style signal kept ---------------------------------

def test_champion_means_equalized():
    # Raw means differ by ~800 (1400 vs 600). After adjustment they should match.
    out = adjusted(synthetic_role_df())
    means = out.groupby("championName")[FEATURE].mean()
    assert abs(means["Malphite"] - means["Jinx"]) < 15


def test_style_signal_within_champion_survives():
    # The styled player takes +300/min vs others on the same champion; that must remain.
    out = adjusted(synthetic_role_df())
    for champion in CHAMPION_BASE:
        on_champ = out[out["championName"] == champion]
        styled = on_champ[on_champ["puuid"] == "styled"][FEATURE].mean()
        others = on_champ[on_champ["puuid"] != "styled"][FEATURE].mean()
        assert styled - others == pytest.approx(STYLE_BONUS, abs=30)


def test_outcome_still_removed():
    # Wins add +50/min in the raw data; after adjustment win vs loss means should match.
    out = adjusted(synthetic_role_df())
    by_win = out.groupby("win")[FEATURE].mean()
    assert abs(by_win[True] - by_win[False]) < 10


def test_adjusted_values_stay_in_readable_units():
    # Adjusted values are re-centered on a "typical game", so they should land between
    # the two champions' raw levels, not around zero.
    out = adjusted(synthetic_role_df())
    assert 600 < out[FEATURE].mean() < 1700


# --- Rare champions pooled by class ---------------------------------------------------

def test_rare_champions_pooled_into_class():
    df = pd.DataFrame(
        {"championName": ["Malphite"] * 40 + ["Orianna"] * 5 + ["Syndra"] * 5 + ["FiddleSticks"] * 3}
    )
    groups = champion_groups(df["championName"], CLASSES, min_games=MIN_GAMES)
    assert set(groups[df["championName"] == "Malphite"]) == {"Malphite"}
    assert set(groups[df["championName"] != "Malphite"]) == {"class:Mage"}


def test_unseen_champion_at_apply_time_uses_its_class():
    # A champion never seen when fitting (e.g. a rare pick in your own games) must not
    # crash and must be adjusted like its class pool.
    train = synthetic_role_df()
    adjuster = fit_outcome_adjuster(train, champion_classes=CLASSES, min_champion_games=MIN_GAMES)
    new = train.head(3).copy()
    new["championName"] = "Syndra"
    out = apply_outcome_adjuster(new, adjuster)
    assert out[FEATURE].notna().all()


def test_old_artifacts_without_champion_terms_still_work():
    # Models fitted before this change have no champion keys; applying them must not crash.
    train = synthetic_role_df()
    adjuster = fit_outcome_adjuster(train, champion_classes=CLASSES, min_champion_games=MIN_GAMES)
    legacy = {k: v for k, v in adjuster.items() if not k.startswith("champion")}
    legacy["regression"] = fit_outcome_adjuster(
        train, champion_classes=None, min_champion_games=MIN_GAMES
    )["regression"]
    out = apply_outcome_adjuster(train.head(5), legacy)
    assert out[FEATURE].notna().all()


# --- Class lookup ------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name, expected",
    [("Malphite", "Tank"), ("FiddleSticks", "Mage"), ("fiddlesticks", "Mage"), ("NotAChamp", "Unknown")],
)
def test_champion_class_lookup_is_case_insensitive(name, expected):
    assert champion_class(name, CLASSES) == expected
