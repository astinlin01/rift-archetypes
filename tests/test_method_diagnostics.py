"""Method diagnostics: the numbers the README's results block quotes.

Three per-feature measures, each computed on raw and on adjusted features:
- champion share of variance (eta squared): how much of a feature is just "which champion"
- |correlation with win|: how much of a feature is just "did the game go well"
- split-half reliability of player means: is there a stable per-player signal at all

Every expected value below is worked out by hand in the comment next to it.
"""
import numpy as np
import pandas as pd
import pytest

from method_diagnostics import (
    champion_variance_share,
    split_half_reliability,
    summarize_role,
    win_correlation,
)


# --- champion share of variance --------------------------------------------------------

def test_champion_variance_share_hand_checked():
    # A: 1, 3 (mean 2)   B: 5, 7 (mean 6)   grand mean 4
    # SS_total   = 9 + 1 + 1 + 9           = 20
    # SS_between = 2*(2-4)^2 + 2*(6-4)^2   = 16      -> 16/20 = 0.8
    values = pd.DataFrame({"f": [1.0, 3.0, 5.0, 7.0]})
    champs = pd.Series(["A", "A", "B", "B"])
    assert champion_variance_share(values, champs)["f"] == pytest.approx(0.8)


def test_champion_variance_share_is_zero_when_champions_have_equal_means():
    # A: 1, 3 (mean 2)   B: 3, 1 (mean 2) -> no between-champion variance
    values = pd.DataFrame({"f": [1.0, 3.0, 3.0, 1.0]})
    champs = pd.Series(["A", "A", "B", "B"])
    assert champion_variance_share(values, champs)["f"] == pytest.approx(0.0)


def test_champion_variance_share_ignores_missing_values():
    # The NaN row is dropped, leaving the 0.8 case above.
    values = pd.DataFrame({"f": [1.0, 3.0, 5.0, 7.0, np.nan]})
    champs = pd.Series(["A", "A", "B", "B", "B"])
    assert champion_variance_share(values, champs)["f"] == pytest.approx(0.8)


# --- correlation with winning ----------------------------------------------------------

def test_win_correlation_hand_checked_and_absolute():
    # x = 1,2,3,4 (deviations -1.5,-.5,.5,1.5; SS 5), win = 0,0,1,1 (deviations ±.5; SS 1)
    # cross-products .75+.25+.25+.75 = 2  ->  r = 2 / sqrt(5 * 1) = 0.8944
    win = pd.Series([False, False, True, True])
    values = pd.DataFrame({"up": [1.0, 2.0, 3.0, 4.0], "down": [4.0, 3.0, 2.0, 1.0]})
    out = win_correlation(values, win)
    assert out["up"] == pytest.approx(2 / np.sqrt(5))
    assert out["down"] == pytest.approx(2 / np.sqrt(5))  # reported as |r|


# --- split-half reliability ------------------------------------------------------------

def test_split_half_perfect_when_players_are_consistent():
    # Each player always produces the same value -> both halves agree exactly -> r = 1.
    values = pd.DataFrame({"f": [1.0] * 4 + [2.0] * 4 + [3.0] * 4})
    puuids = pd.Series(["p1"] * 4 + ["p2"] * 4 + ["p3"] * 4)
    assert split_half_reliability(values, puuids, min_games=4)["f"] == pytest.approx(1.0)


def test_split_half_alternates_games_and_is_not_spearman_brown_corrected():
    # Games alternate into halves (1st, 3rd, ... vs 2nd, 4th, ...):
    #   p1: 0,2,0,2 -> halves (0, 2)    p2: 2,0,2,0 -> (2, 0)    p3: 1,1,1,1 -> (1, 1)
    # half A = [0, 2, 1], half B = [2, 0, 1] -> perfectly opposite -> r = -1
    # (reported as the raw correlation, not Spearman-Brown corrected)
    values = pd.DataFrame({"f": [0.0, 2, 0, 2, 2, 0, 2, 0, 1, 1, 1, 1]})
    puuids = pd.Series(["p1"] * 4 + ["p2"] * 4 + ["p3"] * 4)
    assert split_half_reliability(values, puuids, min_games=4)["f"] == pytest.approx(-1.0)


def test_split_half_excludes_players_below_min_games():
    # p4 has 2 games and would break the perfect agreement (halves 10 vs 0) if included.
    values = pd.DataFrame({"f": [1.0] * 4 + [2.0] * 4 + [3.0] * 4 + [10.0, 0.0]})
    puuids = pd.Series(["p1"] * 4 + ["p2"] * 4 + ["p3"] * 4 + ["p4"] * 2)
    assert split_half_reliability(values, puuids, min_games=4)["f"] == pytest.approx(1.0)


# --- role summary ----------------------------------------------------------------------

def test_summarize_role_averages_each_measure_over_features():
    # Two features: "champ" is pure champion (eta^2 = 1), "flat" has no champion effect.
    # Mean champion share = (1 + 0) / 2 = 0.5.
    values = pd.DataFrame({
        "champ": [1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0],
        "flat": [1.0, 3.0, 3.0, 1.0, 1.0, 3.0, 3.0, 1.0],
    })
    rows = pd.DataFrame({
        "championName": ["A"] * 4 + ["B"] * 4,
        "win": [True, False] * 4,
        "puuid": ["p1"] * 4 + ["p2"] * 4,
    })
    out = summarize_role(values, rows, min_games=4)
    assert out["champion_variance_share"] == pytest.approx(0.5)
    assert set(out) == {"champion_variance_share", "abs_win_correlation", "split_half_reliability"}
