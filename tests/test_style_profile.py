"""
Phase 2 tests: trait axes, soft archetype labels and profile confidence (style_profile.py).

Mechanics only -- the tests pass their own small trait definitions, so they don't depend
on the final feature -> trait mapping (that's a separate decision in playstyle_space.TRAITS).

Pipeline under test, per role:
  game-level adjusted features -> standardized (z) -> per-game trait scores (mean z of the
  trait's available features) -> player profile (mean over games, per-trait game counts)
  -> percentile vs other players' profiles, soft archetype labels, confidence.

Early (timeline) features are adjusted for champion only -- not for the teammates' lead
or game length, which are measured at the end of the game, after the 14:00 snapshot --
and stay NaN for games without a timeline instead of being imputed.
"""

import math

import numpy as np
import pandas as pd
import pytest

from style_profile import (
    apply_early_adjuster,
    confidence_label,
    fit_early_adjuster,
    fit_standardizer,
    game_trait_scores,
    player_profiles,
    profile_reliability,
    single_game_reliability,
    soft_labels,
    standardize,
    trait_percentiles,
)

CLASSES = {"malphite": "Tank", "jinx": "Marksman"}
EARLY = ["gold_diff_at_14", "early_wards_placed"]
TRAITS = [
    ("Lane Bully", ("gold_diff_at_14", "lane_gold_diff_per_min")),
    ("Vision Control", ("early_wards_placed", "vision_score_per_minute")),
]


# --- Early-feature adjustment: champion only, NaN preserved -----------------------------------

def early_df(seed=0, n=600):
    rng = np.random.default_rng(seed)
    champion = np.where(np.arange(n) % 2 == 0, "Malphite", "Jinx")
    gold = np.where(champion == "Malphite", -200.0, 300.0) + rng.normal(0, 100, n)
    has_timeline = np.arange(n) % 3 != 0  # a third of games have no timeline
    return pd.DataFrame({
        "championName": champion,
        "gold_diff_at_14": np.where(has_timeline, gold, np.nan),
        "early_wards_placed": np.where(has_timeline, rng.poisson(5, n).astype(float), np.nan),
        # End-of-game context that must NOT enter the early adjustment:
        "teammates_gold_diff_per_min": rng.normal(0, 100, n),
        "gameLength": rng.uniform(1200, 2400, n),
    })


def test_early_adjustment_removes_champion_effect():
    df = early_df()
    adjuster = fit_early_adjuster(df, EARLY, CLASSES, min_champion_games=30)
    out = apply_early_adjuster(df, adjuster)
    means = out.groupby(df["championName"])["gold_diff_at_14"].mean()
    assert abs(means["Malphite"] - means["Jinx"]) < 25


def test_early_adjustment_ignores_end_of_game_context():
    # Shifting the end-of-game columns must not change adjusted early values at all.
    df = early_df()
    adjuster = fit_early_adjuster(df, EARLY, CLASSES, min_champion_games=30)
    shifted = df.assign(teammates_gold_diff_per_min=df["teammates_gold_diff_per_min"] + 500, gameLength=df["gameLength"] * 2)
    pd.testing.assert_frame_equal(apply_early_adjuster(df, adjuster), apply_early_adjuster(shifted, adjuster))


def test_games_without_timeline_stay_nan():
    df = early_df()
    out = apply_early_adjuster(df, fit_early_adjuster(df, EARLY, CLASSES, min_champion_games=30))
    missing = df["gold_diff_at_14"].isna()
    assert out.loc[missing, "gold_diff_at_14"].isna().all()
    assert out.loc[~missing, "gold_diff_at_14"].notna().all()


# --- Standardizing and trait scores ----------------------------------------------------------------

def test_standardize_uses_population_scale():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 10.0, 30.0, np.nan]})
    z = standardize(df, fit_standardizer(df))
    assert z["a"].mean() == pytest.approx(0.0)
    assert z["a"].std(ddof=0) == pytest.approx(1.0)
    assert math.isnan(z.loc[3, "b"])  # NaN stays NaN


def test_trait_score_is_mean_of_available_features():
    z = pd.DataFrame({
        "gold_diff_at_14": [1.0, np.nan, np.nan],
        "lane_gold_diff_per_min": [3.0, 2.0, np.nan],
        "early_wards_placed": [0.0, 0.0, 0.0],
        "vision_score_per_minute": [-2.0, 1.0, 1.0],
    })
    scores = game_trait_scores(z, TRAITS)
    assert scores.loc[0, "Lane Bully"] == pytest.approx(2.0)   # mean(1, 3)
    assert scores.loc[1, "Lane Bully"] == pytest.approx(2.0)   # no timeline: end-of-game part only
    assert math.isnan(scores.loc[2, "Lane Bully"])             # nothing available
    assert scores.loc[0, "Vision Control"] == pytest.approx(-1.0)


# --- Player profiles and percentiles ----------------------------------------------------------------

def test_player_profile_is_mean_over_games_with_counts():
    scores = pd.DataFrame({"Lane Bully": [1.0, 3.0, np.nan, 0.0], "Vision Control": [0.0, 0.0, 1.0, 2.0]})
    puuids = pd.Series(["me", "me", "me", "other"])
    profiles, counts = player_profiles(scores, puuids)
    assert profiles.loc["me", "Lane Bully"] == pytest.approx(2.0)
    assert counts.loc["me", "Lane Bully"] == 2      # the NaN game doesn't count
    assert counts.loc["me", "Vision Control"] == 3


def test_trait_percentile_vs_reference_profiles():
    reference = pd.DataFrame({"Lane Bully": [0.0, 1.0, 2.0, 3.0]})
    pct = trait_percentiles(pd.Series({"Lane Bully": 1.5}), reference)
    assert pct["Lane Bully"] == pytest.approx(50.0)


# --- Soft archetype labels --------------------------------------------------------------------------

ARCHETYPES = {
    "BOTTOM": {
        "Safe Farmer": {"Farm-Focused": 1, "Death-Prone": -1},
        "Aggressive Skirmisher": {"Death-Prone": 1, "Duelist": 1},
    }
}


def test_soft_label_picks_best_matching_archetype():
    profile = pd.Series({"Farm-Focused": 1.2, "Death-Prone": -0.9, "Duelist": 0.1})
    labels = soft_labels(profile, "BOTTOM", ARCHETYPES)
    assert labels[0][0] == "Safe Farmer"
    assert labels[0][1] > labels[1][1]


def test_soft_label_match_is_a_bounded_percentage():
    # Negative similarity (the opposite profile) clamps to 0%, never below.
    profile = pd.Series({"Farm-Focused": -1.0, "Death-Prone": 1.0, "Duelist": -1.0})
    for _, match in soft_labels(profile, "BOTTOM", ARCHETYPES):
        assert 0.0 <= match <= 100.0
    assert dict(soft_labels(profile, "BOTTOM", ARCHETYPES))["Safe Farmer"] == 0.0


def test_soft_labels_returns_top_n():
    profile = pd.Series({"Farm-Focused": 1.0, "Death-Prone": 0.5, "Duelist": 0.5})
    assert len(soft_labels(profile, "BOTTOM", ARCHETYPES, top=1)) == 1


# --- Confidence ---------------------------------------------------------------------------------------

def test_single_game_reliability_recovers_known_icc():
    # Player effect variance 1, game-to-game noise variance 3 -> ICC = 1 / (1 + 3) = 0.25.
    rng = np.random.default_rng(1)
    players, games = 400, 30
    effect = rng.normal(0, 1, players)
    values = np.repeat(effect, games) + rng.normal(0, math.sqrt(3), players * games)
    scores = pd.DataFrame({"Lane Bully": values})
    puuids = pd.Series(np.repeat([f"p{i}" for i in range(players)], games))
    # Compare against this sample's realized player-effect variance, not the nominal 1:
    # with seed 1 the 400 drawn effects have variance ~0.83, so the true ICC here is
    # ~0.83 / (0.83 + 3) ~= 0.217 (asserting 0.25 would test the RNG, not the estimator).
    expected = effect.var(ddof=1) / (effect.var(ddof=1) + 3)
    assert single_game_reliability(scores, puuids)["Lane Bully"] == pytest.approx(expected, abs=0.03)


def test_profile_reliability_grows_with_games():
    # Spearman-Brown: n games at single-game reliability r -> n*r / (1 + (n-1)*r).
    assert profile_reliability(0.25, 20) == pytest.approx(20 * 0.25 / (1 + 19 * 0.25))
    assert profile_reliability(0.25, 5) < profile_reliability(0.25, 20) < profile_reliability(0.25, 100)
    assert profile_reliability(0.25, 0) == 0.0


@pytest.mark.parametrize("reliability, label", [(0.3, "low"), (0.6, "medium"), (0.85, "high")])
def test_confidence_labels(reliability, label):
    assert confidence_label(reliability) == label
