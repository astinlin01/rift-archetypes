"""
Phase 4 tests: performance comparison (performance.py).

- Champion-adjusted view next to raw: each game's stat minus what's typical on that
  champion (champion only -- no game-state adjustment, so winning still counts), compared
  as player averages against the reference tier's adjusted player averages.
- Biggest gaps = constrained percentile shortfall: only stats with a clear "better"
  direction (lower-is-better ones inverted), worst stat per theme, and gaps your own
  sample can't distinguish from the tier median are flagged as noise, not ranked.
- Your own sample: < 5 games insufficient, 5-9 low confidence, 10+ ok.

Tests pass their own small direction/theme map, so they don't depend on the curated list.
"""

import numpy as np
import pandas as pd
import pytest

from performance import (
    biggest_gaps,
    compare_features,
    fit_champion_adjuster,
    gap_features_for_role,
    own_sample_status,
)

CLASSES = {"malphite": "Tank", "jinx": "Marksman"}


# --- Champion-adjusted vs raw ------------------------------------------------------------------

def reference_df():
    # 10 Malphite players averaging 450..540 damage, 10 Jinx players averaging 950..1040;
    # 10 identical games each (no within-player noise, so values are exact).
    rows = []
    for i in range(10):
        rows += [{"puuid": f"m{i}", "championName": "Malphite", "dmg": 450.0 + 10 * i}] * 10
        rows += [{"puuid": f"j{i}", "championName": "Jinx", "dmg": 950.0 + 10 * i}] * 10
    return pd.DataFrame(rows)


def my_df():
    return pd.DataFrame({"puuid": ["me"] * 10, "championName": ["Malphite"] * 10, "dmg": [600.0] * 10})


def test_raw_percentile_mixes_in_champion_pool():
    # 600 beats all 10 Malphite players and loses to all 10 Jinx players -> 50th.
    out = compare_features(my_df(), reference_df(), ["dmg"])
    assert out.loc["dmg", "your_avg"] == pytest.approx(600.0)
    assert out.loc["dmg", "percentile"] == pytest.approx(50.0)


def test_champion_adjusted_percentile_compares_like_with_like():
    # Group means: Malphite 495, Jinx 995; overall 745.
    # Me: 600 - 495 + 745 = 850. Every reference player adjusts into 700..790 -> 100th.
    ref = reference_df()
    adjuster = fit_champion_adjuster(ref, ["dmg"], CLASSES, min_champion_games=30)
    out = compare_features(my_df(), ref, ["dmg"], adjuster=adjuster)
    assert out.loc["dmg", "your_avg"] == pytest.approx(850.0)
    assert out.loc["dmg", "percentile"] == pytest.approx(100.0)


def test_reference_players_need_min_games():
    ref = pd.concat([reference_df(), pd.DataFrame({"puuid": ["few"] * 3, "championName": ["Jinx"] * 3, "dmg": [0.0] * 3})])
    out = compare_features(my_df(), ref, ["dmg"], min_games=10)
    assert out.loc["dmg", "n_players"] == 20  # the 3-game player is left out


# --- Own sample status -----------------------------------------------------------------------------

@pytest.mark.parametrize("games, status", [(0, "insufficient"), (4, "insufficient"), (5, "low"), (9, "low"), (10, "ok")])
def test_own_sample_status(games, status):
    assert own_sample_status(games) == status


# --- Biggest gaps -------------------------------------------------------------------------------------

GAP_FEATURES = {
    "gold": ("Farming", +1),
    "cs": ("Farming", +1),
    "deaths": ("Survival", -1),
    "vision": ("Vision", +1),
    "kills": ("Combat", +1),
}


def gap_inputs():
    # Reference: 10 players with averages 0..9 on every stat (median 4.5).
    reference = pd.DataFrame({c: np.arange(10, dtype=float) for c in [*GAP_FEATURES, "damage_taken"]})
    # My 4 games. Percentile of a value v vs 0..9 (ties half) = 10*v + 5.
    mine = pd.DataFrame({
        "gold": [2.0] * 4,                 # 25th -> Farming gap, but cs is worse
        "cs": [1.0] * 4,                   # 15th -> Farming's representative
        "deaths": [7.0] * 4,               # 75th, lower is better -> 25th "goodness"
        "vision": [0.0, 6.0, 0.0, 6.0],    # mean 3 -> 35th, but noisy: see test
        "kills": [6.0] * 4,                # 65th -> not a gap
        "damage_taken": [0.0] * 4,         # 5th, but no clear direction -> never a gap
    })
    return mine, reference


def test_gaps_worst_per_theme_in_order():
    mine, reference = gap_inputs()
    gaps = biggest_gaps(mine, reference, GAP_FEATURES, top=3)
    assert [g["feature"] for g in gaps] == ["cs", "deaths"]
    assert gaps[0]["theme"] == "Farming"
    assert gaps[0]["percentile"] == pytest.approx(15.0)
    assert gaps[1]["percentile"] == pytest.approx(25.0)  # inverted: fewer deaths = better


def test_noisy_gap_is_not_ranked():
    # Vision: mean 3, sd 3.46 over 4 games -> SE 1.73; 80% range 3 +/- 1.28*1.73 = 0.78..5.22
    # contains the tier median 4.5, so it can't be told apart from typical.
    mine, reference = gap_inputs()
    assert "vision" not in [g["feature"] for g in biggest_gaps(mine, reference, GAP_FEATURES, top=3)]


def test_gaps_top_n():
    mine, reference = gap_inputs()
    assert [g["feature"] for g in biggest_gaps(mine, reference, GAP_FEATURES, top=1)] == ["cs"]


# --- The curated gap list (reviewed 2026-11-13) ---------------------------------------------------

def test_team_driven_objective_counts_are_not_gaps():
    # Takedown counts mostly track whether the team was winning; the participation
    # shares replace them.
    for role in ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]:
        features = gap_features_for_role(role)
        assert "dragon_takedowns_per_min" not in features
        assert "turret_takedowns_per_min" not in features
        assert features["epic_objective_participation"] == ("Objectives", +1)
        assert features["turret_participation"] == ("Objectives", +1)


def test_solo_kills_only_in_solo_lanes():
    assert gap_features_for_role("TOP")["solo_kills"] == ("Laning", +1)
    assert gap_features_for_role("MIDDLE")["solo_kills"] == ("Laning", +1)
    for role in ["JUNGLE", "BOTTOM", "UTILITY"]:
        assert "solo_kills" not in gap_features_for_role(role)


def test_role_exclusions():
    support = gap_features_for_role("UTILITY")
    assert not [f for f, (theme, _) in support.items() if theme == "Farming"]  # CS taken from the ADC isn't "better"
    assert "lane_minions_first_10" not in gap_features_for_role("JUNGLE")
    assert gap_features_for_role("JUNGLE")["cs_total_per_min"] == ("Farming", +1)


def test_deaths_are_lower_is_better():
    assert gap_features_for_role("BOTTOM")["deaths_per_min"] == ("Survival", -1)


def test_style_stats_are_never_gaps():
    features = gap_features_for_role("TOP")
    for style in ["damage_taken_per_min", "heal_on_teammates_per_min", "cc_time_per_min",
                  "team_gold_share", "team_damage_pct", "wards_placed_per_min"]:
        assert style not in features
