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


# --- "On your champions, in an even game" adjustment ---------------------------------------------
# The ◆ view and the gaps remove what's outside your control: the champion played, your
# teammates' lead (your 4 teammates vs the 4 enemies other than your lane opponent) and game
# length -- the same context the style profile removes. Your own lane lead stays in.
# Early (timeline, <= 14:00) stats are champion-only: the end-of-game context comes after them.

from performance import apply_performance_adjuster, fit_performance_adjuster, gap_size

PERF_CLASSES = {"jinx": "Marksman", "ezreal": "Marksman"}


def even_game_df(seed: int = 0, n: int = 4000) -> pd.DataFrame:
    """gold = champion offset (Jinx +40) + 0.5 x teammates' lead + player skill (+30 for
    'good') + noise. cs_diff_at_10 (early) also tracks teammates' lead, so we can check it
    is *not* adjusted for it."""
    rng = np.random.default_rng(seed)
    champs = np.where(np.arange(n) % 2 == 0, "Jinx", "Ezreal")
    good = np.arange(n) % 10 == 0
    lead = rng.normal(0, 100, n)
    return pd.DataFrame({
        "puuid": np.where(good, "good", [f"p{i % 50}" for i in range(n)]),
        "championName": champs,
        "gameLength": rng.uniform(20, 38, n) * 60,
        "teammates_gold_diff_per_min": lead,
        "teammates_kill_diff_per_min": lead / 300,
        "gold": 450 + np.where(champs == "Jinx", 40, 0) + 0.5 * lead + np.where(good, 30, 0) + rng.normal(0, 20, n),
        "cs_diff_at_10": 0.02 * lead + rng.normal(0, 3, n),
    })


def adjust_even(df):
    adjuster = fit_performance_adjuster(df, ["gold", "cs_diff_at_10"], PERF_CLASSES, early_columns=["cs_diff_at_10"])
    return apply_performance_adjuster(df, adjuster)


def test_even_game_removes_teammates_lead_and_champion():
    df = even_game_df()
    out = adjust_even(df)
    assert abs(np.corrcoef(out["gold"], df["teammates_gold_diff_per_min"])[0, 1]) < 0.03
    by_champ = out["gold"].groupby(df["championName"]).mean()
    assert by_champ["Jinx"] == pytest.approx(by_champ["Ezreal"], abs=2)


def test_even_game_keeps_player_skill_and_units():
    df = even_game_df()
    out = adjust_even(df)
    good = out.loc[df["puuid"] == "good", "gold"].mean()
    others = out.loc[df["puuid"] != "good", "gold"].mean()
    assert good - others == pytest.approx(30, abs=4)        # built in as +30
    assert out["gold"].mean() == pytest.approx(df["gold"].mean(), abs=2)  # still gold/min


def test_early_stats_are_champion_only():
    df = even_game_df()
    out = adjust_even(df)
    # cs_diff_at_10 tracks the lead by construction (0.02 x sd 100 vs noise 3 -> r ~ 0.55);
    # it must stay, since the end-of-game lead comes after 10:00.
    assert np.corrcoef(out["cs_diff_at_10"], df["teammates_gold_diff_per_min"])[0, 1] > 0.4


def test_missing_values_stay_missing():
    df = even_game_df(n=400)
    df.loc[[3, 7], "gold"] = np.nan
    df.loc[[5], "cs_diff_at_10"] = np.nan
    out = adjust_even(df)
    assert out["gold"].isna().sum() == 2 and out.loc[[3, 7], "gold"].isna().all()
    assert out["cs_diff_at_10"].isna().sum() == 1


# --- Gap size in game terms ------------------------------------------------------------------------

@pytest.mark.parametrize("feature, you, median, expected", [
    # per-minute stat -> per game at the role's typical length: (515 - 504) x 30 = 330 gold
    ("gold_earned_per_min", 504.0, 515.0, {"amount": 330.0, "per_game": True}),
    # lower-is-better: (0.25 - 0.20) x 30 = 1.5 deaths
    ("deaths_per_min", 0.25, 0.20, {"amount": 1.5, "per_game": True}),
    # percent stat (stored as a 0-1 share) -> percentage points: (0.62 - 0.55) x 100 = 7
    ("kill_participation", 0.55, 0.62, {"amount": 7.0, "per_game": False}),
    # anything else -> plain difference in its own units
    ("kda", 3.1, 3.5, {"amount": 0.4, "per_game": False}),
])
def test_gap_size(feature, you, median, expected):
    got = gap_size(feature, you, median, game_minutes=30.0, percent_features=["kill_participation"])
    assert got["amount"] == pytest.approx(expected["amount"])
    assert got["per_game"] == expected["per_game"]
