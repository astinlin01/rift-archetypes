"""
Phase 3 tests: champion-pool layer (champion_pool.py + role-aware classes).

- Concentration = effective number of champions, 1 / HHI, with HHI estimated without
  small-sample bias: sum n_i (n_i - 1) / (N (N - 1)) -- the chance two *different* games
  are on the same champion. Capped at the number of distinct champions played (with few
  games the unbiased HHI can reach 0, i.e. "infinitely many champions").
- Top champion share alongside it.
- Class mix per role, with the support override map applied in UTILITY only.
- Reference class mix = mean of each reference player's mix (every player weighs the
  same, matching how trait profiles rank players), compared as both shares and
  percentiles.

Tests pass their own small override map, so they don't depend on the final map content.
"""

import math

import pandas as pd
import pytest

from champion_classes import role_champion_class
from champion_pool import (
    class_mix,
    class_share_percentiles,
    effective_champions,
    player_class_mixes,
    reference_class_mix,
    top_champion,
)

CLASSES = {"lulu": "Support", "thresh": "Support", "karma": "Mage", "lux": "Mage", "nautilus": "Tank", "camille": "Fighter"}
OVERRIDES = {"lulu": "Enchanter", "karma": "Enchanter", "thresh": "Pick", "lux": "Mage", "nautilus": "Tank"}


# --- Concentration --------------------------------------------------------------------------

def test_effective_champions_one_trick():
    # 27 Jinx, 2 Ashe, 1 Ezreal: sum n(n-1) = 27*26 + 2*1 + 0 = 704; N(N-1) = 30*29 = 870.
    counts = pd.Series({"Jinx": 27, "Ashe": 2, "Ezreal": 1})
    assert effective_champions(counts) == pytest.approx(870 / 704)  # ~1.24


def test_effective_champions_capped_at_distinct_played():
    # 10/10/10: unbiased estimate is 870 / 270 = 3.22 -- more than the 3 champions played.
    assert effective_champions(pd.Series({"Jinx": 10, "Ashe": 10, "Ezreal": 10})) == pytest.approx(3.0)
    # 10 games on 10 champions: unbiased HHI = 0 -> capped at 10, not infinite.
    assert effective_champions(pd.Series({f"c{i}": 1 for i in range(10)})) == pytest.approx(10.0)


def test_effective_champions_needs_two_games():
    assert math.isnan(effective_champions(pd.Series({"Jinx": 1})))


def test_top_champion_share():
    name, share = top_champion(pd.Series({"Jinx": 27, "Ashe": 2, "Ezreal": 1}))
    assert name == "Jinx"
    assert share == pytest.approx(0.9)


# --- Role-aware classes ----------------------------------------------------------------------

def test_override_applies_only_in_utility():
    assert role_champion_class("Karma", "UTILITY", CLASSES, OVERRIDES) == "Enchanter"
    assert role_champion_class("Karma", "MIDDLE", CLASSES, OVERRIDES) == "Mage"


def test_unmapped_support_falls_back_to_primary_tag():
    assert role_champion_class("Camille", "UTILITY", CLASSES, OVERRIDES) == "Fighter"


# --- Class mix ------------------------------------------------------------------------------------

def test_class_mix_sums_to_one():
    mix = class_mix(pd.Series(["Lulu", "Lulu", "Lulu", "Lux"]), "UTILITY", CLASSES, OVERRIDES)
    assert mix.sum() == pytest.approx(1.0)
    assert mix["Enchanter"] == pytest.approx(0.75)
    assert mix["Mage"] == pytest.approx(0.25)


def pool_df():
    # A: 4 games, 3 Lulu + 1 Lux       -> Enchanter .75, Mage .25
    # B: 8 games, 4 Nautilus + 4 Lux   -> Tank .5, Mage .5
    # C: 2 games (below the minimum)   -> excluded
    return pd.DataFrame({
        "puuid": ["A"] * 4 + ["B"] * 8 + ["C"] * 2,
        "championName": ["Lulu"] * 3 + ["Lux"] + ["Nautilus"] * 4 + ["Lux"] * 4 + ["Thresh"] * 2,
    })


def test_player_class_mixes_fill_zero_and_apply_minimum():
    mixes = player_class_mixes(pool_df(), "UTILITY", CLASSES, OVERRIDES, min_games=4)
    assert set(mixes.index) == {"A", "B"}
    assert mixes.loc["A", "Tank"] == 0.0
    assert mixes.sum(axis=1).tolist() == pytest.approx([1.0, 1.0])


def test_reference_mix_weighs_players_equally():
    # Per-player mean: Mage (.25 + .5) / 2 = .375. Pooled games would give 5/12 = .417.
    mixes = player_class_mixes(pool_df(), "UTILITY", CLASSES, OVERRIDES, min_games=4)
    reference = reference_class_mix(mixes)
    assert reference["Mage"] == pytest.approx(0.375)
    assert reference["Enchanter"] == pytest.approx(0.375)
    assert reference["Tank"] == pytest.approx(0.25)
    assert reference.sum() == pytest.approx(1.0)


def test_class_share_percentile_counts_ties_half():
    # Tank share 0.5 vs reference players [0, 0, 0.5, 1]: 2 below, 1 tied ->
    # (strictly below 50% + at-or-below 75%) / 2 = 62.5.
    reference = pd.DataFrame({"Tank": [0.0, 0.0, 0.5, 1.0], "Mage": [1.0, 1.0, 0.5, 0.0]})
    pct = class_share_percentiles(pd.Series({"Tank": 0.5}), reference)
    assert pct["Tank"] == pytest.approx(62.5)
    # A class I never play counts as a 0% share, not missing: 0 vs [1, 1, .5, 0] -> 12.5.
    assert pct["Mage"] == pytest.approx(12.5)
