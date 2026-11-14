"""
The dashboard recomputes profiles, labels, confidence, pool stats and gaps in the browser
(so the game-window and champion filters stay live on a static page). That math lives in
dashboard/profile_math.js, a second copy of the Python logic -- these tests run both
copies on the same inputs through Node and require the same answers, so the JS can't
silently drift from the tested Python.

Skipped if Node isn't installed.
"""

import json
import math
import os
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from champion_pool import class_mix, effective_champions, top_champion
from performance import biggest_gaps, own_sample_status
from style_profile import confidence_label, player_profiles, profile_reliability, soft_labels

JS_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "dashboard", "profile_math.js")
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js not installed")


def js(function: str, *args):
    """Call ProfileMath.<function>(*args) in Node and return the JSON result."""
    script = (
        "const M = require(process.argv[1]);"
        "const args = JSON.parse(process.argv[2]);"
        f"process.stdout.write(JSON.stringify(M.{function}(...args)));"
    )
    out = subprocess.run(
        [NODE, "-e", script, os.path.abspath(JS_PATH), json.dumps(args)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)


def nan_to_none(value):
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else value


# --- Style profile ----------------------------------------------------------------------------

def test_trait_profile_matches_python():
    scores = pd.DataFrame({"Lane Bully": [1.0, 3.0, np.nan], "Vision Control": [0.5, np.nan, 2.0]})
    profiles, counts = player_profiles(scores, pd.Series(["me"] * 3))
    games = [{k: nan_to_none(v) for k, v in row.items()} for row in scores.to_dict("records")]
    result = js("traitProfile", games, ["Lane Bully", "Vision Control"])
    for trait in ["Lane Bully", "Vision Control"]:
        assert result["profile"][trait] == pytest.approx(profiles.loc["me", trait])
        assert result["counts"][trait] == counts.loc["me", trait]


def test_trait_profile_all_missing_is_null():
    result = js("traitProfile", [{"Lane Bully": None}], ["Lane Bully"])
    assert result["profile"]["Lane Bully"] is None
    assert result["counts"]["Lane Bully"] == 0


ARCHETYPES = {
    "Safe Farmer": {"Farm-Focused": 1, "Death-Prone": -1},
    "Aggressive Skirmisher": {"Death-Prone": 1, "Duelist": 1},
    "Vision-Focused": {"Vision Control": 1},
}


@pytest.mark.parametrize("profile", [
    {"Farm-Focused": 1.2, "Death-Prone": -0.9, "Duelist": 0.1, "Vision Control": 0.3},
    {"Farm-Focused": -1.0, "Death-Prone": 1.0, "Duelist": -1.0, "Vision Control": 0.0},
    {"Farm-Focused": 0.2, "Death-Prone": None, "Duelist": 0.4, "Vision Control": 1.5},
])
def test_soft_labels_match_python(profile):
    expected = soft_labels(pd.Series(profile, dtype=float), "BOTTOM", {"BOTTOM": ARCHETYPES}, top=3)
    result = js("softLabels", profile, ARCHETYPES, 3)
    assert [name for name, _ in result] == [name for name, _ in expected]
    assert [pct for _, pct in result] == pytest.approx([pct for _, pct in expected])


@pytest.mark.parametrize("r, n", [(0.25, 20), (0.05, 40), (0.6, 3), (0.3, 0)])
def test_confidence_matches_python(r, n):
    reliability = profile_reliability(r, n)
    assert js("profileReliability", r, n) == pytest.approx(reliability)
    assert js("confidenceLabel", reliability) == confidence_label(reliability)


@pytest.mark.parametrize("games", [0, 4, 5, 9, 10, 50])
def test_own_sample_status_matches_python(games):
    assert js("ownSampleStatus", games) == own_sample_status(games)


# --- Champion pool --------------------------------------------------------------------------

@pytest.mark.parametrize("counts", [
    {"Jinx": 27, "Ashe": 2, "Ezreal": 1},
    {"Jinx": 10, "Ashe": 10, "Ezreal": 10},
    {f"c{i}": 1 for i in range(10)},
    {"Jinx": 1},
])
def test_pool_stats_match_python(counts):
    series = pd.Series(counts)
    expected = effective_champions(series)
    result = js("effectiveChampions", counts)
    if math.isnan(expected):
        assert result is None
    else:
        assert result == pytest.approx(expected)
    name, share = top_champion(series)
    assert js("topChampion", counts) == [name, pytest.approx(share)]


def test_class_mix_matches_python():
    # The page receives each game's class already resolved (support overrides applied in
    # Python), so the JS only has to count.
    labels = ["Tank", "Tank", "Pick", "Enchanter"]
    expected = class_mix(pd.Series(labels), "TOP", {"tank": "Tank", "pick": "Pick", "enchanter": "Enchanter"}, {})
    result = js("classMix", labels)
    assert result == pytest.approx(expected.to_dict())


# --- Percentiles and gaps ----------------------------------------------------------------------

def test_percentile_from_breakpoints_close_to_exact():
    # The page only has 51 percentile cutoffs per stat, not every player's average, so
    # its percentiles are interpolated; they should stay within ~2 points of exact.
    from scipy.stats import percentileofscore

    rng = np.random.default_rng(3)
    reference = rng.normal(500, 80, 400)
    breakpoints = np.percentile(reference, np.linspace(0, 100, 51)).tolist()
    for value in [380.0, 450.0, 500.0, 560.0, 640.0]:
        exact = percentileofscore(reference, value, kind="mean")
        assert js("percentileFromBreakpoints", value, breakpoints) == pytest.approx(exact, abs=2.0)


def test_biggest_gaps_match_python():
    # Same fixture as test_performance's gap tests: reference player averages 0..9.
    gap_features = {
        "gold": ["Farming", 1], "cs": ["Farming", 1], "deaths": ["Survival", -1],
        "vision": ["Vision", 1], "kills": ["Combat", 1],
    }
    reference = pd.DataFrame({c: np.arange(10, dtype=float) for c in gap_features})
    mine = pd.DataFrame({
        "gold": [2.0] * 4, "cs": [1.0] * 4, "deaths": [7.0] * 4,
        "vision": [0.0, 6.0, 0.0, 6.0], "kills": [6.0] * 4,
    })
    expected = biggest_gaps(mine, reference, {k: tuple(v) for k, v in gap_features.items()}, top=3)

    # The page gets each stat's reference as percentile cutoffs (median = the 50th).
    curves = {
        c: {"breakpoints": np.percentile(reference[c], np.linspace(0, 100, 51)).tolist(),
            "median": float(reference[c].median())}
        for c in gap_features
    }
    result = js("biggestGaps", mine.to_dict("records"), curves, gap_features, 3)
    assert [g["feature"] for g in result] == [g["feature"] for g in expected]
    assert [g["theme"] for g in result] == [g["theme"] for g in expected]
    for got, want in zip(result, expected):
        assert got["percentile"] == pytest.approx(want["percentile"], abs=6.0)  # coarse grid on 10 players
        assert got["your_avg"] == pytest.approx(want["your_avg"])
        assert got["n_games"] == want["n_games"]


@pytest.mark.parametrize("feature, you, median", [
    ("gold_earned_per_min", 504.0, 515.0),
    ("deaths_per_min", 0.25, 0.20),
    ("kill_participation", 0.55, 0.62),
    ("kda", 3.1, 3.5),
])
def test_gap_size_matches_python(feature, you, median):
    from performance import gap_size
    expected = gap_size(feature, you, median, game_minutes=30.0, percent_features=["kill_participation"])
    result = js("gapSize", feature, you, median, 30.0, ["kill_participation"])
    assert result["amount"] == pytest.approx(expected["amount"])
    assert result["per_game"] == expected["per_game"]


# --- Style map default axes: your two most distinctive traits (furthest from the 50th) ---------------

def test_default_map_axes():
    rows = [
        {"trait": "Lane Bully", "percentile": 73},
        {"trait": "Vision Control", "percentile": 64},
        {"trait": "Death-Prone", "percentile": 17},   # |17 - 50| = 33
        {"trait": "Enchanter", "percentile": 15},     # |15 - 50| = 35
        {"trait": "Damage Carry", "percentile": 48},
    ]
    assert js("defaultMapAxes", rows) == ["Enchanter", "Death-Prone"]


def test_default_map_axes_needs_two_traits():
    assert js("defaultMapAxes", [{"trait": "Lane Bully", "percentile": 73}]) is None
