"""
Phase 5 exporter tests (dashboard_payload.py): what the static page receives.

- Reference curves come from player averages (players with 10+ games in the role), per
  region and tier, and are dropped when a tier has too few players (the page then shows
  "insufficient data (n=X)"). Each curve carries its median for the gap noise check.
- Style: per-game trait scores for the role's traits, plus single-game reliability.
- Champion pool reference: the tier's average class mix and the spread of effective
  champion counts.
- Everything is JSON-safe: NaN becomes null.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest
from conftest import FIXTURES_DIR

from build_features import build_feature_rows
from dashboard_payload import (
    json_number,
    player_average_curves,
    pool_reference,
    reference_by_tier,
    style_scores,
)
from feature_config import add_derived_features
from playstyle_space import ROLE_EXCLUDED_TRAITS, TRAITS, add_game_state_columns


def rows_for(players: dict[str, int], region="NA1", tier="CHALLENGER") -> pd.DataFrame:
    """players: {puuid: games}. One row per game."""
    puuids = [p for p, n in players.items() for _ in range(n)]
    return pd.DataFrame({"puuid": puuids, "region": region, "tier": tier})


# --- Curves -------------------------------------------------------------------------------------

def test_curves_use_player_averages_with_min_games():
    rows = rows_for({"a": 10, "b": 10, "c": 3})
    values = pd.DataFrame({"x": [1.0] * 10 + [3.0] * 10 + [100.0] * 3})
    curves = player_average_curves(rows, values, min_games=10, min_players=2)
    assert curves["x"]["n"] == 2                      # c (3 games) left out
    assert curves["x"]["ref_avg"] == pytest.approx(2.0)
    assert curves["x"]["median"] == pytest.approx(2.0)
    assert curves["x"]["breakpoints"][0] == pytest.approx(1.0)
    assert curves["x"]["breakpoints"][-1] == pytest.approx(3.0)
    assert len(curves["x"]["breakpoints"]) == 51


def test_curves_skip_missing_values_per_player():
    # A player's average ignores games where the stat is missing (e.g. no timeline).
    rows = rows_for({"a": 10, "b": 10})
    values = pd.DataFrame({"x": [2.0] * 5 + [np.nan] * 5 + [4.0] * 10})
    curves = player_average_curves(rows, values, min_games=10, min_players=2)
    assert curves["x"]["ref_avg"] == pytest.approx(3.0)


def test_too_few_players_gives_no_curves():
    rows = rows_for({"a": 10})
    values = pd.DataFrame({"x": [1.0] * 10})
    assert player_average_curves(rows, values, min_games=10, min_players=2) == {}


def test_reference_by_tier_keys_and_insufficient_tiers():
    rows = pd.concat([
        rows_for({"a": 10, "b": 10}, tier="CHALLENGER"),
        rows_for({"c": 10}, tier="DIAMOND"),             # 1 player: below the minimum
    ], ignore_index=True)
    values = pd.DataFrame({"x": np.arange(len(rows), dtype=float)})
    reference = reference_by_tier(rows, values, ["NA1"], ["CHALLENGER", "DIAMOND"], min_games=10, min_players=2)
    assert set(reference["NA1"]) == {"CHALLENGER"}
    # The page needs n for "insufficient data (n=X)", so tiers that fall short report their size.
    assert reference["_player_counts"]["NA1"]["DIAMOND"] == 1


# --- Champion pool reference ----------------------------------------------------------------------

def test_pool_reference():
    rows = rows_for({"a": 10, "b": 10})
    rows["championName"] = ["Malphite"] * 10 + ["Jinx"] * 5 + ["Ashe"] * 5
    rows["champion_class"] = ["Tank"] * 10 + ["Marksman"] * 10
    pool = pool_reference(rows, min_games=10, min_players=2)
    assert pool["class_mix"] == pytest.approx({"Tank": 0.5, "Marksman": 0.5})
    # a: 1 champion; b: 5/5 -> unbiased HHI 40/90 -> 2.25, capped at 2 champions played.
    assert pool["effective_champions"]["breakpoints"][0] == pytest.approx(1.0)
    assert pool["effective_champions"]["breakpoints"][-1] == pytest.approx(2.0)
    assert pool["n"] == 2


# --- Style scores on real fixture games --------------------------------------------------------------

@pytest.fixture(scope="module")
def fixture_df():
    path = os.path.join(FIXTURES_DIR, "matches.ndjson")
    rows = build_feature_rows(path, info_by_puuid={}, min_game_length_seconds=300)
    return add_game_state_columns(add_derived_features(pd.DataFrame(rows)))


def test_style_scores_shape(fixture_df):
    role_df = fixture_df[fixture_df["teamPosition"] == "BOTTOM"].reset_index(drop=True)
    scores, icc, traits = style_scores(role_df, "BOTTOM", champion_classes={}, min_reliability_games=2)
    expected = [t for t, _ in TRAITS if t not in ROLE_EXCLUDED_TRAITS["BOTTOM"]]
    assert traits == expected
    assert list(scores.columns) == expected
    assert scores.index.equals(role_df.index)
    assert set(icc.index) == set(expected)


def test_style_scores_without_timelines_use_end_of_game_features(fixture_df):
    # No timeline columns in the table: traits that also have end-of-game features still
    # get scores; Early Trading (timeline-only) is all missing.
    role_df = fixture_df[fixture_df["teamPosition"] == "MIDDLE"].reset_index(drop=True)
    scores, _, _ = style_scores(role_df, "MIDDLE", champion_classes={}, min_reliability_games=2)
    assert scores["Lane Bully"].notna().all()
    assert scores["Early Trading"].isna().all()


# --- JSON safety ------------------------------------------------------------------------------------------

def test_json_number():
    assert json_number(float("nan")) is None
    assert json_number(None) is None
    assert json_number(np.float64(1.234567)) == 1.2346
    assert json.dumps(json_number(np.int64(3))) == "3"
