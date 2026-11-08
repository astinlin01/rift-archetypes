"""
Accuracy tests for per-participant feature extraction (build_features.py).

Fixtures: tests/fixtures/matches.ndjson holds 5 real ranked games pulled from
data/raw/matches.ndjson, chosen to cover different cases:

  NA1_5634685335  river3 JUNGLE Cho'Gath, win,  S16, 39.8 min  -> jungle CS, long game
  NA1_5612396602  river3 BOTTOM Kai'Sa,   loss, S16, 32.0 min  -> lane CS, loss
  NA1_5368209459  river3 UTILITY Braum,   loss, S15, 16.1 min  -> short game, prior season
  NA1_5201754928  river3 BOTTOM Jinx,     win,  S14, 22.0 min  -> oldest season
  NA1_5603373894  remake, 2.2 min                              -> must produce zero rows

Expected values below were computed by hand from the raw match JSON fields
(totalMinionsKilled, neutralMinionsKilled, challenges.gameLength in seconds),
NOT from the pipeline's own code, so the tests can catch pipeline bugs.
"""

import os

import pandas as pd
import pytest
from conftest import FIXTURES_DIR

from build_features import build_feature_rows
from feature_config import add_derived_features, season_from_patch

FIXTURE_PATH = os.path.join(FIXTURES_DIR, "matches.ndjson")
MIN_GAME_LENGTH_SECONDS = 300  # pipeline default: drop games under 5 minutes


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    rows = build_feature_rows(FIXTURE_PATH, info_by_puuid={}, min_game_length_seconds=MIN_GAME_LENGTH_SECONDS)
    return add_derived_features(pd.DataFrame(rows))


def my_row(features: pd.DataFrame, match_id: str) -> pd.Series:
    """river3's row in a given match, identified by champion + role in that match."""
    match = features[features["matchId"] == match_id]
    mine = {
        "NA1_5634685335": ("Chogath", "JUNGLE"),
        "NA1_5612396602": ("Kaisa", "BOTTOM"),
        "NA1_5368209459": ("Braum", "UTILITY"),
        "NA1_5201754928": ("Jinx", "BOTTOM"),
    }[match_id]
    row = match[(match["championName"] == mine[0]) & (match["teamPosition"] == mine[1])]
    assert len(row) == 1, f"expected exactly one {mine} row in {match_id}, got {len(row)}"
    return row.iloc[0]


# --- CS: includes jungle camps -------------------------------------------------

def test_jungle_cs_includes_camps(features):
    # Cho'Gath: 54 minions + 247 neutral = 301 CS over 2390.530 s (39.842 min)
    # 301 / 39.842 = 7.555 CS/min. Minions alone would be 1.355 -- the old bug.
    row = my_row(features, "NA1_5634685335")
    assert row["cs_total_per_min"] == pytest.approx(7.5548, rel=1e-3)
    assert row["cs_per_min"] == pytest.approx(1.3553, rel=1e-3)
    assert row["neutral_cs_per_min"] == pytest.approx(6.1995, rel=1e-3)


def test_laner_cs_total_equals_minions_when_no_camps(features):
    # Kai'Sa: 325 minions + 0 neutral over 1922.928 s (32.049 min) -> 10.141 CS/min
    row = my_row(features, "NA1_5612396602")
    assert row["cs_total_per_min"] == pytest.approx(10.1408, rel=1e-3)
    assert row["cs_total_per_min"] == pytest.approx(row["cs_per_min"])


# --- Per-minute normalization uses real game length ----------------------------

def test_short_game_per_minute_normalization(features):
    # Braum: 16.104 min game. 18 minions -> 1.1177/min, 5 deaths -> 0.3105/min.
    # Short games are where a wrong denominator would distort rates the most.
    row = my_row(features, "NA1_5368209459")
    assert row["cs_total_per_min"] == pytest.approx(1.1177, rel=1e-3)
    assert row["deaths_per_min"] == pytest.approx(0.3105, rel=1e-3)


def test_game_length_is_seconds(features):
    row = my_row(features, "NA1_5201754928")
    assert row["gameLength"] == pytest.approx(1319.873, rel=1e-6)
    # Jinx: 190 minions / 21.998 min = 8.637 CS/min
    assert row["cs_total_per_min"] == pytest.approx(8.6372, rel=1e-3)


# --- Remakes are dropped ---------------------------------------------------------

def test_remake_produces_no_rows(features):
    assert (features["matchId"] == "NA1_5603373894").sum() == 0


def test_every_kept_game_meets_minimum_length(features):
    assert (features["gameLength"] >= MIN_GAME_LENGTH_SECONDS).all()


# --- Season tagging ----------------------------------------------------------------

@pytest.mark.parametrize(
    "match_id, expected_season",
    [
        ("NA1_5634685335", 16),
        ("NA1_5612396602", 16),
        ("NA1_5368209459", 15),
        ("NA1_5201754928", 14),
    ],
)
def test_season_tagged_from_game_version(features, match_id, expected_season):
    assert my_row(features, match_id)["season"] == expected_season


@pytest.mark.parametrize(
    "patch, expected",
    [("16.17", 16), ("14.24", 14), (None, None), ("", None), ("garbage", None)],
)
def test_season_from_patch(patch, expected):
    assert season_from_patch(patch) == expected
