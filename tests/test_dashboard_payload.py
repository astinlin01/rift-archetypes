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


# --- Style map: one point per reference player ------------------------------------------------------
# The map plots every qualifying reference player (10+ games) as their average trait scores,
# in the role's trait order; the page turns them into percentiles with the tier's curves.

from dashboard_payload import crop_splash, player_trait_points, splash_champions


def test_player_trait_points_are_player_averages():
    rows = rows_for({"a": 10, "b": 12, "few": 3})
    # a: Lane Bully all 1.0, Duelist 0.0/2.0 alternating -> 1.0; b: 2.0 and 4.0; few: excluded
    scores = pd.DataFrame({
        "Lane Bully": [1.0] * 10 + [2.0] * 12 + [9.0] * 3,
        "Duelist": [0.0, 2.0] * 5 + [4.0] * 12 + [9.0] * 3,
    })
    points = player_trait_points(rows, scores, ["Lane Bully", "Duelist"], ["NA1"], ["CHALLENGER"], min_players=2)
    assert sorted(points["NA1"]["CHALLENGER"]) == [[1.0, 1.0], [2.0, 4.0]]


def test_player_trait_points_missing_is_null_and_small_tiers_dropped():
    rows = rows_for({"a": 10, "b": 10})
    scores = pd.DataFrame({"Lane Bully": [np.nan] * 10 + [0.12345] * 10})
    points = player_trait_points(rows, scores, ["Lane Bully"], ["NA1"], ["CHALLENGER"], min_players=2)
    assert sorted(points["NA1"]["CHALLENGER"], key=str) == [[0.123], [None]]  # 3 decimals
    assert player_trait_points(rows, scores, ["Lane Bully"], ["NA1"], ["CHALLENGER"], min_players=3) == {}


# --- Player card splash art -----------------------------------------------------------------------

def test_splash_champions_top_pick_per_account_and_role():
    games = pd.DataFrame({
        "account": ["main"] * 9 + ["alt"] * 4,
        "teamPosition": ["BOTTOM"] * 6 + ["TOP"] * 3 + ["MIDDLE"] * 4,
        "championName": ["Kaisa"] * 4 + ["Ezreal"] * 2 + ["Gnar"] * 3 + ["Ahri"] * 4,
    })
    # main BOTTOM -> Kaisa (6 games >= 5). main TOP has 3 games and alt MIDDLE has 4: too few
    # for a role card, but each account's overall top pick is kept for the all-roles view.
    assert splash_champions(games, min_games=5) == {"Kaisa", "Ahri"}


def test_crop_splash_shape_and_size():
    from io import BytesIO

    from PIL import Image
    source = BytesIO()
    Image.effect_noise((1215, 717), 64).convert("RGB").save(source, "JPEG", quality=95)  # splash size
    out = crop_splash(source.getvalue(), width=1200, height=480)
    image = Image.open(BytesIO(out))
    assert image.format == "JPEG" and image.size == (1200, 480)
    assert len(out) < len(source.getvalue())


# --- Enchanter only where it can happen ----------------------------------------------------------
# Heals/shields on teammates are zero in 86-94% of TOP/MIDDLE/BOTTOM games (82-93% of games are
# on champions that essentially never do it; checked 2026-11-13, season 16). There the
# champion-adjusted score mostly reflects which champions you play, so the trait is dropped,
# as it already was for JUNGLE.

from dashboard_payload import role_traits
from playstyle_space import ROLE_ARCHETYPES
from style_profile import soft_labels


@pytest.mark.parametrize("role", ["TOP", "JUNGLE", "MIDDLE", "BOTTOM"])
def test_enchanter_only_for_support(role):
    assert "Enchanter" not in [t for t, _ in role_traits(role)]


def test_enchanter_kept_for_support():
    assert "Enchanter" in [t for t, _ in role_traits("UTILITY")]


def test_archetypes_only_use_traits_their_role_has():
    for role, archetypes in ROLE_ARCHETYPES.items():
        available = {t for t, _ in role_traits(role)}
        for name, signature in archetypes.items():
            assert set(signature) <= available, f"{role} {name} uses {set(signature) - available}"


def test_utility_mage_still_matches_on_cc_and_teamfighting():
    traits = [t for t, _ in role_traits("BOTTOM")]
    profile = pd.Series(0.0, index=traits)
    profile[["Crowd Control", "Teamfighter"]] = [1.0, 0.5]
    best = soft_labels(profile, "BOTTOM", ROLE_ARCHETYPES, top=1)[0]
    assert best[0] == "Utility Mage" and best[1] > 90
