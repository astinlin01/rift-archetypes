"""
Phase 1b tests: adjust for what's outside the player's control, not for their own impact.

- The adjustment no longer uses the win flag or the full team lead. It uses the
  teammates' lead: your 4 teammates vs the 4 enemies other than your lane opponent.
  Since  team lead = teammates' lead + your own lane lead,  your own contribution stays
  in the features as style.
- New features: team_gold_share (your gold / team gold) and lane_gold_diff_per_min
  (your gold minus your lane opponent's, per minute).
- Redundant kill features are trimmed from the style profile (kept for percentiles).
"""

import json
import os

import numpy as np
import pandas as pd
import pytest
from conftest import FIXTURES_DIR

from build_features import build_feature_rows
from feature_config import FEATURE_COLUMN_NAMES, add_derived_features, add_team_context_features
from playstyle_space import (
    CLUSTER_FEATURE_COLUMNS,
    add_game_state_columns,
    apply_outcome_adjuster,
    fit_outcome_adjuster,
)

FIXTURE_PATH = os.path.join(FIXTURES_DIR, "matches.ndjson")
KAISA_MATCH = "NA1_5612396602"


@pytest.fixture(scope="module")
def fixture_features() -> pd.DataFrame:
    rows = build_feature_rows(FIXTURE_PATH, info_by_puuid={}, min_game_length_seconds=300)
    return add_game_state_columns(add_team_context_features(add_derived_features(pd.DataFrame(rows))))


@pytest.fixture(scope="module")
def kaisa_raw() -> dict:
    """river3's Kai'Sa game straight from the raw match JSON (independent of the pipeline)."""
    with open(FIXTURE_PATH, encoding="utf-8") as infile:
        for line in infile:
            match = json.loads(line)
            if match["metadata"]["matchId"] == KAISA_MATCH:
                return match
    raise AssertionError("fixture match missing")


# --- New features, checked against raw JSON ------------------------------------------

def test_team_gold_share_matches_raw(fixture_features, kaisa_raw):
    parts = kaisa_raw["info"]["participants"]
    me = next(p for p in parts if p["championName"] == "Kaisa")
    team_gold = sum(p["goldEarned"] for p in parts if p["teamId"] == me["teamId"])
    row = fixture_features[(fixture_features["matchId"] == KAISA_MATCH) & (fixture_features["championName"] == "Kaisa")]
    assert row["team_gold_share"].iloc[0] == pytest.approx(me["goldEarned"] / team_gold, rel=1e-6)


def test_lane_gold_diff_matches_raw(fixture_features, kaisa_raw):
    parts = kaisa_raw["info"]["participants"]
    me = next(p for p in parts if p["championName"] == "Kaisa")
    opp = next(p for p in parts if p["teamId"] != me["teamId"] and p["teamPosition"] == "BOTTOM")
    minutes = me["challenges"]["gameLength"] / 60
    row = fixture_features[(fixture_features["matchId"] == KAISA_MATCH) & (fixture_features["championName"] == "Kaisa")]
    assert row["lane_gold_diff_per_min"].iloc[0] == pytest.approx((me["goldEarned"] - opp["goldEarned"]) / minutes, rel=1e-6)


def test_lane_diffs_are_opposite_for_lane_opponents(fixture_features):
    for (match_id, role), pair in fixture_features.groupby(["matchId", "teamPosition"]):
        if len(pair) == 2:
            assert pair["lane_gold_diff_per_min"].sum() == pytest.approx(0.0, abs=1e-9)


def test_team_gold_shares_sum_to_one(fixture_features):
    sums = fixture_features.groupby(["matchId", "win"])["team_gold_share"].sum()
    assert np.allclose(sums, 1.0)


# --- Team lead = teammates' lead + own lane lead ------------------------------------------

@pytest.mark.parametrize("stat", ["gold", "kill"])
def test_team_lead_decomposes(fixture_features, stat):
    team = fixture_features[f"team_{stat}_diff_per_min"]
    teammates = fixture_features[f"teammates_{stat}_diff_per_min"]
    own = fixture_features[f"lane_{stat}_diff_per_min"]
    assert np.allclose(team, teammates + own)


# --- The adjustment keeps your impact, removes your teammates' --------------------------------

FEATURE = "damage_per_minute"


def synthetic_role_df(seed: int = 0, n: int = 4000) -> pd.DataFrame:
    """A 'carry' player creates their own lead and does more damage; teammates' leads also
    inflate everyone's damage. Only the teammate part is outside the player's control."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        carry = i % 10 == 0
        own_lead = rng.normal(60 if carry else 0, 40)       # gold/min vs lane opponent
        teammates_lead = rng.normal(0, 120)                  # gold/min, not this player's doing
        win = (own_lead + teammates_lead + rng.normal(0, 80)) > 0
        row = {
            "championName": "Jinx",
            "puuid": "carry" if carry else f"p{i % 50}",
            "win": bool(win),
            "gameLength": float(rng.uniform(20, 38) * 60),
            "teammates_gold_diff_per_min": teammates_lead,
            "teammates_kill_diff_per_min": teammates_lead / 300,
            "team_gold_diff_per_min": teammates_lead + own_lead,
            "team_kill_diff_per_min": (teammates_lead + own_lead) / 300,
        }
        for column in CLUSTER_FEATURE_COLUMNS:
            row[column] = float(rng.normal(1.0, 0.1))
        row[FEATURE] = 700 + (150 if carry else 0) + 1.0 * teammates_lead + rng.normal(0, 60)
        rows.append(row)
    return pd.DataFrame(rows)


def test_carry_style_survives_adjustment():
    df = synthetic_role_df()
    out = apply_outcome_adjuster(df, fit_outcome_adjuster(df))
    carry = out[df["puuid"] == "carry"][FEATURE].mean()
    others = out[df["puuid"] != "carry"][FEATURE].mean()
    # The carry does +150 damage/min by construction. Adjusting for the win flag (the old
    # method) would shrink this, because the carry also wins more.
    assert carry - others == pytest.approx(150, abs=20)


def test_teammates_lead_is_removed():
    df = synthetic_role_df()
    out = apply_outcome_adjuster(df, fit_outcome_adjuster(df))
    corr = np.corrcoef(out[FEATURE], df["teammates_gold_diff_per_min"])[0, 1]
    assert abs(corr) < 0.03


def test_win_flag_is_not_a_covariate():
    adjuster = fit_outcome_adjuster(synthetic_role_df())
    assert "win" not in adjuster["covariates"]


def test_legacy_artifacts_still_apply():
    # Artifacts saved before this change used win + team-wide leads; they must still load.
    df = synthetic_role_df()
    adjuster = fit_outcome_adjuster(df)
    legacy = dict(adjuster)
    legacy.pop("state_columns")
    from sklearn.linear_model import LinearRegression

    from playstyle_space import LEGACY_STATE_COLUMNS, _row_covariates

    legacy["regression"] = LinearRegression().fit(
        _row_covariates(df, adjuster["length_spline"], LEGACY_STATE_COLUMNS), df[CLUSTER_FEATURE_COLUMNS]
    )
    out = apply_outcome_adjuster(df.head(5), legacy)
    assert out[FEATURE].notna().all()


# --- Kill-feature trim ----------------------------------------------------------------------------

TRIMMED = ["kda", "killing_sprees", "largest_killing_spree", "largest_multikill", "multikills"]


@pytest.mark.parametrize("feature", TRIMMED)
def test_redundant_kill_features_out_of_style_profile(feature):
    assert feature not in CLUSTER_FEATURE_COLUMNS
    assert feature in FEATURE_COLUMN_NAMES  # still available for percentiles


@pytest.mark.parametrize("feature", ["kills_per_min", "solo_kills", "team_gold_share", "lane_gold_diff_per_min"])
def test_kept_and_new_features_in_style_profile(feature):
    assert feature in CLUSTER_FEATURE_COLUMNS
