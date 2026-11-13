"""
Objective participation shares (build_features.py): the share of your team's objectives
you took part in, instead of a per-minute takedown count. A count mostly measures how
many objectives the *team* took (i.e. whether it was winning); the share measures
whether you showed up -- the same idea as kill participation.

- epic_objective_participation = (dragon + Rift Herald + Baron takedowns) /
  the team's (dragon + Herald + Baron) kills. Void grubs aren't included: match-v5 has
  no per-player grub takedown count (only voidMonsterKill, kills alone).
- turret_participation = turretTakedowns / the team's tower kills.
- NaN when the team took none (undefined, not 0%).

Expected values computed by hand from the fixture JSON (info.teams[].objectives and
participants[].challenges), e.g. NA1_5612396602 team 100 took 3 dragons, 1 Herald,
2 Barons (6 epic objectives) and 9 towers.
"""

import math
import os

import pandas as pd
import pytest
from conftest import FIXTURES_DIR

from build_features import build_feature_rows

FIXTURE_PATH = os.path.join(FIXTURES_DIR, "matches.ndjson")


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    return pd.DataFrame(build_feature_rows(FIXTURE_PATH, info_by_puuid={}, min_game_length_seconds=300))


def row(features, match_id, champion):
    match = features[(features["matchId"] == match_id) & (features["championName"] == champion)]
    assert len(match) == 1
    return match.iloc[0]


@pytest.mark.parametrize("champion, epic, turret", [
    ("Udyr", 6 / 6, 4 / 9),     # dragons 3 + Herald 1 + Barons 2 = all 6; 4 of 9 towers
    ("Sivir", 3 / 6, 6 / 9),    # dragons 3, no Herald/Baron
    ("Malphite", 3 / 6, 6 / 9), # dragons 2 + Baron 1
    ("XinZhao", 2 / 2, 0 / 3),  # team 200: 2 dragons, 0 Herald/Baron, 3 towers
    ("Kaisa", 1 / 2, 0 / 3),
])
def test_participation_shares(features, champion, epic, turret):
    r = row(features, "NA1_5612396602", champion)
    assert r["epic_objective_participation"] == pytest.approx(epic)
    assert r["turret_participation"] == pytest.approx(turret)


def test_no_team_objectives_is_nan(features):
    # NA1_5602933445: team 100 (the losing side) took no dragons/Herald/Baron and no towers.
    match = features[features["matchId"] == "NA1_5602933445"]
    losers = match[~match["win"]]
    assert len(losers) == 5
    assert losers["epic_objective_participation"].isna().all()
    assert losers["turret_participation"].isna().all()


def test_shares_are_between_zero_and_one(features):
    for column in ["epic_objective_participation", "turret_participation"]:
        values = features[column].dropna()
        assert ((values >= 0) & (values <= 1)).all(), column


def test_team_total_undercount_is_corrected():
    # Real case (NA1_5611568225): Senna's own turretKills/turretTakedowns = 2, but her
    # team's objectives.tower.kills = 1. Riot's team count is sometimes low; if any
    # teammate took part in more objectives than the team total, the total is wrong,
    # so the largest teammate count is used instead -- shares stay within 0..1.
    from build_features import objective_shares, team_objective_totals

    def player(team, turrets, dragons=0):
        return {"teamId": team, "challenges": {"turretTakedowns": turrets, "dragonTakedowns": dragons}}

    info = {
        "teams": [{"teamId": 200, "objectives": {"tower": {"kills": 1}, "dragon": {"kills": 0}}}],
        "participants": [player(200, 2, dragons=1), player(200, 1)],
    }
    totals = team_objective_totals(info)[200]
    assert totals == {"epic": 1, "tower": 2}
    assert objective_shares(info["participants"][0], totals) == {"epic_objective_participation": 1.0, "turret_participation": 1.0}
    assert objective_shares(info["participants"][1], totals)["turret_participation"] == 0.5
