"""
Early-game (first-batch) timeline feature tests: timeline_features.py.

Fixtures: tests/fixtures/timelines.ndjson holds the first 16 minutes of three of river3's
games (also in matches.ndjson). Expected values were read by hand from the raw timeline
frames/events -- frame i is the snapshot at minute i; "before 14" means event timestamp
< 14:00 -- NOT computed with the module under test.

  Cho'Gath JUNGLE (pid 2) vs Jarvan IV (pid 7)   NA1_5634685335
  Kai'Sa  BOTTOM  (pid 9) vs Sivir     (pid 4)   NA1_5612396602
  Braum   UTILITY (pid 10) vs Lulu     (pid 5)   NA1_5368209459

Synthetic timelines cover what these games don't: solo kills, executes, short games
and teams with no early kills or objectives.
"""

import gzip
import json
import math
import os

import pandas as pd
import pytest
from conftest import FIXTURES_DIR

from timeline_features import (
    TIMELINE_FEATURE_COLUMNS,
    extract_timeline_features,
    read_timeline,
    select_timeline_match_ids,
    write_timeline,
)


def load_fixtures():
    with open(os.path.join(FIXTURES_DIR, "matches.ndjson"), encoding="utf-8") as infile:
        matches = {m["metadata"]["matchId"]: m for m in map(json.loads, infile)}
    with open(os.path.join(FIXTURES_DIR, "timelines.ndjson"), encoding="utf-8") as infile:
        timelines = {t["metadata"]["matchId"]: t for t in map(json.loads, infile)}
    return matches, timelines


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    matches, timelines = load_fixtures()
    rows = []
    for match_id, timeline in timelines.items():
        rows.extend(extract_timeline_features(timeline, matches[match_id]))
    return pd.DataFrame(rows)


def row_for(features, match_id, participant_id):
    row = features[(features["matchId"] == match_id) & (features["participantId"] == participant_id)]
    assert len(row) == 1
    return row.iloc[0]


# --- Hand-checked values for river3 ---------------------------------------------------------

EXPECTED = {
    # Cho'Gath: CS@10 (0+78)-(5+68)=5; gold@14 5957-6339=-382; xp@14 6420-6207=213;
    # K+A before 14 = 3+1 of 14 team kills; took all 3 of the team's early objectives (grubs).
    ("NA1_5634685335", 2): dict(
        cs_diff_at_10=5, gold_diff_at_14=-382, xp_diff_at_14=213, early_kill_participation=4 / 14,
        early_deaths=4, early_solo_kills=0, damage_to_champions_at_14=4196, damage_taken_at_14=10993,
        early_objective_participation=1.0, early_wards_placed=0, early_wards_killed=2, early_control_wards=0,
    ),
    # Kai'Sa: CS@10 81-90=-9; gold@14 6514-4887=1627; xp@14 5236-5009=227; K+A 4+4 of 9;
    # team took 1 early objective (dragon) without her.
    ("NA1_5612396602", 9): dict(
        cs_diff_at_10=-9, gold_diff_at_14=1627, xp_diff_at_14=227, early_kill_participation=8 / 9,
        early_deaths=1, early_solo_kills=0, damage_to_champions_at_14=5808, damage_taken_at_14=6055,
        early_objective_participation=0.0, early_wards_placed=5, early_wards_killed=0, early_control_wards=0,
    ),
    # Braum: CS@10 14-7=7; gold@14 3797-4344=-547; xp@14 3654-4801=-1147; K+A 1+2 of 7;
    # team took 3 grubs without him; 9 wards incl. 1 control ward, 3 wards killed.
    ("NA1_5368209459", 10): dict(
        cs_diff_at_10=7, gold_diff_at_14=-547, xp_diff_at_14=-1147, early_kill_participation=3 / 7,
        early_deaths=3, early_solo_kills=0, damage_to_champions_at_14=3029, damage_taken_at_14=9978,
        early_objective_participation=0.0, early_wards_placed=9, early_wards_killed=3, early_control_wards=1,
    ),
    # --- More bot-lane coverage + other roles (NaN = undefined: team took no early objectives) ---
    # Ezreal BOTTOM (9) vs Syndra (4), won the game despite a rough early game: 6 early deaths.
    # CS@10 78-70=8; gold@14 6121-7302=-1181; xp@14 5359-5621=-262; K4+A3 of 12.
    ("NA1_5607421247", 9): dict(
        cs_diff_at_10=8, gold_diff_at_14=-1181, xp_diff_at_14=-262, early_kill_participation=7 / 12,
        early_deaths=6, early_solo_kills=0, damage_to_champions_at_14=10878, damage_taken_at_14=9862,
        early_objective_participation=0.0, early_wards_placed=5, early_wards_killed=0, early_control_wards=0,
    ),
    # Miss Fortune BOTTOM (4) vs Ezreal (9), lane stomp: CS@10 83-70=13; gold@14 7443-5265=2178;
    # xp@14 5911-4860=1051; K5+A3 of 14; 1 early solo kill.
    ("NA1_5607484806", 4): dict(
        cs_diff_at_10=13, gold_diff_at_14=2178, xp_diff_at_14=1051, early_kill_participation=8 / 14,
        early_deaths=1, early_solo_kills=1, damage_to_champions_at_14=9282, damage_taken_at_14=7257,
        early_objective_participation=0.0, early_wards_placed=6, early_wards_killed=1, early_control_wards=0,
    ),
    # Miss Fortune BOTTOM (9) vs Kai'Sa (4), stomped in lane but won fights: 3 early solo kills
    # while CS@10 62-82=-20 and gold@14 5831-8214=-2383; xp@14 5434-5496=-62; K4+A0 of 10.
    ("NA1_5672457959", 9): dict(
        cs_diff_at_10=-20, gold_diff_at_14=-2383, xp_diff_at_14=-62, early_kill_participation=4 / 10,
        early_deaths=6, early_solo_kills=3, damage_to_champions_at_14=7453, damage_taken_at_14=10410,
        early_objective_participation=math.nan, early_wards_placed=6, early_wards_killed=0, early_control_wards=0,
    ),
    # Ezreal BOTTOM (9) vs Seraphine (4), 15.1-minute surrender: the 14:00 snapshot still exists.
    # CS@10 83-54=29; gold@14 5761-5380=381; xp@14 4268-4987=-719; K2+A0 of 10.
    ("NA1_5595155144", 9): dict(
        cs_diff_at_10=29, gold_diff_at_14=381, xp_diff_at_14=-719, early_kill_participation=2 / 10,
        early_deaths=4, early_solo_kills=0, damage_to_champions_at_14=8603, damage_taken_at_14=6628,
        early_objective_participation=0.0, early_wards_placed=4, early_wards_killed=0, early_control_wards=0,
    ),
    # Viktor MIDDLE (8) vs Fizz (3): CS@10 69-60=9; gold@14 6050-4062=1988; xp@14 7081-5594=1487;
    # K4+A3 of 10, 2 early solo kills, 0 deaths.
    ("NA1_5612376675", 8): dict(
        cs_diff_at_10=9, gold_diff_at_14=1988, xp_diff_at_14=1487, early_kill_participation=7 / 10,
        early_deaths=0, early_solo_kills=2, damage_to_champions_at_14=9356, damage_taken_at_14=6856,
        early_objective_participation=math.nan, early_wards_placed=5, early_wards_killed=0, early_control_wards=0,
    ),
    # Gnar TOP (1) vs Vladimir (6): CS@10 65-68=-3; gold@14 4255-6499=-2244; xp@14 6298-8081=-1783;
    # K1+A2 of 8.
    ("NA1_5602933445", 1): dict(
        cs_diff_at_10=-3, gold_diff_at_14=-2244, xp_diff_at_14=-1783, early_kill_participation=3 / 8,
        early_deaths=4, early_solo_kills=0, damage_to_champions_at_14=8428, damage_taken_at_14=10574,
        early_objective_participation=math.nan, early_wards_placed=5, early_wards_killed=0, early_control_wards=0,
    ),
}


@pytest.mark.parametrize("key", list(EXPECTED))
def test_hand_checked_values(features, key):
    row = row_for(features, *key)
    for column, expected in EXPECTED[key].items():
        if isinstance(expected, float) and math.isnan(expected):
            assert math.isnan(row[column]), column
        else:
            assert row[column] == pytest.approx(expected, rel=1e-6), column


def test_one_row_per_participant_with_all_features(features):
    assert len(features) == 10 * features["matchId"].nunique()
    assert set(TIMELINE_FEATURE_COLUMNS) <= set(features.columns)
    assert features.groupby("matchId")["participantId"].nunique().eq(10).all()


@pytest.mark.parametrize("column", ["cs_diff_at_10", "gold_diff_at_14", "xp_diff_at_14"])
def test_lane_opponent_diffs_are_opposite(features, column):
    for _, pair in features.groupby(["matchId", "teamPosition"]):
        assert pair[column].sum() == pytest.approx(0.0)


def test_rows_carry_puuid_for_joining(features):
    matches, _ = load_fixtures()
    me = next(p for p in matches["NA1_5612396602"]["info"]["participants"] if p["participantId"] == 9)
    assert row_for(features, "NA1_5612396602", 9)["puuid"] == me["puuid"]


# --- Synthetic edge cases ----------------------------------------------------------------------

POSITIONS = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]


def synthetic(events: list[dict], minutes: int = 16):
    """Minimal match + timeline: pids 1-5 team 100, 6-10 team 200, same lane order."""
    participants = [
        {"participantId": pid, "puuid": f"puuid{pid}", "teamId": 100 if pid <= 5 else 200,
         "teamPosition": POSITIONS[(pid - 1) % 5]}
        for pid in range(1, 11)
    ]
    frame_stats = {
        "totalGold": 500, "xp": 0, "minionsKilled": 0, "jungleMinionsKilled": 0,
        "damageStats": {"totalDamageDoneToChampions": 0, "totalDamageTaken": 0},
    }
    frames = [
        {"timestamp": i * 60000, "events": [e for e in events if (i - 1) * 60000 < e["timestamp"] <= i * 60000],
         "participantFrames": {str(pid): dict(frame_stats) for pid in range(1, 11)}}
        for i in range(minutes + 1)
    ]
    match = {"metadata": {"matchId": "SYN_1"}, "info": {"participants": participants}}
    timeline = {"metadata": {"matchId": "SYN_1"},
                "info": {"frames": frames, "participants": [{"participantId": p["participantId"], "puuid": p["puuid"]} for p in participants]}}
    return timeline, match


def synthetic_row(events, minutes=16, pid=3):
    timeline, match = synthetic(events, minutes)
    return next(r for r in extract_timeline_features(timeline, match) if r["participantId"] == pid)


def kill(ts_min, killer, victim, assists=()):
    return {"type": "CHAMPION_KILL", "timestamp": int(ts_min * 60000), "killerId": killer,
            "victimId": victim, "assistingParticipantIds": list(assists)}


def test_solo_kill_counted_assisted_kill_not():
    row = synthetic_row([kill(5, 3, 8), kill(7, 3, 8, assists=[2])])
    assert row["early_solo_kills"] == 1


def test_execute_is_a_death_but_nobodys_kill():
    # Killed by a tower/minion: killerId 0. Counts as the victim's death, not as a team kill.
    row = synthetic_row([kill(6, 0, 3)])
    assert row["early_deaths"] == 1
    assert math.isnan(row["early_kill_participation"])  # team has no kills: undefined, not 0


def test_events_after_14_ignored():
    row = synthetic_row([kill(13.9, 3, 8), kill(14.1, 3, 8)])
    assert row["early_solo_kills"] == 1


def test_undefined_ward_type_not_counted():
    # Riot's "UNDEFINED" ward type covers non-ward objects (e.g. Teemo mushrooms).
    events = [
        {"type": "WARD_PLACED", "timestamp": 300000, "creatorId": 3, "wardType": "UNDEFINED"},
        {"type": "WARD_PLACED", "timestamp": 310000, "creatorId": 3, "wardType": "CONTROL_WARD"},
    ]
    row = synthetic_row(events)
    assert row["early_wards_placed"] == 1
    assert row["early_control_wards"] == 1


def test_no_team_objectives_is_undefined_not_zero():
    assert math.isnan(synthetic_row([])["early_objective_participation"])


def test_short_game_snapshot_features_are_nan():
    # A game that ended before minute 14 has no @14 snapshot.
    row = synthetic_row([], minutes=12)
    assert math.isnan(row["gold_diff_at_14"])
    assert not math.isnan(row["cs_diff_at_10"])


# --- Storage --------------------------------------------------------------------------------------

def test_write_read_roundtrip_is_gzipped(tmp_path):
    _, timelines = load_fixtures()
    timeline = timelines["NA1_5612396602"]
    path = write_timeline(timeline, str(tmp_path))
    assert path.endswith(".json.gz")
    with gzip.open(path, "rt", encoding="utf-8") as infile:
        json.load(infile)  # really gzip
    assert read_timeline("NA1_5612396602", str(tmp_path)) == timeline
    assert os.path.getsize(path) < len(json.dumps(timeline)) / 5


# --- Subset selection ----------------------------------------------------------------------------

def test_subset_is_my_games_plus_regulars():
    df = pd.DataFrame({
        "matchId": ["m1", "m2", "m3"] + [f"r{i}" for i in range(10)] + [f"x{i}" for i in range(9)],
        "puuid": ["me", "me", "me"] + ["regular"] * 10 + ["occasional"] * 9,
        "teamPosition": ["MIDDLE"] * 22,
        "season": [16] * 22,
    })
    selected = select_timeline_match_ids(df, my_puuids=["me"], min_player_games=10)
    assert selected == sorted(["m1", "m2", "m3"] + [f"r{i}" for i in range(10)])
