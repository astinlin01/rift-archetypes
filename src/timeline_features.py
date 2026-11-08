"""
Early-game features from match-v5 timelines (first batch).

Why early game: end-of-game stats are heavily shaped by how the game snowballed; what a
player does in the first ~14 minutes (before turret plates fall and games diverge) is
mostly cause rather than effect, so it's a cleaner playstyle signal.

Timelines are a per-minute snapshot of every participant ("frames": gold, XP, CS,
damage, position) plus timestamped events (kills, wards, objectives, ...). Frame i is
the snapshot at minute i; the events stored in frame i happened during minute i.

Features (per participant, vs their lane opponent = same teamPosition on the other team):
- cs_diff_at_10          CS (minions + jungle camps) lead at 10:00
- gold_diff_at_14        total gold lead at 14:00
- xp_diff_at_14          XP lead at 14:00
- early_kill_participation  (kills + assists) / team kills before 14:00; NaN if the team had none
- early_deaths           deaths before 14:00 (executes included)
- early_solo_kills       kills with no assisting champion before 14:00
- damage_to_champions_at_14, damage_taken_at_14   cumulative at 14:00 (trading/aggression)
- early_objective_participation  share of the team's dragons / Herald / void grubs before
                         14:00 the player took part in; NaN if the team took none
- early_wards_placed, early_wards_killed, early_control_wards   before 14:00

Snapshot features are NaN when the game ended before that minute, or when the lane
opponent is ambiguous (e.g. a lane swap tagged two players the same role).

Raw timelines are stored gzipped, one file per match (data/raw/timelines/<matchId>.json.gz,
~35-40 KB each vs ~0.4-1.3 MB raw), so new timeline features can be re-extracted later
without refetching.
"""

from __future__ import annotations

import gzip
import json
import os
from collections import Counter

import pandas as pd

DEFAULT_TIMELINE_DIR = "data/raw/timelines"
EARLY_CUTOFF_MS = 14 * 60_000
OBJECTIVE_MONSTERS = frozenset({"DRAGON", "RIFTHERALD", "HORDE"})  # HORDE = void grubs
# Riot's "UNDEFINED" ward type covers non-ward objects (e.g. Teemo mushrooms).
NON_WARD_TYPES = frozenset({"UNDEFINED"})

TIMELINE_FEATURE_COLUMNS = [
    "cs_diff_at_10",
    "gold_diff_at_14",
    "xp_diff_at_14",
    "early_kill_participation",
    "early_deaths",
    "early_solo_kills",
    "damage_to_champions_at_14",
    "damage_taken_at_14",
    "early_objective_participation",
    "early_wards_placed",
    "early_wards_killed",
    "early_control_wards",
]

NAN = float("nan")


# ---------- Storage ----------

def timeline_path(match_id: str, timeline_dir: str = DEFAULT_TIMELINE_DIR) -> str:
    return os.path.join(timeline_dir, f"{match_id}.json.gz")


def write_timeline(timeline: dict, timeline_dir: str = DEFAULT_TIMELINE_DIR) -> str:
    os.makedirs(timeline_dir, exist_ok=True)
    path = timeline_path(timeline["metadata"]["matchId"], timeline_dir)
    tmp_path = path + ".tmp"
    with gzip.open(tmp_path, "wt", encoding="utf-8", compresslevel=6) as outfile:
        json.dump(timeline, outfile, separators=(",", ":"))
    os.replace(tmp_path, path)  # atomic: an interrupted fetch never leaves a half-written file
    return path


def read_timeline(match_id: str, timeline_dir: str = DEFAULT_TIMELINE_DIR) -> dict:
    with gzip.open(timeline_path(match_id, timeline_dir), "rt", encoding="utf-8") as infile:
        return json.load(infile)


# ---------- Subset selection ----------

def select_timeline_match_ids(df: pd.DataFrame, my_puuids: list[str], min_player_games: int = 10) -> list[str]:
    """Matches to fetch timelines for: all of my games, plus every game of each player with
    >= min_player_games in a role (the players that make up the reference averages).
    Filter df to the season(s) of interest before calling."""
    counts = df.groupby(["puuid", "teamPosition"]).size()
    regulars = counts[counts >= min_player_games].reset_index()[["puuid", "teamPosition"]]
    regular_rows = df.merge(regulars, on=["puuid", "teamPosition"])
    mine = df[df["puuid"].isin(my_puuids)]
    return sorted(set(regular_rows["matchId"]) | set(mine["matchId"]))


# ---------- Feature extraction ----------

def _frame_at(frames: list[dict], minute: int) -> dict | None:
    """The snapshot at `minute`, or None if the game ended before it (a game's last frame
    is stamped at its end time, so check the timestamp, not just the index)."""
    if len(frames) <= minute or frames[minute].get("timestamp", 0) < minute * 60_000:
        return None
    return frames[minute]


def _lane_opponents(participants: list[dict]) -> dict[int, int]:
    """participantId -> lane opponent's participantId, where the pairing is unambiguous."""
    by_side_role = Counter((p["teamId"], p.get("teamPosition")) for p in participants)
    opponents = {}
    for p in participants:
        role = p.get("teamPosition")
        if not role or by_side_role[(p["teamId"], role)] != 1:
            continue
        others = [q for q in participants if q["teamId"] != p["teamId"] and q.get("teamPosition") == role]
        if len(others) == 1:
            opponents[p["participantId"]] = others[0]["participantId"]
    return opponents


def _snapshot_diff(frame: dict | None, pid: int, opp: int | None, value) -> float:
    if frame is None or opp is None:
        return NAN
    mine, theirs = frame["participantFrames"][str(pid)], frame["participantFrames"][str(opp)]
    return float(value(mine) - value(theirs))


def _snapshot(frame: dict | None, pid: int, value) -> float:
    return NAN if frame is None else float(value(frame["participantFrames"][str(pid)]))


def extract_timeline_features(timeline: dict, match: dict) -> list[dict]:
    """One row per participant: identifiers + TIMELINE_FEATURE_COLUMNS."""
    match_id = match["metadata"]["matchId"]
    participants = match["info"]["participants"]
    team_of = {p["participantId"]: p["teamId"] for p in participants}
    opponents = _lane_opponents(participants)

    frames = timeline["info"]["frames"]
    at_10, at_14 = _frame_at(frames, 10), _frame_at(frames, 14)
    early = [e for frame in frames for e in frame.get("events", []) if e.get("timestamp", 0) < EARLY_CUTOFF_MS]

    kills = [e for e in early if e.get("type") == "CHAMPION_KILL"]
    # killerId 0 = executed by a tower/minion/monster: a death, but nobody's kill.
    team_kills = Counter(team_of[e["killerId"]] for e in kills if e.get("killerId") in team_of)

    objectives = [e for e in early if e.get("type") == "ELITE_MONSTER_KILL" and e.get("monsterType") in OBJECTIVE_MONSTERS]
    team_objectives = Counter(e.get("killerTeamId") for e in objectives)

    def involved(e: dict, pid: int) -> bool:
        return e.get("killerId") == pid or pid in (e.get("assistingParticipantIds") or [])

    rows = []
    for p in participants:
        pid, team = p["participantId"], p["teamId"]
        opp = opponents.get(pid)
        my_kills = [e for e in kills if e.get("killerId") == pid]
        my_assists = [e for e in kills if pid in (e.get("assistingParticipantIds") or [])]
        my_wards = [
            e for e in early
            if e.get("type") == "WARD_PLACED" and e.get("creatorId") == pid and e.get("wardType") not in NON_WARD_TYPES
        ]
        rows.append({
            "matchId": match_id,
            "participantId": pid,
            "puuid": p.get("puuid"),
            "teamPosition": p.get("teamPosition"),
            "cs_diff_at_10": _snapshot_diff(at_10, pid, opp, lambda f: f["minionsKilled"] + f["jungleMinionsKilled"]),
            "gold_diff_at_14": _snapshot_diff(at_14, pid, opp, lambda f: f["totalGold"]),
            "xp_diff_at_14": _snapshot_diff(at_14, pid, opp, lambda f: f["xp"]),
            "early_kill_participation": (
                (len(my_kills) + len(my_assists)) / team_kills[team] if team_kills[team] else NAN
            ),
            "early_deaths": sum(1 for e in kills if e.get("victimId") == pid),
            "early_solo_kills": sum(1 for e in my_kills if not e.get("assistingParticipantIds")),
            "damage_to_champions_at_14": _snapshot(at_14, pid, lambda f: f["damageStats"]["totalDamageDoneToChampions"]),
            "damage_taken_at_14": _snapshot(at_14, pid, lambda f: f["damageStats"]["totalDamageTaken"]),
            "early_objective_participation": (
                sum(1 for e in objectives if e.get("killerTeamId") == team and involved(e, pid)) / team_objectives[team]
                if team_objectives[team] else NAN
            ),
            "early_wards_placed": len(my_wards),
            "early_wards_killed": sum(1 for e in early if e.get("type") == "WARD_KILL" and e.get("killerId") == pid),
            "early_control_wards": sum(1 for e in my_wards if e.get("wardType") == "CONTROL_WARD"),
        })
    return rows
