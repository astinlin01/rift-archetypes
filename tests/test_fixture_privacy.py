"""Privacy tests for the committed fixtures.

The fixtures are real Riot match payloads, so they arrive full of other players' identities
(PUUIDs, summoner IDs, Riot IDs). The repo is public, so those must be replaced with stable
fakes. The feature pipeline never reads them: tests find river3's row by champion + role,
so anonymizing changes no hand-checked value.

Match IDs stay real on purpose: a single match is public on any stats site, and old
commits' tests reference these IDs (see the history rewrite in the README).
"""
import json
import os

import pytest

from anonymize_fixtures import anonymize, ANON_PREFIX, ANON_TAGLINE

from conftest import FIXTURES_DIR

FIXTURE_FILES = ["matches.ndjson", "timelines.ndjson"]
ID_KEYS = ("puuid", "summonerId")


def load(name):
    with open(os.path.join(FIXTURES_DIR, name), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def walk(obj):
    """Yield every dict nested anywhere inside obj."""
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk(v)


@pytest.mark.parametrize("name", FIXTURE_FILES)
def test_no_real_player_ids(name):
    for game in load(name):
        for participant_puuid in game["metadata"]["participants"]:
            assert participant_puuid.startswith(ANON_PREFIX)
        for d in walk(game):
            for key in ID_KEYS:
                if key in d:
                    assert d[key].startswith(ANON_PREFIX), f"{key} not anonymized: {d[key][:12]}..."


def test_no_real_riot_ids_or_profile_details():
    for game in load("matches.ndjson"):
        for p in game["info"]["participants"]:
            assert p["riotIdGameName"].startswith("Player-")
            assert p["riotIdTagline"] == ANON_TAGLINE
            assert p["summonerName"] == ""
            assert p["profileIcon"] == 0
            assert p["summonerLevel"] == 0


def test_metadata_participants_match_info_order():
    """metadata.participants and info.participants must still line up after anonymizing."""
    for name in FIXTURE_FILES:
        for game in load(name):
            assert game["metadata"]["participants"] == [p["puuid"] for p in game["info"]["participants"]]


def test_same_player_maps_to_same_fake_across_files():
    """A match's timeline and its match payload must agree on who is who."""
    matches = {g["metadata"]["matchId"]: g for g in load("matches.ndjson")}
    timelines = load("timelines.ndjson")
    assert timelines, "expected timeline fixtures"
    for tl in timelines:
        match = matches[tl["metadata"]["matchId"]]
        assert tl["metadata"]["participants"] == match["metadata"]["participants"]


# --- The anonymizer itself ---------------------------------------------------------------

REAL = {
    "metadata": {"matchId": "NA1_1", "participants": ["realPuuidA", "realPuuidB"]},
    "info": {"participants": [
        {"puuid": "realPuuidA", "summonerId": "sumA", "riotIdGameName": "Faker",
         "riotIdTagline": "KR1", "summonerName": "Faker", "profileIcon": 29, "summonerLevel": 512,
         "championName": "Ahri", "kills": 7},
        {"puuid": "realPuuidB", "summonerId": "sumB", "riotIdGameName": "Bob",
         "riotIdTagline": "NA1", "summonerName": "", "profileIcon": 1, "summonerLevel": 30,
         "championName": "Jinx", "kills": 2},
    ]},
}


def test_anonymize_is_deterministic_for_a_salt_and_differs_across_salts():
    a1 = anonymize(json.loads(json.dumps(REAL)), salt="s1")
    a2 = anonymize(json.loads(json.dumps(REAL)), salt="s1")
    b = anonymize(json.loads(json.dumps(REAL)), salt="s2")
    assert a1 == a2
    assert a1["metadata"]["participants"] != b["metadata"]["participants"]


def test_anonymize_is_idempotent():
    """Re-running on already-anonymized data must not change it (the history rewrite relies on this)."""
    once = anonymize(json.loads(json.dumps(REAL)), salt="s1")
    twice = anonymize(json.loads(json.dumps(once)), salt="s1")
    assert once == twice


def test_anonymize_leaves_gameplay_fields_alone():
    out = anonymize(json.loads(json.dumps(REAL)), salt="s1")
    assert [p["championName"] for p in out["info"]["participants"]] == ["Ahri", "Jinx"]
    assert [p["kills"] for p in out["info"]["participants"]] == [7, 2]
    assert out["metadata"]["matchId"] == "NA1_1"
    # Distinct players stay distinct.
    assert len(set(out["metadata"]["participants"])) == 2
