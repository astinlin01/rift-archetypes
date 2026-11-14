"""Your Riot IDs come from the RIOT_IDS setting (.env), not from hardcoded lists in the scripts."""
import pytest

from accounts import parse_riot_ids, require_riot_ids


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("river3#na1", ["river3#na1"]),
        ("a#na1, b#euw ,c#kr", ["a#na1", "b#euw", "c#kr"]),  # whitespace around commas
        ('"a#na1,b#euw"', ["a#na1", "b#euw"]),  # quoted value, as some .env editors write it
        ("a#na1,,A#NA1", ["a#na1"]),  # blanks dropped; Riot IDs are case-insensitive, so dedupe
        ("Gamer Name#NA1", ["Gamer Name#NA1"]),  # spaces inside a game name are legal
        ("", []),
        (None, []),
    ],
)
def test_parse_riot_ids(raw, expected):
    assert parse_riot_ids(raw) == expected


def test_parse_rejects_missing_tag():
    with pytest.raises(ValueError, match="gameName#tagLine"):
        parse_riot_ids("river3")


def test_require_prefers_cli_ids_over_env(monkeypatch):
    monkeypatch.setenv("RIOT_IDS", "env#na1")
    assert require_riot_ids(["cli#na1"]) == ["cli#na1"]
    assert require_riot_ids([]) == ["env#na1"]


def test_require_explains_how_to_configure_when_empty(monkeypatch):
    monkeypatch.setenv("RIOT_IDS", "")
    with pytest.raises(SystemExit, match="RIOT_IDS"):
        require_riot_ids([])


# --- Which accounts go on the public page ----------------------------------------------------------
# PUBLISH_RIOT_IDS is separate from RIOT_IDS (all your accounts, used for fetching), so a plain
# rebuild can't put an account on the public site that you didn't choose to publish.

from accounts import require_publish_riot_ids


def test_publish_uses_its_own_list(monkeypatch):
    monkeypatch.setenv("RIOT_IDS", "main#na1,alt#na1")
    monkeypatch.setenv("PUBLISH_RIOT_IDS", "main#na1")
    assert require_publish_riot_ids([]) == ["main#na1"]


def test_publish_prefers_cli_ids(monkeypatch):
    monkeypatch.setenv("PUBLISH_RIOT_IDS", "main#na1")
    assert require_publish_riot_ids(["cli#na1"]) == ["cli#na1"]


def test_publish_never_falls_back_to_all_accounts(monkeypatch):
    # Unset publish list + set RIOT_IDS must stop, not quietly publish every account.
    monkeypatch.setenv("RIOT_IDS", "main#na1,alt#na1")
    monkeypatch.setenv("PUBLISH_RIOT_IDS", "")
    with pytest.raises(SystemExit, match="PUBLISH_RIOT_IDS"):
        require_publish_riot_ids([])
