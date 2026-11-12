"""Replace player identities in Riot match/timeline NDJSON with stable fakes.

Usage:
    python src/anonymize_fixtures.py tests/fixtures/matches.ndjson tests/fixtures/timelines.ndjson [--salt S]

All files passed in one run share a salt, so a player gets the same fake ID in a match payload
and its timeline. Fakes are HMAC-SHA256(salt, real_id): stable within a run, but without the
salt nobody can test "is PUUID X in these fixtures?". If --salt is omitted a random one is used
and discarded, which is what you want unless you need to reproduce an earlier run exactly.

Idempotent: values that are already fake are left alone, so re-running is safe.
"""
import argparse
import hashlib
import hmac
import json
import secrets

ANON_PREFIX = "anon-"
ANON_TAGLINE = "ANON"
ID_KEYS = ("puuid", "summonerId")


def _fake(salt: str, real: str) -> str:
    if real.startswith(ANON_PREFIX):
        return real
    digest = hmac.new(salt.encode(), real.encode(), hashlib.sha256).hexdigest()
    return ANON_PREFIX + digest[:24]


def _scrub(obj, salt: str):
    if isinstance(obj, list):
        return [_scrub(v, salt) for v in obj]
    if not isinstance(obj, dict):
        return obj
    out = {k: _scrub(v, salt) for k, v in obj.items()}
    for key in ID_KEYS:
        if isinstance(out.get(key), str):
            out[key] = _fake(salt, out[key])
    if "riotIdGameName" in out:
        # Name derives from the (already fake) PUUID, so it's stable and idempotent too.
        if not str(out["riotIdGameName"]).startswith("Player-"):
            out["riotIdGameName"] = "Player-" + out.get("puuid", "")[len(ANON_PREFIX):][:6]
        out["riotIdTagline"] = ANON_TAGLINE
    if "summonerName" in out:
        out["summonerName"] = ""
    for key in ("profileIcon", "summonerLevel"):
        if key in out:
            out[key] = 0
    return out


def anonymize(game: dict, salt: str) -> dict:
    """Return a copy of one match or timeline payload with identities replaced."""
    game = _scrub(game, salt)
    meta = game.get("metadata", {})
    meta["participants"] = [_fake(salt, p) for p in meta.get("participants", [])]
    return game


def anonymize_file(path: str, salt: str) -> None:
    with open(path, encoding="utf-8") as f:
        games = [json.loads(line) for line in f if line.strip()]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for game in games:
            f.write(json.dumps(anonymize(game, salt)) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", help="NDJSON files to anonymize in place.")
    parser.add_argument("--salt", default=None, help="HMAC salt. Random if omitted.")
    args = parser.parse_args()
    salt = args.salt or secrets.token_hex(16)
    for path in args.paths:
        anonymize_file(path, salt)
        print(f"anonymized {path}")


if __name__ == "__main__":
    main()
