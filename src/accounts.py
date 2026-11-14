"""Which Riot accounts count as "you".

Set them once in .env (gitignored) instead of in code:

    RIOT_IDS=yourname#na1, smurf#na1
    PUBLISH_RIOT_IDS=yourname#na1

RIOT_IDS is every account you own (fetching, comparisons). PUBLISH_RIOT_IDS is the subset
that goes on the public dashboard; the export never falls back to RIOT_IDS, so an account
only reaches the public site if you listed it there.

Every script that needs your accounts calls default_riot_ids() / require_riot_ids(), so the
list lives in one place and your accounts never land in the public repo.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

ENV_VAR = "RIOT_IDS"
PUBLISH_ENV_VAR = "PUBLISH_RIOT_IDS"


def parse_riot_ids(raw: str | None, env_var: str = ENV_VAR) -> list[str]:
    """Split a comma-separated RIOT_IDS value into clean, deduplicated gameName#tagLine strings."""
    if not raw:
        return []
    ids: list[str] = []
    seen: set[str] = set()
    for part in raw.strip().strip('"').strip("'").split(","):
        riot_id = part.strip()
        if not riot_id:
            continue
        name, _, tag = riot_id.partition("#")
        if not name.strip() or not tag.strip():
            raise ValueError(f"{env_var}: {riot_id!r} is not in gameName#tagLine form")
        if riot_id.lower() not in seen:  # Riot IDs are case-insensitive
            seen.add(riot_id.lower())
            ids.append(riot_id)
    return ids


def default_riot_ids() -> list[str]:
    """Riot IDs from RIOT_IDS (environment or .env). Empty list if unset."""
    load_dotenv()  # no override: a real environment variable beats .env
    return parse_riot_ids(os.getenv(ENV_VAR))


def require_riot_ids(cli_ids: list[str] | None) -> list[str]:
    """CLI --riot-id values if given, else RIOT_IDS; exit with setup instructions if neither."""
    ids = list(cli_ids or []) or default_riot_ids()
    if not ids:
        raise SystemExit(
            f"No Riot IDs given. Pass --riot-id name#tag, or add {ENV_VAR}=name#tag[,other#tag] to .env."
        )
    return ids


def require_publish_riot_ids(cli_ids: list[str] | None) -> list[str]:
    """Accounts for the public dashboard: CLI --riot-id values if given, else PUBLISH_RIOT_IDS.
    Deliberately no fallback to RIOT_IDS -- exits with instructions instead."""
    load_dotenv()
    ids = list(cli_ids or []) or parse_riot_ids(os.getenv(PUBLISH_ENV_VAR), PUBLISH_ENV_VAR)
    if not ids:
        raise SystemExit(
            f"No accounts to publish. Add {PUBLISH_ENV_VAR}=name#tag[,other#tag] to .env (the accounts "
            f"shown on the public dashboard), or pass --riot-id name#tag. {ENV_VAR} is not used here on purpose."
        )
    return ids
