"""
Champion -> class lookup from Riot's Data Dragon champion.json.

Each champion gets its *primary* (first) Data Dragon tag, e.g. Jayce ["Fighter",
"Marksman"] -> "Fighter". One class per champion keeps the champion-pool layer and the
rare-champion pooling in playstyle_space.py simple to compute and explain.

Lookups are case-insensitive: match-v5 spells some champions differently from Data
Dragon (match data says "FiddleSticks", Data Dragon says "Fiddlesticks").

The file is cached at data/raw/ddragon/champion.json. To refresh it after a new
champion release, re-download:
    https://ddragon.leagueoflegends.com/cdn/<version>/data/en_US/champion.json
(latest <version> is the first entry of https://ddragon.leagueoflegends.com/api/versions.json).
"""

from __future__ import annotations

import json
import os

DEFAULT_CHAMPION_DATA_PATH = "data/raw/ddragon/champion.json"
UNKNOWN_CLASS = "Unknown"


def load_champion_classes(path: str = DEFAULT_CHAMPION_DATA_PATH) -> dict[str, str]:
    """Return {lowercased champion id/name: primary class tag}."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Champion data not found at {path}. Download Data Dragon's champion.json there "
            "(see this module's docstring for the URL)."
        )
    with open(path, "r", encoding="utf-8") as infile:
        champions = json.load(infile)["data"]

    classes: dict[str, str] = {}
    for key, champion in champions.items():
        tags = champion.get("tags") or [UNKNOWN_CLASS]
        # Index by the data key, the id ("MonkeyKing", what match-v5 uses) and the display
        # name ("Wukong"), so any spelling resolves.
        for alias in (key, champion.get("id"), champion.get("name")):
            if alias:
                classes[str(alias).lower()] = tags[0]
    return classes


def champion_class(champion_name: str | None, classes: dict[str, str]) -> str:
    if not champion_name:
        return UNKNOWN_CLASS
    return classes.get(str(champion_name).lower(), UNKNOWN_CLASS)


# ---------- Support subclasses (UTILITY only) ----------
#
# Data Dragon tags lump most supports into "Support" (Lulu, Thresh, Pyke alike) or
# "Tank", which says little about how a support plays. In UTILITY this hand-curated map
# takes precedence; elsewhere (Karma mid, Lux mid) the primary tag still applies.
# Supports not listed fall back to their primary tag (Pantheon -> Fighter, Elise ->
# Assassin). Keys are lowercased match-v5 championName spellings. Add new supports here.
#   Enchanter: heals, shields, buffs      Pick: hooks / long-range catch CC
#   Tank: engage and peel                 Mage: damage and poke
SUPPORT_SUBCLASSES: dict[str, str] = {
    **dict.fromkeys(
        ["lulu", "nami", "janna", "sona", "yuumi", "milio", "soraka", "seraphine", "renata", "karma", "zilean"],
        "Enchanter",
    ),
    **dict.fromkeys(["thresh", "blitzcrank", "bard", "rakan", "morgana", "neeko", "pyke"], "Pick"),
    **dict.fromkeys(
        [
            "nautilus", "alistar", "rell", "leona", "braum", "poppy", "shen", "maokai", "tahmkench",
            "galio", "amumu", "skarner", "sion", "zac", "taric",
        ],
        "Tank",
    ),
    **dict.fromkeys(
        [
            "lux", "swain", "velkoz", "zyra", "xerath", "brand", "zoe", "hwei", "mel", "anivia",
            "lissandra", "taliyah", "sylas", "fiddlesticks",
        ],
        "Mage",
    ),
    # Data Dragon's primary tag is "Support", but she's played for damage.
    "senna": "Marksman",
}
SUPPORT_ROLE = "UTILITY"


def role_champion_class(
    champion_name: str | None,
    role: str | None,
    classes: dict[str, str],
    support_overrides: dict[str, str] = SUPPORT_SUBCLASSES,
) -> str:
    """Class used by the champion-pool layer: the support subclass in UTILITY if the
    champion is mapped, otherwise the primary Data Dragon tag."""
    if role == SUPPORT_ROLE and champion_name:
        override = support_overrides.get(str(champion_name).lower())
        if override:
            return override
    return champion_class(champion_name, classes)
