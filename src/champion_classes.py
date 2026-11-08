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
