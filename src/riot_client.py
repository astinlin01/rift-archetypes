"""
Shared Riot API client: handles routing, authentication, and rate limiting.

Every other script in this project should go through this class rather than
calling requests directly, so rate-limit handling lives in exactly one place.
"""

from __future__ import annotations

import os
import time
from urllib.parse import quote

import requests
from dotenv import load_dotenv

# Always prefer the current .env value in this repo over any inherited shell var.
load_dotenv(override=True)

API_KEY = os.getenv("RIOT_API_KEY", "").strip().strip('"').strip("'")
if not API_KEY:
    raise RuntimeError("RIOT_API_KEY not found. Check that your .env file exists and is loaded.")

# Platform routing (league-v4, summoner-v4) -> regional routing (match-v5, account-v1)
PLATFORM_TO_REGION = {
    "na1": "americas",
    "br1": "americas",
    "la1": "americas",
    "la2": "americas",
    "oc1": "americas",
    "euw1": "europe",
    "eun1": "europe",
    "tr1": "europe",
    "ru": "europe",
    "kr": "asia",
    "jp1": "asia",
}

# Dev key limits are 20 req/sec and 100 req/2min. A 1.2s delay between every
# call keeps you comfortably under both without needing a fancier token bucket.
REQUEST_DELAY_SECONDS = 1.2
REQUEST_TIMEOUT_SECONDS = 30


class RiotClient:
    def __init__(self, platform: str):
        if platform not in PLATFORM_TO_REGION:
            raise ValueError(f"Unknown platform '{platform}'. Add it to PLATFORM_TO_REGION.")
        self.platform = platform
        self.region = PLATFORM_TO_REGION[platform]
        self.headers = {"X-Riot-Token": API_KEY}

    def _get(self, url: str, params: dict | None = None) -> dict:
        """Single GET request with rate-limit handling and basic retries."""
        while True:
            try:
                response = requests.get(
                    url,
                    headers=self.headers,
                    params=params,
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
            except requests.RequestException as exc:
                print(f"Request error ({exc.__class__.__name__}). Retrying in 5s...")
                time.sleep(5)
                continue

            time.sleep(REQUEST_DELAY_SECONDS)

            if response.status_code == 200:
                return response.json()

            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 5))
                print(f"Rate limited. Waiting {retry_after}s...")
                time.sleep(retry_after)
                continue

            if response.status_code in (503, 504):
                print(f"Server error {response.status_code}. Retrying in 5s...")
                time.sleep(5)
                continue

            if response.status_code == 401:
                raise RuntimeError(
                    "Riot API returned 401 Unauthorized. Your RIOT_API_KEY is invalid or expired. "
                    "Regenerate a key in the Riot Developer Portal and update .env."
                )

            # 403 (bad/expired key), 404 (not found), etc. fail loudly.
            response.raise_for_status()

    def get_league_entries(self, queue: str, tier: str, division: str, page: int = 1) -> list:
        """
        Pull one page of players at a given tier/division.
        Note: MASTER/GRANDMASTER/CHALLENGER use a different endpoint.
        """
        url = (
            f"https://{self.platform}.api.riotgames.com/"
            f"lol/league/v4/entries/{queue}/{tier}/{division}"
        )
        return self._get(url, params={"page": page})

    def get_top_league(self, queue: str, tier: str) -> dict:
        """Pull top-tier league payload for MASTER, GRANDMASTER, or CHALLENGER."""
        tier_normalized = tier.upper()
        allowed = {"MASTER", "GRANDMASTER", "CHALLENGER"}
        if tier_normalized not in allowed:
            raise ValueError(f"tier must be one of {sorted(allowed)} for get_top_league")

        endpoint_map = {
            "MASTER": "masterleagues",
            "GRANDMASTER": "grandmasterleagues",
            "CHALLENGER": "challengerleagues",
        }
        endpoint = endpoint_map[tier_normalized]
        url = f"https://{self.platform}.api.riotgames.com/lol/league/v4/{endpoint}/by-queue/{queue}"
        return self._get(url)

    def get_summoner_by_id(self, summoner_id: str) -> dict:
        """Resolve encrypted summonerId to summoner-v4 payload, including puuid."""
        url = f"https://{self.platform}.api.riotgames.com/lol/summoner/v4/summoners/{summoner_id}"
        return self._get(url)

    def get_match_ids(self, puuid: str, queue: int = 420, count: int = 20, start: int = 0) -> list:
        """Pull ranked match IDs for a player by puuid."""
        url = f"https://{self.region}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
        params = {"queue": queue, "count": count, "start": start}
        return self._get(url, params=params)

    def get_match(self, match_id: str) -> dict:
        """Pull full match details for a single match ID."""
        url = f"https://{self.region}.api.riotgames.com/lol/match/v5/matches/{match_id}"
        return self._get(url)

    def get_account_by_riot_id(self, game_name: str, tag_line: str, region: str | None = None) -> dict:
        """Resolve Riot ID (gameName#tagLine) to a global account payload containing puuid."""
        target_region = region or self.region
        encoded_game_name = quote(game_name, safe="")
        encoded_tag_line = quote(tag_line, safe="")
        url = (
            f"https://{target_region}.api.riotgames.com/"
            f"riot/account/v1/accounts/by-riot-id/{encoded_game_name}/{encoded_tag_line}"
        )
        return self._get(url)

    def get_account_by_riot_id_any_region(self, game_name: str, tag_line: str) -> dict:
        """Resolve Riot ID by trying all account-v1 regional clusters."""
        last_error: Exception | None = None
        for cluster in (self.region, "americas", "europe", "asia"):
            try:
                return self.get_account_by_riot_id(game_name=game_name, tag_line=tag_line, region=cluster)
            except requests.HTTPError as exc:
                last_error = exc
                continue
        if last_error is not None:
            raise last_error
        raise RuntimeError("Failed to resolve Riot ID across account-v1 regional clusters.")
