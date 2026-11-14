"""
Export the JSON payload for the static self-analysis dashboard (dashboard/template.html,
merged by build_dashboard.py).

The page recomputes everything it shows in the browser (dashboard/profile_math.js), so
the game-window, champion, role, region and tier filters work with no server. This
script therefore exports, per role:
- your games: raw headline stats, champion-adjusted stats, per-game trait scores and
  the champion's class (support subclasses applied);
- reference curves per region and tier (percentile cutoffs of reference players'
  averages): raw stats, champion-adjusted stats, trait profiles, and the champion-pool
  baseline;
- each trait's single-game reliability, the role's archetype signatures and gap list.

Run from repo root (after build_features.py, and build_timeline_features.py if you have
timelines):
    python src/export_dashboard_data.py
"""

from __future__ import annotations

import argparse
import base64
import json
import os

import numpy as np
import pandas as pd
import requests

from accounts import require_publish_riot_ids
from champion_classes import load_champion_classes, role_champion_class
from dashboard_payload import (
    crop_splash,
    json_number,
    player_trait_points,
    pool_reference,
    reference_by_tier,
    splash_champions,
    style_scores,
)
from feature_config import ALL_FEATURE_FIELDS, VALID_ROLES, add_derived_features, current_season
from performance import (
    FULL_OWN_GAMES,
    MIN_OWN_GAMES,
    apply_performance_adjuster,
    fit_performance_adjuster,
    gap_features_for_role,
)
from pipeline_io import iter_ndjson
from playstyle_space import ROLE_ARCHETYPES, add_game_state_columns
from style_profile import CONFIDENCE_THRESHOLDS
from timeline_features import TIMELINE_FEATURE_COLUMNS

FEATURE_COLUMNS = [f.name for f in ALL_FEATURE_FIELDS]
MIN_REFERENCE_SAMPLE = 20  # skip a (role, tier) reference if fewer *players* than this
# Reference curves are built from per-player averages, not individual games: the
# dashboard compares *your average* over a window, and an average over many games has
# far less spread than single games -- ranking it against single games would squash a
# genuinely top-10% average down to ~70th percentile.
MIN_PLAYER_GAMES_FOR_REFERENCE = 10
MIN_PRO_GAMES_FOR_REFERENCE = 5
# A pro league only has ~10-16 players per role, so the solo-queue player minimum
# (MIN_REFERENCE_SAMPLE) would rule every league out.
MIN_PRO_REFERENCE_SAMPLE = 8
BREAKPOINT_COUNT = 51  # percentile grid: 0, 2, 4, ..., 100
ALL_ROLES_KEY = "ALL"

# Highest-to-lowest, so the "Compare vs." dropdown reads Challenger > Grandmaster >
# Master > Diamond instead of alphabetically.
TIER_RANK = {"CHALLENGER": 0, "GRANDMASTER": 1, "MASTER": 2, "DIAMOND": 3}
REGION_RANK = {"NA1": 0, "KR": 1}
REGION_DISPLAY_NAMES = {"NA1": "NA"}

DDRAGON_VERSIONS_URL = "https://ddragon.leagueoflegends.com/api/versions.json"
DDRAGON_ICON_URL = "https://ddragon.leagueoflegends.com/cdn/{version}/img/champion/{champion}.png"
CHAMPION_ICON_CACHE_DIR = "data/raw/champion_icon_cache"
# Splash art has no version in its URL. Cached already cropped for the player card.
DDRAGON_SPLASH_URL = "https://ddragon.leagueoflegends.com/cdn/img/champion/splash/{champion}_0.jpg"
CHAMPION_SPLASH_CACHE_DIR = "data/raw/champion_splash_cache"

PRO_LEAGUE_RANK = {"LCK": 0, "LPL": 1}
MIN_PRO_PLAYER_GAMES = 5  # skip a pro player from the "which pro do you play like" list below this

# Subset of HEADLINE_FEATURES that Oracle's Elixir's schema has a clean equivalent for
# (see build_pro_features.py's PRO_FEATURE_MAP-equivalent construction) -- no CC-time or
# lane-opponent-diff data in pro match data, so those two are dropped here.
PRO_HEADLINE_FEATURES = [
    ("damage_per_minute", "Damage / min", False),
    ("gold_earned_per_min", "Gold / min", False),
    ("vision_score_per_minute", "Vision score / min", False),
    ("kda", "KDA", False),
    ("kill_participation", "Kill participation", False),
    ("team_damage_pct", "Team damage share", False),
    ("cs_total_per_min", "CS / min", False),
    ("damage_taken_per_min", "Damage taken / min", True),
    ("control_wards_per_min", "Control wards / min", False),
    ("deaths_per_min", "Deaths / min", True),
]

# Curated subset + display labels for the headline percentile comparison.
# invert=True means a lower raw value is typically the better outcome for this role.
HEADLINE_FEATURES = [
    ("damage_per_minute", "Damage / min", False),
    ("gold_earned_per_min", "Gold / min", False),
    ("vision_score_per_minute", "Vision score / min", False),
    ("kda", "KDA", False),
    ("kill_participation", "Kill participation", False),
    ("team_damage_pct", "Team damage share", False),
    ("cs_total_per_min", "CS / min", False),
    ("cc_time_per_min", "CC time on enemies / min", False),
    ("damage_taken_per_min", "Damage taken / min", True),
    ("control_wards_per_min", "Control wards / min", False),
    ("deaths_per_min", "Deaths / min", True),
    ("max_cs_advantage_lane_opp", "Max CS lead on opponent", False),
]

# Display labels for the gap list (headline labels are reused where they exist).
GAP_FEATURE_LABELS = {
    "lane_minions_first_10": "Lane minions by 10:00",
    "lane_gold_diff_per_min": "Lane gold lead / min",
    "turret_plates_taken": "Turret plates",
    "cs_diff_at_10": "CS lead at 10:00",
    "gold_diff_at_14": "Gold lead at 14:00",
    "xp_diff_at_14": "XP lead at 14:00",
    "early_deaths": "Deaths before 14:00",
    "wards_killed_per_min": "Wards killed / min",
    "damage_to_objectives_per_min": "Objective damage / min",
    "epic_objective_participation": "Dragon / Herald / Baron participation",
    "turret_participation": "Turret participation",
    "early_objective_participation": "Early objective participation",
    "solo_kills": "Solo kills",
}
# Units for a gap's size in game terms ("≈330 gold per 31-min game", "0.4 KDA").
# Percent features are shown in percentage points instead.
GAP_UNITS = {
    "gold_earned_per_min": "gold", "cs_total_per_min": "CS", "damage_per_minute": "damage",
    "deaths_per_min": "deaths", "vision_score_per_minute": "vision score",
    "control_wards_per_min": "control wards", "wards_killed_per_min": "wards cleared",
    "damage_to_objectives_per_min": "objective damage", "lane_gold_diff_per_min": "lane gold lead",
    "kda": "KDA", "lane_minions_first_10": "minions", "max_cs_advantage_lane_opp": "CS",
    "turret_plates_taken": "plates", "cs_diff_at_10": "CS", "gold_diff_at_14": "gold",
    "xp_diff_at_14": "XP", "early_deaths": "deaths", "solo_kills": "solo kills",
}
# Shares shown as percentages on the page.
PERCENT_FEATURES = [
    "kill_participation", "team_damage_pct", "epic_objective_participation",
    "turret_participation", "early_objective_participation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export interactive dashboard JSON payload.")
    parser.add_argument("--features-path", default="data/processed/participant_features.parquet")
    parser.add_argument("--matches-path", default="data/raw/matches.ndjson")
    parser.add_argument("--match-ids-path", default="data/raw/match_ids.ndjson")
    parser.add_argument(
        "--riot-id",
        action="append",
        default=None,
        help="Riot ID to publish, e.g. 'yourname#na1'. Repeatable. Defaults to PUBLISH_RIOT_IDS in .env.",
    )
    parser.add_argument(
        "--cached-icons-only",
        action="store_true",
        help="Don't contact Data Dragon; embed only champion icons and splash art already cached on disk.",
    )
    parser.add_argument(
        "--timeline-features-path",
        default="data/processed/timeline_features.parquet",
        help="Early-game features (build_timeline_features.py). Optional: without it, traits use end-of-game features only.",
    )
    parser.add_argument("--pro-features-path", default="data/processed/pro_player_features.parquet")
    parser.add_argument("--output-path", default="data/processed/dashboard_data.json")
    return parser.parse_args()


def resolve_puuid_from_match_ids(match_ids_path: str, riot_id: str) -> str:
    # Riot IDs are case-insensitive ("name#NA1" and "name#na1" are one account).
    wanted = riot_id.strip().lower()
    for row in iter_ndjson(match_ids_path):
        if str(row.get("source_riot_id", "")).strip().lower() == wanted:
            return row["source_puuid"]
    raise RuntimeError(f"No match_ids.ndjson rows found for riot id {riot_id}")


def load_game_creation_by_match_id(matches_path: str, match_ids: set[str]) -> dict[str, int]:
    creation_by_id: dict[str, int] = {}
    for match in iter_ndjson(matches_path):
        match_id = match.get("metadata", {}).get("matchId")
        if match_id in match_ids:
            creation_by_id[match_id] = match.get("info", {}).get("gameCreation")
    return creation_by_id


def player_averages(rows: pd.DataFrame, player_column: str, min_games: int, feature_columns: list[str]) -> pd.DataFrame:
    counts = rows.groupby(player_column).size()
    regulars = counts[counts >= min_games].index
    return rows[rows[player_column].isin(regulars)].groupby(player_column)[feature_columns].mean()


def build_curves_for_role_df(
    role_df: pd.DataFrame,
    headline_features=HEADLINE_FEATURES,
    player_column: str = "puuid",
    min_games: int = MIN_PLAYER_GAMES_FOR_REFERENCE,
    min_players: int = MIN_REFERENCE_SAMPLE,
) -> dict:
    feature_columns = [c for c, _, _ in headline_features]
    averages = player_averages(role_df, player_column, min_games, feature_columns)
    if len(averages) < min_players:
        return {}
    feature_curves = {}
    quantiles = np.linspace(0, 100, BREAKPOINT_COUNT)
    for column in feature_columns:
        series = averages[column].dropna()
        if series.empty:
            continue
        feature_curves[column] = {
            "ref_avg": round(float(series.mean()), 3),
            # Spread of player averages -- the dashboard uses it to place you relative to
            # your solo-queue peers when matching against pros (see renderProMatches).
            "std": round(float(series.std(ddof=0)) or 1.0, 4),
            "n": int(len(series)),  # number of players, not games
            "breakpoints": [round(float(v), 4) for v in np.percentile(series, quantiles)],
        }
    return feature_curves


def load_pro_features(pro_features_path: str) -> pd.DataFrame | None:
    if not os.path.exists(pro_features_path):
        print(f"No pro feature table found at {pro_features_path}; skipping pro comparison. Run build_pro_features.py to add it.")
        return None
    return pd.read_parquet(pro_features_path)


def build_pro_reference(pro_df: pd.DataFrame) -> tuple[dict, list[str]]:
    """Percentile reference curves per (role, league), built from Oracle's Elixir data.

    Mirrors build_reference_curves' shape but keyed by league instead of region/tier,
    and uses PRO_HEADLINE_FEATURES (a subset of our own features that has a clean
    equivalent in Oracle's Elixir's schema -- see build_pro_features.py).
    """
    leagues = sorted(pro_df["league"].unique(), key=lambda l: PRO_LEAGUE_RANK.get(l, len(PRO_LEAGUE_RANK)))

    reference: dict[str, dict] = {}
    for role in VALID_ROLES:
        reference[role] = {}
        role_df = pro_df[pro_df["role"] == role]
        for league in leagues:
            league_role_df = role_df[role_df["league"] == league]
            feature_curves = build_curves_for_role_df(
                league_role_df,
                PRO_HEADLINE_FEATURES,
                player_column="playername",
                min_games=MIN_PRO_GAMES_FOR_REFERENCE,
                min_players=MIN_PRO_REFERENCE_SAMPLE,
            )
            if feature_curves:
                reference[role][league] = feature_curves
    return reference, leagues


def build_pro_player_profiles(pro_df: pd.DataFrame) -> dict:
    """Per-pro-player season-average stat profiles per role, standardized against the
    pro population's own scale, for a client-side "which pro do you play like" nearest-
    neighbor match. A player's team/league label uses whichever they played the most
    games under this season (handles a mid-split team change without splitting them
    into two entries).
    """
    feature_cols = [c for c, _, _ in PRO_HEADLINE_FEATURES]
    profiles: dict[str, dict] = {}

    for role in VALID_ROLES:
        role_df = pro_df[pro_df["role"] == role]
        if role_df.empty:
            continue

        # Standardize against the spread of *player averages* (not single pro games),
        # matching how the dashboard standardizes you against your solo-queue peers'
        # averages -- so both z-scores mean "how far from typical for your population".
        qualifying = player_averages(role_df, "playername", MIN_PRO_PLAYER_GAMES, feature_cols)
        if qualifying.empty:
            continue
        mean = qualifying.mean()
        std = qualifying.std(ddof=0).replace(0, 1.0)

        players = []
        for player_name, player_df in role_df.groupby("playername"):
            if len(player_df) < MIN_PRO_PLAYER_GAMES:
                continue
            representative_team = player_df["teamname"].value_counts().idxmax()
            representative_league = player_df["league"].value_counts().idxmax()
            raw_avg = player_df[feature_cols].mean()
            z = (raw_avg - mean) / std
            top_champs = player_df["champion"].value_counts().head(3).index.tolist()

            players.append(
                {
                    "name": player_name,
                    "team": representative_team,
                    "league": representative_league,
                    "games": int(len(player_df)),
                    "win_rate": round(float(player_df["win"].mean()), 3),
                    "top_champs": top_champs,
                    "z": {col: round(float(z[col]), 4) for col in feature_cols},
                }
            )

        if players:
            profiles[role] = {
                "mean": {col: round(float(mean[col]), 4) for col in feature_cols},
                "std": {col: round(float(std[col]), 4) for col in feature_cols},
                "players": players,
            }

    return profiles


def build_reference_curves(df: pd.DataFrame, regions: list[str], tiers: list[str]) -> dict:
    reference: dict[str, dict] = {}
    for role in list(VALID_ROLES) + [ALL_ROLES_KEY]:
        reference[role] = {}
        role_df = df if role == ALL_ROLES_KEY else df[df["teamPosition"] == role]
        for region in regions:
            region_role_df = role_df[role_df["region"] == region]
            region_curves = {}
            for tier in tiers:
                tier_role_df = region_role_df[region_role_df["tier"] == tier]
                feature_curves = build_curves_for_role_df(tier_role_df)
                if feature_curves:
                    region_curves[tier] = feature_curves
            if region_curves:
                reference[role][region] = region_curves
    return reference


def fetch_champion_splashes(champion_names: set[str], download: bool = True) -> dict[str, str]:
    """Base64 data-URI splash art for the player card, cropped to the card's shape and
    compressed (crop_splash), so each is ~60 KB instead of ~250 KB. The original is cached,
    so the crop can be tuned without downloading again."""
    splashes: dict[str, str] = {}
    os.makedirs(CHAMPION_SPLASH_CACHE_DIR, exist_ok=True)
    for champion in sorted(champion_names):
        cache_path = os.path.join(CHAMPION_SPLASH_CACHE_DIR, f"{champion}_0.jpg")  # Data Dragon's name
        if not os.path.exists(cache_path) and download:
            try:
                response = requests.get(DDRAGON_SPLASH_URL.format(champion=champion), timeout=15)
                if response.status_code == 200:
                    with open(cache_path, "wb") as outfile:
                        outfile.write(response.content)
                else:
                    print(f"  no Data Dragon splash for '{champion}' (status {response.status_code})")
            except Exception as exc:
                print(f"  failed to fetch splash for '{champion}': {exc}")
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as infile:
                cropped = crop_splash(infile.read())
            splashes[champion] = "data:image/jpeg;base64," + base64.b64encode(cropped).decode("ascii")
    return splashes


def fetch_champion_icons(champion_names: set[str], download: bool = True) -> dict[str, str]:
    """Base64 data-URI icons for the given champion names, fetched from Data Dragon and
    cached on disk. Embedding as data URIs is required because the dashboard, once
    published as an Artifact, runs under a CSP that blocks image requests to external
    hosts -- only same-document data: URIs work.
    """
    icons: dict[str, str] = {}
    if not champion_names:
        return icons

    os.makedirs(CHAMPION_ICON_CACHE_DIR, exist_ok=True)

    version = None  # stays None when not downloading: only cached icons get embedded
    if download:
        try:
            response = requests.get(DDRAGON_VERSIONS_URL, timeout=10)
            response.raise_for_status()
            version = response.json()[0]
        except Exception as exc:
            print(f"Warning: couldn't fetch Data Dragon version ({exc}); using any cached icons only.")
    else:
        print("Icon download skipped (--cached-icons-only); using cached icons.")

    for champion in sorted(champion_names):
        cache_path = os.path.join(CHAMPION_ICON_CACHE_DIR, f"{champion}.png")
        if not os.path.exists(cache_path) and version:
            url = DDRAGON_ICON_URL.format(version=version, champion=champion)
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    with open(cache_path, "wb") as outfile:
                        outfile.write(response.content)
                else:
                    print(f"  no Data Dragon icon for '{champion}' (status {response.status_code})")
            except Exception as exc:
                print(f"  failed to fetch icon for '{champion}': {exc}")

        if os.path.exists(cache_path):
            with open(cache_path, "rb") as infile:
                encoded = base64.b64encode(infile.read()).decode("ascii")
            icons[champion] = f"data:image/png;base64,{encoded}"

    return icons


def merge_timeline_features(df: pd.DataFrame, path: str) -> pd.DataFrame:
    """Attach early-game features by (matchId, puuid). Games without a stored timeline
    keep NaN (never imputed). No file -> unchanged, with a note."""
    if not os.path.exists(path):
        print(f"No timeline features at {path}; traits will use end-of-game features only.")
        return df
    timeline = pd.read_parquet(path, columns=["matchId", "puuid", *TIMELINE_FEATURE_COLUMNS])
    merged = df.drop(columns=[c for c in TIMELINE_FEATURE_COLUMNS if c in df.columns]).merge(
        timeline, on=["matchId", "puuid"], how="left"
    )
    covered = merged[TIMELINE_FEATURE_COLUMNS[0]].notna().mean()
    print(f"Timeline features merged: {covered:.0%} of rows have a timeline.")
    return merged


def build_role_payload(
    role: str,
    df: pd.DataFrame,
    my_df: pd.DataFrame,
    regions: list[str],
    tiers: list[str],
    champion_classes: dict[str, str],
) -> tuple[dict, dict]:
    """(role payload, {(matchId, puuid): per-game extras for your games in this role}).

    Everything is fitted on the role's whole season population, so you and every
    reference tier share one set of champion baselines and one trait scale.
    """
    role_df = df[df["teamPosition"] == role].reset_index(drop=True)
    role_df["champion_class"] = [
        role_champion_class(name, role, champion_classes) for name in role_df["championName"]
    ]

    scores, icc, traits = style_scores(role_df, role, champion_classes)

    gap_features = gap_features_for_role(role)
    performance_columns = [
        c for c in dict.fromkeys([c for c, _, _ in HEADLINE_FEATURES] + list(gap_features)) if c in role_df.columns
    ]
    # ◆ view: champion, teammates' lead and game length removed (early stats: champion only).
    performance_adjuster = fit_performance_adjuster(
        role_df, performance_columns, champion_classes, early_columns=TIMELINE_FEATURE_COLUMNS
    )
    adjusted = apply_performance_adjuster(role_df, performance_adjuster)

    pool_by_tier: dict = {}
    for region in regions:
        for tier in tiers:
            subset = role_df[(role_df["region"] == region) & (role_df["tier"] == tier)]
            if len(subset):
                pool_by_tier.setdefault(region, {})[tier] = pool_reference(subset)

    payload = {
        "traits": traits,
        "trait_reliability": {t: json_number(icc.get(t)) for t in traits},
        "archetypes": ROLE_ARCHETYPES.get(role, {}),
        "gap_features": {f: list(spec) for f, spec in gap_features.items() if f in role_df.columns},
        "style_reference": reference_by_tier(role_df, scores, regions, tiers),
        # Style map: every reference player's trait averages, in `traits` order.
        "trait_points": player_trait_points(role_df, scores, traits, regions, tiers),
        "adjusted_reference": reference_by_tier(role_df, adjusted, regions, tiers),
        # Adjusted values are at this game length, so per-minute gaps convert to per-game here.
        "median_game_minutes": json_number(float((role_df["gameLength"] / 60).median())),
        "pool_reference": pool_by_tier,
    }

    mine = role_df["puuid"].isin(set(my_df["puuid"])).to_numpy()
    extras = {}
    for i in role_df.index[mine]:
        extras[(role_df.at[i, "matchId"], role_df.at[i, "puuid"])] = {
            "championClass": role_df.at[i, "champion_class"],
            "traits": {t: json_number(scores.at[i, t]) for t in traits},
            "adjusted": {c: json_number(adjusted.at[i, c]) for c in performance_columns},
        }
    return payload, extras


def run(args: argparse.Namespace) -> None:
    df = add_game_state_columns(add_derived_features(pd.read_parquet(args.features_path)))
    df = merge_timeline_features(df, args.timeline_features_path)
    riot_ids = require_publish_riot_ids(args.riot_id)

    # Everything -- reference curves, style baselines, and your own games -- is
    # restricted to one season so you're never compared across a stat-baseline shift.
    season = current_season(df)
    if season is not None:
        before = len(df)
        df = df[df["season"] == season].reset_index(drop=True)
        print(f"Season-matching to season {season}: kept {len(df)} of {before} rows.")
    else:
        print("Warning: feature table has no season column -- rebuild with build_features.py to season-match.")

    account_frames = []
    for riot_id in riot_ids:
        try:
            puuid = resolve_puuid_from_match_ids(args.match_ids_path, riot_id)
        except RuntimeError as error:
            print(f"Warning: {error}. Skipping (run fetch_recent_self.py for this account first).")
            continue
        account_df = df[df["puuid"] == puuid].copy()
        if account_df.empty:
            print(f"Warning: no feature rows found for {riot_id} (puuid {puuid}). Skipping.")
            continue
        account_df["account"] = riot_id
        account_frames.append(account_df)

    if not account_frames:
        raise RuntimeError("No feature rows found for any requested Riot ID. Run build_features.py after fetching matches.")

    my_df = pd.concat(account_frames, ignore_index=True)

    if "gameCreation" not in my_df.columns:
        creation_by_id = load_game_creation_by_match_id(args.matches_path, set(my_df["matchId"]))
        my_df["gameCreation"] = my_df["matchId"].map(creation_by_id)
    my_df = my_df.sort_values("gameCreation", ascending=False).reset_index(drop=True)

    tiers = sorted(
        (t for t in df["tier"].unique() if t != "UNKNOWN"),
        key=lambda t: TIER_RANK.get(t, len(TIER_RANK)),
    )
    regions = sorted(
        (r for r in df["region"].unique() if r != "UNKNOWN"),
        key=lambda r: REGION_RANK.get(r, len(REGION_RANK)),
    )
    print(f"Reference tiers with data: {tiers}")
    print(f"Reference regions with data: {regions}")

    print("Building reference percentile curves...")
    reference = build_reference_curves(df, regions, tiers)

    pro_df = load_pro_features(args.pro_features_path)
    if pro_df is not None:
        print("Building pro reference percentile curves...")
        pro_reference, pro_leagues = build_pro_reference(pro_df)
        print("Building pro player profiles for nearest-match comparison...")
        pro_player_profiles = build_pro_player_profiles(pro_df)
    else:
        pro_reference, pro_leagues, pro_player_profiles = {}, [], {}

    champion_classes = load_champion_classes()
    roles_payload = {}
    game_extras: dict[tuple[str, str], dict] = {}
    for role in VALID_ROLES:
        print(f"Building style, pool and performance payload for {role}...")
        role_payload, extras = build_role_payload(role, df, my_df, regions, tiers, champion_classes)
        roles_payload[role] = role_payload
        game_extras.update(extras)

    games = []
    for _, row in my_df.iterrows():
        feature_values = {column: json_number(row[column]) for column, _, _ in HEADLINE_FEATURES}
        games.append(
            {
                "matchId": row["matchId"],
                "account": row["account"],
                "champion": row["championName"],
                "role": row["teamPosition"],
                "win": bool(row["win"]),
                "gameCreation": int(row["gameCreation"]) if pd.notna(row["gameCreation"]) else None,
                "features": feature_values,
                **game_extras.get((row["matchId"], row["puuid"]), {}),
            }
        )

    print("Fetching champion icons...")
    champion_icons = fetch_champion_icons(set(my_df["championName"].unique()), download=not args.cached_icons_only)
    print(f"  {len(champion_icons)} icon(s) available.")
    print("Fetching player card splash art...")
    champion_splashes = fetch_champion_splashes(
        splash_champions(my_df, MIN_OWN_GAMES), download=not args.cached_icons_only
    )
    print(f"  {len(champion_splashes)} splash(es) available: {sorted(champion_splashes)}")

    # Per-account summaries, for notebook/debug introspection only -- the published
    # dashboard recomputes everything it shows from the filtered `games` list directly.
    accounts_summary = {}
    for riot_id in riot_ids:
        account_df = my_df[my_df["account"] == riot_id]
        if account_df.empty:
            continue
        accounts_summary[riot_id] = {
            "wins": int(account_df["win"].sum()),
            "losses": int((~account_df["win"]).sum()),
            "win_rate": round(float(account_df["win"].mean()), 3),
            "total_games": int(len(account_df)),
            "role_counts": account_df["teamPosition"].value_counts().to_dict(),
            "champion_counts": account_df["championName"].value_counts().to_dict(),
        }

    payload = {
        "season": season,
        "accounts": list(accounts_summary.keys()),
        "accounts_summary": accounts_summary,
        "min_own_games": MIN_OWN_GAMES,
        "full_own_games": FULL_OWN_GAMES,
        "confidence_thresholds": [list(t) for t in CONFIDENCE_THRESHOLDS],
        "tiers": tiers,
        "regions": regions,
        "region_display_names": REGION_DISPLAY_NAMES,
        "headline_features": [{"feature": c, "label": l, "invert": inv} for c, l, inv in HEADLINE_FEATURES],
        "feature_labels": {**{c: l for c, l, _ in HEADLINE_FEATURES}, **GAP_FEATURE_LABELS},
        "percent_features": PERCENT_FEATURES,
        "gap_units": GAP_UNITS,
        "pro_leagues": pro_leagues,
        "pro_headline_features": [{"feature": c, "label": l, "invert": inv} for c, l, inv in PRO_HEADLINE_FEATURES],
        "champion_icons": champion_icons,
        "champion_splashes": champion_splashes,
        "reference": reference,
        "pro_reference": pro_reference,
        "pro_player_profiles": pro_player_profiles,
        "roles": roles_payload,
        "games": games,
    }

    parent = os.path.dirname(args.output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as outfile:
        json.dump(payload, outfile)
    size_kb = os.path.getsize(args.output_path) / 1024
    print(f"Dashboard data written to: {args.output_path} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    run(parse_args())
