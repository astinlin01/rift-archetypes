"""
Curated feature configuration for playstyle clustering.

Hand-reviewed against data/raw/feature_variable_summary.md rather than
re-parsed from it at runtime, so the feature set is auditable and versioned
in code. Update this file (and rerun eda_overview.py / summarize_feature_variables.py
to sanity-check) whenever new candidate fields should be considered.
"""

from __future__ import annotations

from dataclasses import dataclass

VALID_ROLES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")

# Identity/context columns carried through the feature table but never fed
# into scaling/clustering.
CONTEXT_COLUMNS = (
    "matchId",
    "puuid",
    "championName",
    "teamPosition",
    "win",
    "queueId",
    "gameLength",
    "tier",
    "region",
    "gameCreation",
    "patch",
    "season",
)


@dataclass(frozen=True)
class FeatureField:
    # Dotted path under a single participant dict, e.g. "challenges.kda" or "kills".
    path: str
    # Output column name.
    name: str
    # "raw": use as-is. "per_minute": divide by (gameLength / 60).
    normalize: str = "raw"


# Fields already expressed as a rate/ratio/normalized score by Riot -- use as-is.
ALREADY_NORMALIZED_FIELDS = [
    FeatureField("challenges.damagePerMinute", "damage_per_minute"),
    FeatureField("challenges.goldPerMinute", "gold_per_minute"),
    FeatureField("challenges.visionScorePerMinute", "vision_score_per_minute"),
    FeatureField("challenges.kda", "kda"),
    FeatureField("challenges.killParticipation", "kill_participation"),
    FeatureField("challenges.teamDamagePercentage", "team_damage_pct"),
    FeatureField("challenges.damageTakenOnTeamPercentage", "damage_taken_on_team_pct"),
]

# Raw counts normalized to a per-minute rate so games of different lengths
# are comparable.
PER_MINUTE_FIELDS = [
    FeatureField("kills", "kills_per_min", "per_minute"),
    FeatureField("deaths", "deaths_per_min", "per_minute"),
    FeatureField("assists", "assists_per_min", "per_minute"),
    FeatureField("totalDamageDealtToChampions", "damage_to_champions_per_min", "per_minute"),
    FeatureField("damageDealtToObjectives", "damage_to_objectives_per_min", "per_minute"),
    FeatureField("damageDealtToBuildings", "damage_to_buildings_per_min", "per_minute"),
    FeatureField("damageSelfMitigated", "damage_self_mitigated_per_min", "per_minute"),
    FeatureField("totalDamageTaken", "damage_taken_per_min", "per_minute"),
    FeatureField("totalHeal", "total_heal_per_min", "per_minute"),
    FeatureField("totalHealsOnTeammates", "heal_on_teammates_per_min", "per_minute"),
    FeatureField("totalDamageShieldedOnTeammates", "shield_on_teammates_per_min", "per_minute"),
    FeatureField("timeCCingOthers", "cc_time_per_min", "per_minute"),
    FeatureField("totalMinionsKilled", "cs_per_min", "per_minute"),
    FeatureField("neutralMinionsKilled", "neutral_cs_per_min", "per_minute"),
    FeatureField("goldEarned", "gold_earned_per_min", "per_minute"),
    FeatureField("visionScore", "vision_score_per_min_raw", "per_minute"),
    FeatureField("wardsPlaced", "wards_placed_per_min", "per_minute"),
    FeatureField("wardsKilled", "wards_killed_per_min", "per_minute"),
    FeatureField("challenges.controlWardsPlaced", "control_wards_per_min", "per_minute"),
    FeatureField("challenges.stealthWardsPlaced", "stealth_wards_per_min", "per_minute"),
    FeatureField("challenges.turretTakedowns", "turret_takedowns_per_min", "per_minute"),
    FeatureField("challenges.dragonTakedowns", "dragon_takedowns_per_min", "per_minute"),
    FeatureField("challenges.baronTakedowns", "baron_takedowns_per_min", "per_minute"),
    FeatureField("challenges.abilityUses", "ability_uses_per_min", "per_minute"),
    FeatureField("challenges.skillshotsHit", "skillshots_hit_per_min", "per_minute"),
    FeatureField("challenges.skillshotsDodged", "skillshots_dodged_per_min", "per_minute"),
    FeatureField("challenges.wardTakedowns", "ward_takedowns_per_min", "per_minute"),
]

# Small-count / already-bounded stats left as raw totals per game.
RAW_COUNT_FIELDS = [
    FeatureField("challenges.soloKills", "solo_kills"),
    FeatureField("challenges.killingSprees", "killing_sprees"),
    FeatureField("largestKillingSpree", "largest_killing_spree"),
    FeatureField("largestMultiKill", "largest_multikill"),
    FeatureField("challenges.multikills", "multikills"),
    FeatureField("champLevel", "champ_level"),
    FeatureField("challenges.maxKillDeficit", "max_kill_deficit"),
    FeatureField("challenges.buffsStolen", "buffs_stolen"),
    FeatureField("challenges.epicMonsterSteals", "epic_monster_steals"),
    FeatureField("objectivesStolenAssists", "objectives_stolen_assists"),
    FeatureField("challenges.laneMinionsFirst10Minutes", "lane_minions_first_10"),
    FeatureField("challenges.maxCsAdvantageOnLaneOpponent", "max_cs_advantage_lane_opp"),
    FeatureField("challenges.maxLevelLeadLaneOpponent", "max_level_lead_lane_opp"),
    FeatureField("challenges.turretPlatesTaken", "turret_plates_taken"),
]

ALL_FEATURE_FIELDS: list[FeatureField] = (
    ALREADY_NORMALIZED_FIELDS + PER_MINUTE_FIELDS + RAW_COUNT_FIELDS
)

# Features computed from other features after extraction (build_features.add_derived_features).
# cs_total_per_min is the "CS" number op.gg / the client show (lane minions + jungle
# camps) -- cs_per_min alone is minions only and reads ~0.3/min low for an ADC.
DERIVED_FEATURE_COLUMNS = ["cs_total_per_min"]

FEATURE_COLUMN_NAMES = [field.name for field in ALL_FEATURE_FIELDS] + DERIVED_FEATURE_COLUMNS


def add_derived_features(df):
    """Compute DERIVED_FEATURE_COLUMNS in place-safe fashion (returns a new frame)."""
    df = df.copy()
    df["cs_total_per_min"] = df["cs_per_min"].fillna(0) + df["neutral_cs_per_min"].fillna(0)
    return df


def current_season(df) -> int | None:
    """The season most of the tier-labeled (pool) population was played in.

    Stat baselines move a lot between seasons (bot-lane median CS/min went from ~7.4 in
    season 15 to ~8.1 in season 16), so every comparison is restricted to this season.
    """
    if "season" not in df.columns:
        return None
    pool = df[df["tier"] != "UNKNOWN"]["season"].dropna()
    if pool.empty:
        pool = df["season"].dropna()
    return int(pool.mode().iloc[0]) if not pool.empty else None


def season_from_patch(patch: str | None) -> int | None:
    """'16.17' -> 16. Riot's major version number is the season (14 = 2024, 16 = 2026)."""
    if not patch:
        return None
    try:
        return int(str(patch).split(".")[0])
    except ValueError:
        return None


def get_nested(participant: dict, dotted_path: str):
    value = participant
    for key in dotted_path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value
