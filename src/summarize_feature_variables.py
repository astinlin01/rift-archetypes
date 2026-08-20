"""
Group and summarize potential analysis variables from the EDA markdown report.

Run from repo root:
    python src/summarize_feature_variables.py
"""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable


GROUP_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("Identity and context", ("championName", "championId", "individualPosition", "teamPosition", "role", "lane")),
    ("Objectives and structures", ("baron", "dragon", "elder", "herald", "rift", "turret", "inhibitor", "nexus", "epicMonster", "voidMonster")),
    ("Communication", ("ping", "fistBump")),
    ("Combat and takedowns", ("kill", "death", "assist", "takedown", "multikill", "kda", "immobil", "crowdControl", "spell", "skillshot", "damage")),
    ("Economy and farming", ("gold", "minion", "cs", "jungleMonster", "monsterKills", "experience", "champExperience")),
    ("Vision and map control", ("ward", "vision", "scuttle", "river", "bush")),
    ("Support and utility", ("heal", "shield", "saveAlly", "cleanse", "support", "ally", "buff", "teleport")),
    ("Communication", ("ping", "fistBump")),
    ("Progression and items", ("item", "legendary", "level", "consumable", "championTransform")),
    ("Pace and timing", ("time", "duration", "minute", "early", "late", "fast", "gameLength", "gameCreation", "Timestamp")),
]

LEAKAGE_TERMS = (
    "win",
    "gameEnded",
    "surrender",
    "teamWin",
    "winner",
    "endOfGame",
)
IDENTIFIER_TERMS = (
    "puuid",
    "summonerId",
    "summonerName",
    "matchId",
    "gameId",
    "accountId",
    "profileIcon",
)


@dataclass
class Variable:
    field: str
    present: int
    occurrences_per_record: float
    missing_pct: float | None
    types: str
    numeric_count: int | None
    mean: float | None
    p50: float | None
    p90: float | None
    max_value: float | None
    top_categories: str

    @property
    def leaf_name(self) -> str:
        return self.field.rsplit(".", 1)[-1].replace("[]", "")

    @property
    def is_numeric(self) -> bool:
        return self.numeric_count is not None and self.numeric_count > 0

    @property
    def is_constant(self) -> bool:
        return self.is_numeric and self.mean == self.p50 == self.max_value

    @property
    def is_sparse(self) -> bool:
        return self.missing_pct is not None and self.missing_pct >= 50.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize feature candidates from eda_report.md.")
    parser.add_argument("--report-path", default="data/raw/eda_report.md")
    parser.add_argument("--output-path", default="data/raw/feature_variable_summary.md")
    parser.add_argument("--min-mean", type=float, default=0.0, help="Optional minimum numeric mean for candidate output.")
    return parser.parse_args()


def parse_number(value: str) -> float | None:
    if value.strip() in {"", "None", "-"}:
        return None
    try:
        return float(value.strip())
    except ValueError:
        return None


def parse_numeric_summary(value: str) -> tuple[int | None, float | None, float | None, float | None, float | None]:
    count_match = re.search(r"n=(\d+)", value)
    mean_match = re.search(r"mean=([-+]?\d+(?:\.\d+)?)", value)
    p50_match = re.search(r"p50=([-+]?\d+(?:\.\d+)?)", value)
    p90_match = re.search(r"p90=([-+]?\d+(?:\.\d+)?)", value)
    max_match = re.search(r"max=([-+]?\d+(?:\.\d+)?)", value)
    return (
        int(count_match.group(1)) if count_match else None,
        parse_number(mean_match.group(1)) if mean_match else None,
        parse_number(p50_match.group(1)) if p50_match else None,
        parse_number(p90_match.group(1)) if p90_match else None,
        parse_number(max_match.group(1)) if max_match else None,
    )


def split_table_row(line: str) -> list[str]:
    return [part.strip() for part in line.strip().strip("|").split("|")]


def iter_variables(report_path: str) -> Iterable[Variable]:
    in_field_profile = False
    with open(report_path, "r", encoding="utf-8") as infile:
        for raw_line in infile:
            line = raw_line.strip()
            if line == "### Field Profile":
                in_field_profile = True
                continue
            if not in_field_profile or not line.startswith("|") or line.startswith("|---"):
                continue

            parts = split_table_row(line)
            if len(parts) != 8 or parts[0] == "field":
                continue

            numeric_count, mean, p50, p90, max_value = parse_numeric_summary(parts[5])
            yield Variable(
                field=parts[0],
                present=int(parts[1]),
                occurrences_per_record=float(parts[2]),
                missing_pct=parse_number(parts[3]),
                types=parts[4],
                numeric_count=numeric_count,
                mean=mean,
                p50=p50,
                p90=p90,
                max_value=max_value,
                top_categories=parts[6],
            )


def classify(variable: Variable) -> str:
    field = variable.field
    leaf_name = variable.leaf_name

    if any(term.lower() in field.lower() for term in IDENTIFIER_TERMS):
        return "Identifiers and joins"
    if any(term.lower() in field.lower() for term in LEAKAGE_TERMS):
        return "Outcome or leakage fields"
    if variable.is_constant:
        return "Constant or unusable fields"

    for group_name, terms in GROUP_RULES:
        if any(term.lower() in leaf_name.lower() for term in terms):
            return group_name

    if variable.is_sparse:
        return "Sparse situational fields"
    if variable.is_numeric:
        return "Other numeric variables"
    return "Other categorical variables"


def status(variable: Variable) -> str:
    if variable.field.startswith("info.participants[]."):
        if any(term.lower() in variable.field.lower() for term in IDENTIFIER_TERMS):
            return "exclude: identifier"
        if any(term.lower() in variable.field.lower() for term in LEAKAGE_TERMS):
            return "exclude: target leakage"
        if variable.is_constant:
            return "exclude: constant"
        if variable.is_sparse:
            return "review: sparse"
        return "candidate"
    return "context only"


def display_number(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def render_report(variables: list[Variable], report_path: str, min_mean: float) -> str:
    groups: dict[str, list[Variable]] = defaultdict(list)
    for variable in variables:
        groups[classify(variable)].append(variable)

    candidate_variables = [
        variable
        for variable in variables
        if status(variable) == "candidate"
        and (variable.mean is None or variable.mean >= min_mean)
    ]
    status_counts = Counter(status(variable) for variable in variables)

    lines = [
        "# Potential Feature Variable Summary",
        "",
        f"Source: `{report_path}`",
        f"Generated at: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Recommended Starting Point",
        "",
        "Use participant-level rows as the modeling unit. Start with candidate variables below, normalize count-like variables by game duration where appropriate, and avoid mixing outcome fields into unsupervised playstyle clustering.",
        "",
        f"- Total variables parsed: {len(variables)}",
        f"- Candidate participant variables passing min_mean={min_mean:.3f}: {len(candidate_variables)}",
        f"- Status counts: {', '.join(f'{key}={value}' for key, value in sorted(status_counts.items()))}",
        "",
        "## Group Summary",
        "",
        "| group | variables | numeric | candidate | review/exclude |",
        "|---|---:|---:|---:|---:|",
    ]

    for group_name in sorted(groups):
        group_variables = groups[group_name]
        numeric_count = sum(variable.is_numeric for variable in group_variables)
        candidate_count = sum(status(variable) == "candidate" for variable in group_variables)
        review_count = len(group_variables) - candidate_count
        lines.append(f"| {group_name} | {len(group_variables)} | {numeric_count} | {candidate_count} | {review_count} |")

    lines.extend(["", "## Candidate Variables", ""])
    lines.append("| group | field | present | missing% | mean | p50 | p90 | max | notes |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")

    for group_name in sorted(groups):
        for variable in sorted(groups[group_name], key=lambda item: item.field.lower()):
            if variable not in candidate_variables:
                continue
            missing = "-" if variable.missing_pct is None else f"{variable.missing_pct:.2f}"
            lines.append(
                f"| {group_name} | `{variable.field}` | {variable.present} | {missing} | "
                f"{display_number(variable.mean)} | {display_number(variable.p50)} | "
                f"{display_number(variable.p90)} | {display_number(variable.max_value)} | {variable.types} |"
            )

    lines.extend(["", "## Review and Exclusions", ""])
    lines.append("| group | field | status | present | missing% | mean | top categories |")
    lines.append("|---|---|---|---:|---:|---:|---|")

    for group_name in sorted(groups):
        for variable in sorted(groups[group_name], key=lambda item: item.field.lower()):
            if variable in candidate_variables:
                continue
            missing = "-" if variable.missing_pct is None else f"{variable.missing_pct:.2f}"
            lines.append(
                f"| {group_name} | `{variable.field}` | {status(variable)} | {variable.present} | "
                f"{missing} | {display_number(variable.mean)} | {variable.top_categories} |"
            )

    lines.extend(
        [
            "",
            "## Feature Engineering Notes",
            "",
            "- Keep `championName`, `individualPosition`, and item IDs as context or grouping variables; raw champion and item IDs are not naturally continuous features.",
            "- Prefer rates such as `damagePerMinute`, `goldPerMinute`, `visionScorePerMinute`, kill participation, and objective participation over raw totals when comparing games of different lengths.",
            "- Treat fields with missingness above 50% as situational indicators. Impute explicitly or convert them to a presence flag plus a value.",
            "- Remove constant fields and the zero-filled challenge fields before scaling or clustering.",
            "- Exclude wins, surrender, and game-end fields when the goal is playstyle clustering; keep them separately for downstream validation.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    if not os.path.exists(args.report_path):
        raise FileNotFoundError(f"EDA report not found: {args.report_path}")

    variables = list(iter_variables(args.report_path))
    report = render_report(variables, args.report_path, args.min_mean)

    parent = os.path.dirname(args.output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as outfile:
        outfile.write(report)

    print(f"Parsed variables: {len(variables)}")
    print(f"Feature summary written to: {args.output_path}")


if __name__ == "__main__":
    run(parse_args())