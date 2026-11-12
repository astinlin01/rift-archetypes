"""
Method diagnostics: does the adjustment do what the README claims?

For each role (season-matched), on the style features (CLUSTER_FEATURE_COLUMNS), raw vs
adjusted (playstyle_space's teammates-lead + game-length + champion adjustment):

- champion share of variance  eta^2 = SS_between_champions / SS_total. "How much of this
                              feature is just which champion was played." Should drop.
- |correlation with win|      "How much of this feature is just whether the game went
                              well." Should drop, but not to zero: the adjustment removes
                              the teammates' share of the outcome and keeps your own.
- split-half reliability      each player's games alternate into two halves; correlate the
                              half means across players (10+ games). "Is there a stable
                              per-player signal?" Should stay well above 0. Raw
                              correlation, not Spearman-Brown corrected.

Each measure is averaged over features, then reported per role and as a range across roles.

Run from repo root:
    python src/method_diagnostics.py
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from champion_classes import load_champion_classes
from feature_config import VALID_ROLES, current_season
from playstyle_space import (
    CLUSTER_FEATURE_COLUMNS,
    add_game_state_columns,
    apply_outcome_adjuster,
    fit_outcome_adjuster,
)

MIN_PLAYER_GAMES = 10
MEASURES = ("champion_variance_share", "abs_win_correlation", "split_half_reliability")


def champion_variance_share(values: pd.DataFrame, champions: pd.Series) -> pd.Series:
    """Per feature: share of variance between champions (eta squared). Missing values dropped."""
    out = {}
    for column in values.columns:
        mask = values[column].notna().to_numpy()
        v = values[column].to_numpy(dtype=float)[mask]
        groups = champions.to_numpy()[mask]
        ss_total = ((v - v.mean()) ** 2).sum()
        group_means = pd.Series(v).groupby(groups).transform("mean").to_numpy()
        ss_between = ((group_means - v.mean()) ** 2).sum()
        out[column] = ss_between / ss_total if ss_total > 0 else np.nan
    return pd.Series(out)


def win_correlation(values: pd.DataFrame, win: pd.Series) -> pd.Series:
    """Per feature: |Pearson correlation| with the win flag. Missing values dropped."""
    w = pd.Series(win.to_numpy(dtype=float), index=values.index)
    return values.apply(lambda col: abs(col.corr(w)))


def split_half_reliability(values: pd.DataFrame, puuids: pd.Series, min_games: int = MIN_PLAYER_GAMES) -> pd.Series:
    """Per feature: correlation across players of their two half-means (games alternate in
    row order, so sort rows by time first). Players with fewer than min_games valid games
    for that feature are excluded."""
    out = {}
    players = pd.Series(puuids.to_numpy(), index=values.index)
    for column in values.columns:
        df = pd.DataFrame({"v": values[column], "p": players}).dropna()
        counts = df["p"].map(df["p"].value_counts())
        df = df[counts >= min_games]
        if df["p"].nunique() < 3:
            out[column] = np.nan
            continue
        half = df.groupby("p").cumcount() % 2
        means = df.groupby([df["p"], half])["v"].mean().unstack()
        out[column] = means[0].corr(means[1])
    return pd.Series(out)


def summarize_role(values: pd.DataFrame, rows: pd.DataFrame, min_games: int = MIN_PLAYER_GAMES) -> dict:
    """Each measure averaged over features (ignoring features where it's undefined; NaN if
    it's undefined for all of them, e.g. fewer than 3 players with enough games)."""
    per_feature = {
        "champion_variance_share": champion_variance_share(values, rows["championName"]),
        "abs_win_correlation": win_correlation(values, rows["win"]),
        "split_half_reliability": split_half_reliability(values, rows["puuid"], min_games),
    }
    return {name: float(s.mean()) if s.notna().any() else np.nan for name, s in per_feature.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raw vs adjusted method diagnostics per role.")
    parser.add_argument("--features-path", default="data/processed/participant_features.parquet")
    parser.add_argument("--report-path", default="data/processed/method_diagnostics.md")
    parser.add_argument("--min-games", type=int, default=MIN_PLAYER_GAMES)
    return parser.parse_args()


def _range(results: dict, version: str, measure: str) -> str:
    vals = [results[role][version][measure] for role in results]
    return f"{min(vals):.2f}–{max(vals):.2f}"


def run(args: argparse.Namespace) -> None:
    df = add_game_state_columns(pd.read_parquet(args.features_path))
    season = current_season(df)
    if season is not None:
        df = df[df["season"] == season]
    classes = load_champion_classes()

    results: dict[str, dict] = {}
    for role in VALID_ROLES:
        role_df = df[df["teamPosition"] == role].sort_values("gameCreation").reset_index(drop=True)
        if role_df.empty:
            continue
        raw = role_df[CLUSTER_FEATURE_COLUMNS]
        adjusted = apply_outcome_adjuster(role_df, fit_outcome_adjuster(role_df, classes))
        results[role] = {
            "raw": summarize_role(raw, role_df, args.min_games),
            "adjusted": summarize_role(adjusted, role_df, args.min_games),
            "n_games": len(role_df),
            "n_players": int((role_df["puuid"].value_counts() >= args.min_games).sum()),
        }
        r = results[role]
        print(f"[{role}] n={r['n_games']}, players with {args.min_games}+ games={r['n_players']}  "
              + "  ".join(f"{m}: {r['raw'][m]:.3f} -> {r['adjusted'][m]:.3f}" for m in MEASURES))

    lines = [
        f"# Method diagnostics (season {season})",
        "",
        f"Style features ({len(CLUSTER_FEATURE_COLUMNS)}), each measure averaged over features. "
        "Adjusted = teammates' lead + game length + champion (playstyle_space).",
        "",
        "| role | games | players (10+) | champion share raw | adjusted | abs(corr win) raw | adjusted | split-half raw | adjusted |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for role, r in results.items():
        cells = [f"{r[v][m]:.3f}" for m in MEASURES for v in ("raw", "adjusted")]
        lines.append(f"| {role} | {r['n_games']} | {r['n_players']} | " + " | ".join(cells) + " |")
    lines += ["", "Range across roles:", ""]
    for m in MEASURES:
        lines.append(f"- {m}: raw {_range(results, 'raw', m)}, adjusted {_range(results, 'adjusted', m)}")
    os.makedirs(os.path.dirname(args.report_path) or ".", exist_ok=True)
    with open(args.report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[-4:]))
    print(f"Report written to {args.report_path}")


if __name__ == "__main__":
    run(parse_args())
