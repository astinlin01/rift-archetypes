"""
Accuracy tests for the self-vs-reference comparison (compare_self.py).

These use small synthetic frames instead of real games, because what's under
test is the comparison logic (season matching, percentile math, who counts as a
reference player), and synthetic values make the expected answers obvious.
"""

import pandas as pd
import pytest

import compare_self
from compare_self import MIN_REFERENCE_PLAYER_GAMES, build_comparison_rows

FEATURE = "cs_total_per_min"


@pytest.fixture(autouse=True)
def single_feature(monkeypatch):
    # Compare one feature only so the expected numbers stay readable.
    monkeypatch.setattr(compare_self, "FEATURE_COLUMNS", [FEATURE])


def player_games(puuid: str, value: float, n: int, season: int = 16, tier: str = "CHALLENGER") -> list[dict]:
    return [
        {"puuid": puuid, FEATURE: value, "season": season, "tier": tier, "teamPosition": "BOTTOM"}
        for _ in range(n)
    ]


# --- Percentiles ---------------------------------------------------------------

def test_percentile_ranks_against_player_averages():
    # 4 reference players averaging 6, 7, 8, 9 CS/min. You average 7.5.
    # You beat 2 of 4 players -> 50th percentile (scipy kind="mean").
    n = MIN_REFERENCE_PLAYER_GAMES
    ref = pd.DataFrame(
        player_games("a", 6.0, n) + player_games("b", 7.0, n) + player_games("c", 8.0, n) + player_games("d", 9.0, n)
    )
    me = pd.DataFrame(player_games("me", 7.5, 5, tier="UNKNOWN"))

    [row] = build_comparison_rows(me, ref, "BOTTOM")

    assert row["your_avg"] == pytest.approx(7.5)
    assert row["reference_avg"] == pytest.approx(7.5)  # mean of the 4 player averages
    assert row["your_percentile_in_reference"] == pytest.approx(50.0)


def test_percentile_is_0_to_100_not_basis_points():
    n = MIN_REFERENCE_PLAYER_GAMES
    ref = pd.DataFrame(player_games("a", 6.0, n) + player_games("b", 7.0, n))
    me = pd.DataFrame(player_games("me", 99.0, 5, tier="UNKNOWN"))

    [row] = build_comparison_rows(me, ref, "BOTTOM")

    assert row["your_percentile_in_reference"] == pytest.approx(100.0)


def test_occasional_players_excluded_from_reference():
    # Player "x" has only 1 game at 20 CS/min. Their single-game average would drag
    # the reference up, so they must not count. Remaining players: 6 and 8 -> ref avg 7.
    n = MIN_REFERENCE_PLAYER_GAMES
    ref = pd.DataFrame(player_games("a", 6.0, n) + player_games("b", 8.0, n) + player_games("x", 20.0, 1))
    me = pd.DataFrame(player_games("me", 7.0, 5, tier="UNKNOWN"))

    [row] = build_comparison_rows(me, ref, "BOTTOM")

    assert row["reference_avg"] == pytest.approx(7.0)
    assert row["your_percentile_in_reference"] == pytest.approx(50.0)


# --- Season matching -------------------------------------------------------------

def test_comparison_is_season_matched():
    # Reference pool is all season 16. You have season 15 games at 1.0 CS/min and
    # season 16 games at 8.0. Only the season 16 games may be compared.
    n = MIN_REFERENCE_PLAYER_GAMES
    df = pd.DataFrame(
        player_games("a", 7.0, n)
        + player_games("b", 9.0, n)
        + player_games("me", 1.0, 5, season=15, tier="UNKNOWN")
        + player_games("me", 8.0, 5, season=16, tier="UNKNOWN")
    )

    filtered = compare_self.filter_to_current_season(df)

    assert set(filtered["season"]) == {16}
    me = filtered[filtered["puuid"] == "me"]
    assert me[FEATURE].mean() == pytest.approx(8.0)
