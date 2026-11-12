# Rift Archetypes

**What kind of League of Legends player are you, apart from whether you won?**

Rift Archetypes pulls ~26k season-16 high-elo solo queue games (~260k player-games, Diamond → Challenger, NA + KR) from the Riot API and measures each player's *playstyle* while controlling for the champion they played and how their teammates did. The result is an interactive scouting report: your style traits, the archetypes you're closest to, and percentile comparisons against every reference tier and against LCK/LPL pros.

<!-- TODO: add a dashboard screenshot at docs/dashboard.png after the trait-axis dashboard lands, then uncomment:
![Scouting report dashboard](docs/dashboard.png)
-->

## Results

Raw match stats mostly measure *which champion you played* and *how the game went*. After the adjustment described below, both effects are largely gone, and a smaller but stable per-player signal remains.

| Season 16, per role (range across the 5 roles) | Raw stats | Adjusted | What it means |
|---|---:|---:|---|
| Champion's share of feature variance | 16–23% | **0.2–1.7%** | Style is measured relative to the champion you played. "Malphite tanks more than Jinx" is no longer a style difference. |
| Correlation with winning (absolute) | 0.19–0.24 | **0.04–0.11** | Your teammates' share of the outcome is removed. Your own contribution is kept on purpose. |
| Split-half reliability of player profiles | 0.51–0.56 | **0.41–0.47** | Half of a player's games still predicts the other half: there's a real, stable style signal to measure. |

Each number is the mean over 42 style features; players with 10+ games for reliability. Reproduce with `python src/method_diagnostics.py`, whose calculations are covered by hand-checked tests.

**Takeaways**

- **Most of what looks like playstyle in raw stats is champion and game state.** Removing both cuts the champion effect by 89–99% and the win correlation by 50–78% (depending on role), while keeping 75–84% of the player-level reliability.
- **Playstyle is a continuum, not a set of types.** K-means found weak cluster structure (silhouette 0.12–0.15) and collapsed every role into carry / farmer / low-resource. That's why the project is moving to continuous trait axes with soft archetype labels (see [Roadmap](#roadmap)).

## The hard part: separating playstyle from outcome

Naively clustering per-minute stats doesn't find playstyles. It finds *how the game went*. A stomp inflates damage, gold, KDA and objectives all at once, so the first version's "archetypes" had win rates of ~15% / ~60% / ~90%: losing, average, winning. Most of the project is about removing that confound without throwing away real style signal.

1. **Adjust for game context, then measure the residuals.** Per role, every feature is regressed on game-context covariates and style is measured on what's left over: "how did this game differ from a typical game in the same situation?" (`src/playstyle_space.py`)
2. **Control for teammates' lead, not the result.** Regressing on the win flag would also subtract out styles that *cause* wins. The adjustment controls only for your four teammates' lead over the four enemies outside your lane, so your own contribution stays in. That's why the win correlation drops to 0.04–0.11 rather than zero.
3. **Adjust for champion.** Champion fixed effects, with rare champions (under 30 games in a role) pooled into their Data Dragon class.
4. **Add early-game timeline features.** End-of-game totals are mostly *effect*. What a player does before 14:00 (CS diff @10, gold/XP diff @14, early trades, roams, warding) is mostly *cause*, so it's a cleaner style signal. These are adjusted for champion only, since end-of-game context comes after them. (`src/timeline_features.py`)
5. **Measure confidence.** Each trait gets a reliability from how much a single game reveals about a player (ICC) and how many games you've played (Spearman–Brown), shown as low / medium / high. (`src/style_profile.py`)

Comparisons follow a few fixed conventions:

- **Season-matched.** Baselines shift between seasons (bot-lane median CS/min ~7.4 in S15, ~8.1 in S16), so you're only ever compared within one season.
- **Averages vs. averages.** Your average is ranked against other players' averages (10+ games), not against single games, which have far more spread.
- **Percentiles, not raw differences.**
- **Percentiles use raw stats.** The adjustment is only for measuring style. For "how good is your gold/min", being ahead *is* the point.

## Roadmap

The first version grouped games with k-means. Even after the champion and teammate adjustments, every non-support role split mainly into carry / farmer / low-resource, with cluster win rates from 37% to 74%: a resource-share axis rather than a playstyle. So the product is moving from clusters to continuous trait axes.

**Done**
- [x] Champion adjustment, with rare champions pooled by class
- [x] Teammates'-lead adjustment in place of the win flag
- [x] Early-game timeline features, tested on 9 hand-checked games across all five roles
- [x] Trait-axis profiles, soft archetype labels ("72% Safe Farmer, 41% Vision-Focused Carry") and per-trait confidence
- [x] Method diagnostics (the [Results](#results) table)

**Next**
- [ ] Timeline backfill for the whole season, so early-game traits cover the full reference population
- [ ] Dashboard rewrite around trait axes, soft labels and confidence, replacing the cluster cards and k slider (and the dashboard's separate cluster refit)
- [ ] Champion pool as its own layer: your class mix and champion concentration vs. the reference tier
- [ ] Class-matched performance percentiles, so a Jinx main isn't compared against every ADC game, plus a "biggest gaps" list
- [ ] An explicit "insufficient data (n=X)" message wherever a comparison group is too small, instead of hiding the section

**Later**
- [ ] Pro team clustering from Oracle's Elixir data (Riot's API doesn't expose LCK/LEC matches)

## Testing

The accuracy suite encodes the analysis conventions against real games whose expected values were checked by hand (`tests/`): CS including jungle camps, per-minute normalization, season tagging, remake exclusion, outcome and champion adjustment, timeline features across all five roles, trait profiles and confidence, and the diagnostics behind the Results table.

```
pytest
```

The fixture games are real match payloads with every player identity replaced by a salted hash (`src/anonymize_fixtures.py`, enforced by `tests/test_fixture_privacy.py`).

## Quickstart

```
python -m venv venv
venv\Scripts\activate            # source venv/bin/activate on macOS/Linux
pip install -r requirements.txt
copy .env.example .env           # cp on macOS/Linux; then fill in RIOT_API_KEY and RIOT_IDS
```

- `RIOT_API_KEY`: a development key from the [Riot developer portal](https://developer.riotgames.com). These expire every 24h.
- `RIOT_IDS`: your account(s) as `gameName#tagLine`, comma-separated. Every script that takes `--riot-id` defaults to this.

## Pipeline

1. **Player pool:** `fetch_league_pool.py`, one file per tier and region
2. **Match IDs:** `fetch_match_ids.py`, deduped across the pool, always including your own accounts
3. **Match data:** `fetch_match_details.py` and `fetch_match_timelines.py`
4. **Features:** `build_features.py` and `build_timeline_features.py`
5. **Style model:** `build_clusters.py` fits the per-role adjustment; `method_diagnostics.py` checks it
6. **You:** `compare_self.py`, or `fetch_recent_self.py` to refresh just your own games
7. **Dashboard:** `export_dashboard_data.py`, then `build_dashboard.py`

The full pool pull takes a while on a dev key's rate limit. Every script, its options and the exact run order are in **[docs/pipeline.md](docs/pipeline.md)**.

---

## License and credits

Code: [MIT](LICENSE). Match data via the [Riot Games API](https://developer.riotgames.com); pro match data from [Oracle's Elixir](https://oracleselixir.com).

Rift Archetypes isn't endorsed by Riot Games and doesn't reflect the views or opinions of Riot Games or anyone officially involved in producing or managing Riot Games properties. Riot Games, and all associated properties are trademarks or registered trademarks of Riot Games, Inc.
