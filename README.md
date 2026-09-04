# Rift Archetypes

Rift Archetypes collects League of Legends solo queue data and clusters playstyles per role using unsupervised learning.

Pipeline:
1. Build a player pool from Riot league endpoints.
2. Fetch and dedupe match IDs across that pool.
3. Fetch full match payloads.
4. Build a per-participant feature table.
5. Fit per-role playstyle clusters.
6. Compare your own matches against a reference tier.

## Scripts

### src/fetch_league_pool.py
Builds the raw player pool file.

Default behavior:
- `--tier DIAMOND`
- `--division II`
- `--num-pages 1`
- Output: `data/raw/league_entries.json`
- Overwrites output by default.

Key parameters:
- `--tier`
  - If `CHALLENGER`, `GRANDMASTER`, or `MASTER`: uses top-league endpoint (no division/page semantics).
  - Otherwise: uses paged entries endpoint and requires tier/division context.
- `--division`
  - Used only for non-top tiers.
- `--num-pages`
  - Used only for non-top tiers.
- `--positive-winrate-only`
  - Keeps only entries where `wins > losses`.
- `--append`
  - Merges with existing output by puuid instead of overwriting.
- `--output-path`
  - Destination JSON file.

Examples:
- Diamond II default slice:
  - `python src/fetch_league_pool.py`
- Diamond II with more pages:
  - `python src/fetch_league_pool.py --tier DIAMOND --division II --num-pages 5`
- Challenger only, positive winrate:
  - `python src/fetch_league_pool.py --tier CHALLENGER --positive-winrate-only --output-path data/raw/league_entries_challenger_poswr.json`

### src/fetch_match_ids.py
Reads league pool entries and collects deduped match IDs.

Outputs:
- `data/raw/match_ids.ndjson`
- `data/raw/checkpoints/match_ids_checkpoint.json`
- `data/raw/failures/match_ids_failures.ndjson`

Important behavior:
- Always includes configured Riot IDs (currently your three accounts) even if they are not present in the league pool file.
- Resolves Riot IDs to puuid via account-v1.
- Dedupes match IDs globally.

### src/fetch_match_details.py
Reads `match_ids.ndjson` and fetches full match payloads.

Outputs:
- `data/raw/matches.ndjson`
- `data/raw/checkpoints/match_details_checkpoint.json`
- `data/raw/failures/match_details_failures.ndjson`

### src/fetch_recent_self.py
Pulls your own most recent ranked matches directly (resolve Riot ID -> recent match IDs -> match details) without scanning the full league pool. Appends into the same `match_ids.ndjson`/`matches.ndjson` files, so a normal `build_features.py` run picks the new games up.

- Run: `python src/fetch_recent_self.py --riot-id "river3#na1" --count 20`

### src/eda_overview.py
Profiles the collected JSON/NDJSON files and writes a markdown schema and descriptive-statistics report.

- Report: `data/raw/eda_report.md`
- Example: `python src/eda_overview.py --max-records 500`

### src/summarize_feature_variables.py
Reads `eda_report.md`, groups potential participant-level variables by analysis theme, and flags identifiers, target leakage, constants, and sparse fields.

- Report: `data/raw/feature_variable_summary.md`
- Run: `python src/summarize_feature_variables.py`

### src/feature_config.py
Curated (hand-reviewed, not auto-parsed) list of participant fields used for clustering, grouped by normalization strategy (already-normalized, per-minute rate, raw count). Update this file when adding/removing candidate features.

### src/build_features.py
Builds the per-participant feature table used for clustering.

- Drops non-solo-role rows (keeps `TOP`/`JUNGLE`/`MIDDLE`/`BOTTOM`/`UTILITY` only) and remakes/early-surrenders shorter than `--min-game-length-seconds` (default 5 min).
- Applies per-minute normalization from `feature_config.py`.
- Attaches source tier per puuid by matching against all `data/raw/league_entries*.json` pool files.
- Output: `data/processed/participant_features.parquet`
- Run: `python src/build_features.py`

### src/build_clusters.py
Fits a separate `StandardScaler` → `PCA` → `KMeans` pipeline per role (role is the dominant driver of raw stats, so clustering is done within-role rather than globally). Sweeps k and reports silhouette score to pick the cluster count per role (override with `--k-override ROLE=K`).

- Models: `models/<role>_cluster.joblib`
- Cluster assignments: `data/processed/participant_clusters.parquet`
- Centroid report: `data/processed/cluster_centroid_report.md`
- Run: `python src/build_clusters.py`

### src/compare_self.py
Assigns your own matches to the fitted per-role clusters and compares your per-feature averages/percentiles against a reference tier (default Challenger).

- Run: `python src/compare_self.py --riot-id "river3#na1"` (or `--puuid <puuid>`, repeatable)
- Output: `data/processed/self_comparison.csv`

### notebooks/01_explore_clusters.ipynb
Exploration only (not required for the pipeline to run) -- PCA scatter plots per role, centroid bar charts, and tier mix per cluster, reading from `data/processed/` and `models/`.

### src/export_dashboard_data.py + src/build_dashboard.py
Builds a self-contained HTML "scouting report" dashboard for one player -- percentile comparison vs. a reference tier, cluster archetype cards, a PCA scatter with your games highlighted, and a recent-games table. `export_dashboard_data.py` writes `data/processed/dashboard_data.json`; `build_dashboard.py` merges it into `dashboard/template.html` to produce `dashboard/river3_scouting_report.html`, which can be published as a Claude Artifact.

- Run: `python src/export_dashboard_data.py --riot-id "river3#na1"` then `python src/build_dashboard.py`

## Typical run order

1. Build pool (one file per tier -- `build_features.py` infers tier from the filename for apex tiers, so don't merge them into one file)
- `python src/fetch_league_pool.py --tier DIAMOND --division II`
- `python src/fetch_league_pool.py --tier MASTER --output-path data/raw/league_entries_master.json`
- `python src/fetch_league_pool.py --tier GRANDMASTER --output-path data/raw/league_entries_grandmaster.json`
- `python src/fetch_league_pool.py --tier CHALLENGER --output-path data/raw/league_entries_challenger.json`

2. Collect match IDs
- `python src/fetch_match_ids.py --resume`

3. Fetch match payloads
- `python src/fetch_match_details.py --resume`

4. Build features
- `python src/build_features.py`

5. Fit clusters
- `python src/build_clusters.py`

6. Compare yourself
- `python src/compare_self.py --riot-id "river3#na1"`

7. Build the dashboard
- `python src/export_dashboard_data.py --riot-id "river3#na1"`
- `python src/build_dashboard.py`

To refresh with just your own newest games (skips re-scanning the full pool):
- `python src/fetch_recent_self.py --riot-id "river3#na1"`
- then re-run steps 4, 6, 7 (no need to refit clusters unless the population itself changed materially)

## Not built yet

- Timeline-based features (gold/xp diff curves, item timing) -- a natural V2 on top of the current aggregate-stat features.
- Pro team clustering -- Riot's public API doesn't expose LCK/LEC/etc. match data; planned to use an external source like Oracle's Elixir.

## Notes

- Riot developer keys expire frequently. If requests suddenly fail with 401, regenerate key and update `.env`.
- Raw data files under `data/raw/` are intentionally gitignored, as are `data/processed/` and `models/` (both regenerable from `data/raw/`).
