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
Reads `match_ids.ndjson` and fetches full match payloads. Auto-routes each match ID to the correct regional cluster based on its platform prefix (`KR_...` -> `asia`, `NA1_...` -> `americas`, etc.) -- match-v5 lookups 404 through the wrong cluster, so a single run safely handles a `match_ids.ndjson` that mixes regions. `--platform` is only a fallback for an unrecognized prefix.

Outputs:
- `data/raw/matches.ndjson`
- `data/raw/checkpoints/match_details_checkpoint.json`
- `data/raw/failures/match_details_failures.ndjson`

### src/fetch_recent_self.py
Pulls your own ranked match history directly (resolve Riot ID -> paginate match-v5 -> match details) without scanning the full league pool. By default it paginates until it runs out of history (100 IDs per call, the API max); pass `--max-games` to cap it. Appends into the same `match_ids.ndjson`/`matches.ndjson` files, so a normal `build_features.py` run picks the new games up. Safe to re-run -- already-fetched IDs/details are skipped.

- Run: `python src/fetch_recent_self.py --riot-id "river3#na1"` (all history) or add `--max-games 50`

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
- Attaches source tier **and region** per puuid by matching against all `data/raw/league_entries*.json` pool files (`fetch_league_pool.py` tags every saved row with `platform`; files predating that default to NA1).
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

## Notebooks

All scripts in `src/` are runnable interactively from `notebooks/` -- each notebook cell
simulates the script's CLI invocation (`sys.argv` + `parse_args()`), so the notebooks stay in
sync with the scripts automatically rather than duplicating logic.

- **notebooks/00_pull_data.ipynb** -- pull your own recent/full match history (wraps `fetch_recent_self.py`), rebuild the feature table, and optionally refresh the full ranked pool.
- **notebooks/01_explore_clusters.ipynb** -- PCA scatter plots per role, centroid bar charts, and tier mix per cluster, reading from `data/processed/` and `models/`.
- **notebooks/02_self_analysis.ipynb** -- interactive version of `compare_self.py`: a styled percentile table, a percentile bar chart, cluster distribution, and a per-game stat trend over time.
- **notebooks/03_generate_dashboard.ipynb** -- runs `export_dashboard_data.py` + `build_dashboard.py` and sanity-checks the result. Publishing the output as a Claude Artifact still has to happen from a Claude Code conversation.

### src/export_dashboard_data.py + src/build_dashboard.py
Builds a self-contained, **interactive** HTML "scouting report" dashboard. `export_dashboard_data.py` precomputes everything up front -- for every configured account, every role, every reference tier, and cluster counts k=3-6 (percentile reference curves via percentile breakpoints, PCA scatter positions + per-point cluster assignments, PCA axis explainability captions) -- so the published dashboard filters entirely client-side with no server round trip. `build_dashboard.py` merges the resulting `data/processed/dashboard_data.json` into `dashboard/template.html` to produce `dashboard/river3_scouting_report.html`, which can be published as a Claude Artifact.

Dashboard filters:
- **Account** -- switch between configured Riot IDs (e.g. main + smurf) entirely in the browser.
- **Game window** -- last 20 / last 50 / all time.
- **Role** -- any role you have games in, or "All Roles" for an overall (non-role-specific) view -- the percentile chart still works there (aggregated across roles), but the archetype cards and cluster-space scatter require picking a specific role, since clustering is fit separately per role.
- **Champion** -- narrow to one champion within the selected role, or all champions.
- **Region** -- which platform's ranked population to compare against (NA, KR, ...; only regions with enough sample size for the selected role are offered).
- **Compare vs.** -- reference tier within the selected region, ordered Challenger > Grandmaster > Master > Diamond (whichever tiers have enough sample size).
- **Playstyles (k)** -- 3 to 6 clusters; cards, scatter colors, legend, and table archetypes all update together.

Note: clustering (archetype cards, cluster-space scatter) is fit on the **combined** population across all regions for a role -- it isn't split by region. Only the percentile-comparison reference curves are region-specific.

Each archetype card shows two comparisons against the population: your win rate within that cluster, and your *share* of games landing in it (over/under-represented vs. the population's own distribution across clusters).

Champion icons are fetched from Data Dragon at export time, cached under `data/raw/champion_icon_cache/`, and embedded as base64 data URIs in the dashboard JSON -- a published Artifact's CSP blocks image requests to external hosts, so hotlinking the CDN directly wouldn't render.

Note: cluster counts here are fit fresh per k on the same per-role PCA space purely for the dashboard's slider -- they don't change or replace the "official" per-role models in `models/` that `compare_self.py` uses (those keep whatever k `build_clusters.py`'s silhouette search picked).

- Run: `python src/export_dashboard_data.py --riot-id "river3#na1" --riot-id "river#ell"` (repeatable; defaults to both configured accounts) then `python src/build_dashboard.py`

## Typical run order

1. Build pool (one file per tier *and* per region -- `build_features.py` infers tier from the filename for apex tiers, so don't merge tiers into one file; `fetch_league_pool.py` tags every row with its `--platform` so region is never ambiguous even if you do)
- `python src/fetch_league_pool.py --tier DIAMOND --division II`
- `python src/fetch_league_pool.py --tier MASTER --output-path data/raw/league_entries_master.json`
- `python src/fetch_league_pool.py --tier GRANDMASTER --output-path data/raw/league_entries_grandmaster.json`
- `python src/fetch_league_pool.py --tier CHALLENGER --output-path data/raw/league_entries_challenger.json`
- For another region, add `--platform <region>` and a matching filename, e.g.: `python src/fetch_league_pool.py --platform kr --tier CHALLENGER --output-path data/raw/league_entries_kr_challenger.json`

2. Collect match IDs (once per pool file -- `fetch_match_ids.py` only reads one `--league-entries-path` per run; pass the matching `--platform` for non-NA files, and add `--skip-default-riot-ids` on the non-Diamond runs to avoid redundantly re-resolving your own accounts each time). Master/Grandmaster pools can be huge (NA and KR Master were both 10,000 players, the API's page cap) -- cap with `--max-players` to keep runtime and dataset size proportionate to the other tiers. Match IDs are globally unique across regions (platform-prefixed, e.g. `KR_...` vs `NA1_...`), so every region can share the same `match_ids.ndjson`/`matches.ndjson`.
- `python src/fetch_match_ids.py --resume`
- `python src/fetch_match_ids.py --league-entries-path data/raw/league_entries_master.json --max-players 300 --skip-default-riot-ids --checkpoint-path data/raw/checkpoints/match_ids_master_checkpoint.json`
- `python src/fetch_match_ids.py --league-entries-path data/raw/league_entries_grandmaster.json --max-players 300 --skip-default-riot-ids --checkpoint-path data/raw/checkpoints/match_ids_gm_checkpoint.json`
- `python src/fetch_match_ids.py --league-entries-path data/raw/league_entries_challenger.json --skip-default-riot-ids --checkpoint-path data/raw/checkpoints/match_ids_challenger_checkpoint.json`
- `python src/fetch_match_ids.py --platform kr --league-entries-path data/raw/league_entries_kr_challenger.json --skip-default-riot-ids --checkpoint-path data/raw/checkpoints/match_ids_kr_challenger_checkpoint.json` (repeat per KR tier, same `--max-players` caps as above)

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
