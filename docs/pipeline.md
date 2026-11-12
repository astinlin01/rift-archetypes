# Pipeline reference

Full reference for every script, the notebooks and the run order. For the why behind the method, see the [README](../README.md).

Pipeline:
1. Build a player pool from Riot league endpoints.
2. Fetch and dedupe match IDs across that pool.
3. Fetch full match payloads (and timelines for the early-game features).
4. Build a per-participant feature table.
5. Fit per-role playstyle clusters.
6. Compare your own matches against a reference tier.
7. Export the interactive dashboard.

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
- Always includes your own accounts (`RIOT_IDS` in `.env`) even if they are not present in the league pool file. Disable with `--skip-default-riot-ids`.
- Resolves Riot IDs to puuid via account-v1.
- Dedupes match IDs globally.

### src/fetch_match_details.py
Reads `match_ids.ndjson` and fetches full match payloads. Auto-routes each match ID to the correct regional cluster based on its platform prefix (`KR_...` -> `asia`, `NA1_...` -> `americas`, etc.). match-v5 lookups 404 through the wrong cluster, so a single run safely handles a `match_ids.ndjson` that mixes regions. `--platform` is only a fallback for an unrecognized prefix.

Outputs:
- `data/raw/matches.ndjson`
- `data/raw/checkpoints/match_details_checkpoint.json`
- `data/raw/failures/match_details_failures.ndjson`

### src/fetch_match_timelines.py + src/build_timeline_features.py
Resumable fetch of match-v5 timelines for the subset that matters (your own games + players with 10+ games in a role), or `--all` for a full-season backfill. Timelines are stored gzipped, one file per match (`data/raw/timelines/<matchId>.json.gz`, ~35-40 KB each vs ~0.4-1.3 MB raw), so new features can be re-extracted later without refetching. `build_timeline_features.py` turns them into the early-game feature table described in `src/timeline_features.py`.

### src/fetch_recent_self.py
Pulls your own ranked match history directly (resolve Riot ID -> paginate match-v5 -> match details) without scanning the full league pool. By default it paginates until it runs out of history (100 IDs per call, the API max); pass `--max-games` to cap it. Appends into the same `match_ids.ndjson`/`matches.ndjson` files, so a normal `build_features.py` run picks the new games up. Safe to re-run: already-fetched IDs/details are skipped.

- Run: `python src/fetch_recent_self.py` (all history for every account in `RIOT_IDS`), add `--max-games 50`, or `--since 2026-01-08` to only pull the current season (comparisons are season-matched anyway, so older games are never used). `--riot-id "yourname#na1"` overrides `RIOT_IDS`.

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
- Applies per-minute normalization from `feature_config.py`, plus derived features (`cs_total_per_min` = lane minions + jungle camps, the "CS" number op.gg and the client show; `cs_per_min` alone is minions only).
- Tags every row with `patch`, `season` (Riot's major version: 16 = 2026) and `gameCreation`.
- Attaches source tier **and region** per puuid by matching against all `data/raw/league_entries*.json` pool files (`fetch_league_pool.py` tags every saved row with `platform`; files predating that default to NA1).
- Output: `data/processed/participant_features.parquet`
- Run: `python src/build_features.py`

### src/playstyle_space.py
Shared "playstyle space" transform used by clustering, `compare_self.py`, the dashboard export and the notebooks: context adjustment → `StandardScaler` → `PCA`. See [the hard part](../README.md#the-hard-part-separating-playstyle-from-outcome) for why the adjustment exists. Game length is a covariate (a spline) because champion level, killing sprees, turret plates and max lane leads are per-game totals or maxima that grow with game length, non-linearly in champion level's case (XP curve, cap at 18). Without it, the shortest games formed their own "low everything" cluster. Exact-duplicate features (e.g. Riot's `gold_per_minute` vs. our `gold_earned_per_min`) are excluded from clustering so they don't double-weight one dimension. The fitted adjustment lives in each role's model artifact, so every consumer projects rows identically.

Clusters are named after their two most distinctive playstyle traits (e.g. "Vision Control · Frontline", "Enchanter · Skillshot Mage"). Traits are groups of standardized, adjusted features, filtered per role so names like "Jungle Farmer" only appear for junglers.

### src/style_profile.py
Trait-axis style profiles, the replacement for k-means as the main "what kind of player am I" view (built and tested; not yet wired into the dashboard). Per role: adjusted features → standardized against the role population → per-game trait scores (mean of the trait's *available* features, so games without a timeline still score on end-of-game features) → player profile (mean over games, with a per-trait game count) → percentiles against players with 10+ games → soft archetype labels (cosine match to the role's archetype signatures, clamped to 0–100%) → confidence (single-game reliability as ICC(1), scaled to the player's game count with Spearman–Brown; low < 0.5 ≤ medium < 0.7 ≤ high).

Early (timeline) features are adjusted for champion only: they happen before 14:00, so adjusting them for end-of-game context like teammates' lead or game length would subtract out effects the early game caused.

### src/champion_classes.py
Primary Data Dragon class per champion, used to pool rare champions in the champion adjustment. Case-insensitive, since match data spells `FiddleSticks` and Data Dragon `Fiddlesticks`.

### src/method_diagnostics.py
Checks that the adjustment does what the README claims, per role, raw vs adjusted: champion share of feature variance (should drop), |correlation with win| (should drop, not to zero), and split-half reliability of player means (should stay well above zero).

- Report: `data/processed/method_diagnostics.md`
- Run: `python src/method_diagnostics.py`

### src/build_clusters.py
Fits a separate adjustment → `StandardScaler` → `PCA` → `KMeans` pipeline per role (role is the dominant driver of raw stats, so clustering is done within-role rather than globally). Sweeps k and reports silhouette score to pick the cluster count per role (override with `--k-override ROLE=K`). The centroid report lists each cluster's name, win rate and most common champions.

- Models: `models/<role>_cluster.joblib`
- Cluster assignments: `data/processed/participant_clusters.parquet`
- Centroid report: `data/processed/cluster_centroid_report.md`
- Run: `python src/build_clusters.py`

### src/compare_self.py
Assigns your own matches to the fitted per-role clusters and compares your per-feature averages/percentiles against a reference tier (default Challenger).

- Run: `python src/compare_self.py` (accounts from `RIOT_IDS`; or `--riot-id "yourname#na1"` / `--puuid <puuid>`, repeatable)
- Output: `data/processed/self_comparison.csv`

### src/build_pro_features.py
Builds a per-player-per-game feature table from [Oracle's Elixir](https://oracleselixir.com/tools/downloads)'s professional match data CSV (LCK/LPL by default; any league code in the `league` column works via `--leagues`), for comparing your own stats against pro players. Oracle's Elixir's schema doesn't have our `challenges`-style derived stats, so only a subset of features carry a clean equivalent (no CC time, no lane-opponent-diff data). See the script for exactly which fields map to what.

Their download is Google Drive-hosted and its anonymous-download quota is sometimes exhausted for a while. If `gdown <file-id>` fails with a quota error, download the file manually in a logged-in browser instead and place it at `--input-path`.

- Run: `python src/build_pro_features.py`
- Output: `data/processed/pro_player_features.parquet`

### src/export_dashboard_data.py + src/build_dashboard.py
Builds a self-contained, **interactive** HTML "scouting report" dashboard. `export_dashboard_data.py` precomputes everything up front for every configured account, every role, every reference tier, and cluster counts k=3-6 (percentile reference curves via percentile breakpoints, PCA scatter positions + per-point cluster assignments, PCA axis explainability captions), so the dashboard filters entirely client-side with no server round trip. `build_dashboard.py` merges the resulting `data/processed/dashboard_data.json` into `dashboard/template.html` to produce a single HTML file.

Dashboard filters:
- **Account**: switch between configured Riot IDs (e.g. main + smurf) entirely in the browser.
- **Game window**: last 20 / last 50 / all time.
- **Role**: any role you have games in, or "All Roles" for an overall (non-role-specific) view. The percentile chart still works there (aggregated across roles), but the archetype cards and cluster-space scatter require picking a specific role, since clustering is fit separately per role.
- **Champion**: narrow to one champion within the selected role, or all champions.
- **Region**: which platform's ranked population to compare against (NA, KR, ...; only regions with enough sample size for the selected role are offered).
- **Compare vs.**: reference tier within the selected region, ordered Challenger > Grandmaster > Master > Diamond (whichever tiers have enough sample size).
- **Playstyles (k)**: 3 to 6 clusters; cards, scatter colors, legend, and table archetypes all update together.

Note: clustering (archetype cards, cluster-space scatter) is fit on the **combined** population across all regions for a role. It isn't split by region. Only the percentile-comparison reference curves are region-specific.

Each archetype card shows the cluster's dominant traits and most common champions, plus two comparisons against the population: your win rate within that cluster, and your *share* of games landing in it (over/under-represented vs. the population's own distribution across clusters).

Champion icons are fetched from Data Dragon at export time, cached under `data/raw/champion_icon_cache/`, and embedded as base64 data URIs in the dashboard JSON, so the HTML file works offline and under strict content security policies.

Note: cluster counts here are fit fresh per k on the same per-role PCA space purely for the dashboard's slider. They don't change or replace the "official" per-role models in `models/` that `compare_self.py` uses (those keep whatever k `build_clusters.py`'s silhouette search picked).

A separate **"vs. Professional Play"** section compares you against LCK/LPL pro players (a league toggle switches between them), built from `build_pro_features.py`'s output if present. It uses a smaller feature set than the solo-queue comparison since Oracle's Elixir's schema doesn't cover everything Riot's match-v5 API does.

Below that, **"Which pro do you play like?"** finds your nearest pro player match: each qualifying pro (5+ games in a role this season) gets a season-average stat profile, standardized against the pro population's own mean/std per feature; your current stat profile (respecting every active filter: account, window, role, champion) is standardized the same way and compared by Euclidean distance to find the 3 closest pros in the selected league. This runs entirely client-side.

- Run: `python src/build_pro_features.py` (optional, for the pro comparison) then `python src/export_dashboard_data.py` (accounts from `RIOT_IDS`, or repeat `--riot-id`) then `python src/build_dashboard.py`

## Notebooks

All scripts in `src/` are runnable interactively from `notebooks/`. Each notebook cell simulates the script's CLI invocation (`sys.argv` + `parse_args()`), so the notebooks stay in sync with the scripts automatically rather than duplicating logic.

- **notebooks/00_pull_data.ipynb**: pull your own recent/full match history (wraps `fetch_recent_self.py`), rebuild the feature table, and optionally refresh the full ranked pool.
- **notebooks/01_explore_clusters.ipynb**: PCA scatter plots per role, centroid bar charts, and tier mix per cluster, reading from `data/processed/` and `models/`.
- **notebooks/02_self_analysis.ipynb**: interactive version of `compare_self.py`: a styled percentile table, a percentile bar chart, cluster distribution, and a per-game stat trend over time.
- **notebooks/03_generate_dashboard.ipynb**: runs `export_dashboard_data.py` + `build_dashboard.py` and sanity-checks the result.

## Typical run order

1. Build pool (one file per tier *and* per region: `build_features.py` infers tier from the filename for apex tiers, so don't merge tiers into one file; `fetch_league_pool.py` tags every row with its `--platform` so region is never ambiguous even if you do)
- `python src/fetch_league_pool.py --tier DIAMOND --division II`
- `python src/fetch_league_pool.py --tier MASTER --output-path data/raw/league_entries_master.json`
- `python src/fetch_league_pool.py --tier GRANDMASTER --output-path data/raw/league_entries_grandmaster.json`
- `python src/fetch_league_pool.py --tier CHALLENGER --output-path data/raw/league_entries_challenger.json`
- For another region, add `--platform <region>` and a matching filename, e.g.: `python src/fetch_league_pool.py --platform kr --tier CHALLENGER --output-path data/raw/league_entries_kr_challenger.json`

2. Collect match IDs (once per pool file: `fetch_match_ids.py` only reads one `--league-entries-path` per run; pass the matching `--platform` for non-NA files, and add `--skip-default-riot-ids` on the non-Diamond runs to avoid redundantly re-resolving your own accounts each time). Master/Grandmaster pools can be huge (NA and KR Master were both 10,000 players, the API's page cap), so cap with `--max-players` to keep runtime and dataset size proportionate to the other tiers. Match IDs are globally unique across regions (platform-prefixed, e.g. `KR_...` vs `NA1_...`), so every region can share the same `match_ids.ndjson`/`matches.ndjson`.
- `python src/fetch_match_ids.py --resume`
- `python src/fetch_match_ids.py --league-entries-path data/raw/league_entries_master.json --max-players 300 --skip-default-riot-ids --checkpoint-path data/raw/checkpoints/match_ids_master_checkpoint.json`
- `python src/fetch_match_ids.py --league-entries-path data/raw/league_entries_grandmaster.json --max-players 300 --skip-default-riot-ids --checkpoint-path data/raw/checkpoints/match_ids_gm_checkpoint.json`
- `python src/fetch_match_ids.py --league-entries-path data/raw/league_entries_challenger.json --skip-default-riot-ids --checkpoint-path data/raw/checkpoints/match_ids_challenger_checkpoint.json`
- `python src/fetch_match_ids.py --platform kr --league-entries-path data/raw/league_entries_kr_challenger.json --skip-default-riot-ids --checkpoint-path data/raw/checkpoints/match_ids_kr_challenger_checkpoint.json` (repeat per KR tier, same `--max-players` caps as above)

3. Fetch match payloads (and timelines)
- `python src/fetch_match_details.py --resume`
- `python src/fetch_match_timelines.py`

4. Build features
- `python src/build_features.py`
- `python src/build_timeline_features.py`

5. Fit clusters
- `python src/build_clusters.py`

6. Compare yourself
- `python src/compare_self.py`

7. Build the dashboard
- `python src/export_dashboard_data.py`
- `python src/build_dashboard.py`

To refresh with just your own newest games (skips re-scanning the full pool):
- `python src/fetch_recent_self.py`
- then re-run steps 4, 6, 7 (no need to refit clusters unless the population itself changed materially)

## Notes

- Riot developer keys expire every 24h. If requests suddenly fail with 401, regenerate the key and update `.env`.
- Raw data files under `data/raw/` are intentionally gitignored, as are `data/processed/` and `models/` (both regenerable from `data/raw/`).
