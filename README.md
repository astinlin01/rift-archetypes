# Rift Archetypes

Rift Archetypes collects League of Legends solo queue data and prepares it for unsupervised playstyle clustering.

Current collection pipeline:
1. Build a player pool from Riot league endpoints.
2. Fetch and dedupe match IDs across that pool.
3. Fetch full match payloads for feature engineering.

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

## Why your accounts are added in match IDs but not in league pool

This is expected with current design:
- `fetch_league_pool.py` is a direct snapshot of the selected rank slice from Riot league endpoints.
- Your accounts are only present there if they naturally belong to that fetched slice.
- `fetch_match_ids.py` intentionally force-includes your Riot IDs so your accounts are guaranteed in downstream match collection.

In short:
- league pool = objective rank slice snapshot
- match ID stage = rank slice snapshot plus required personal accounts

## Typical run order

1. Build pool
- `python src/fetch_league_pool.py`

2. Collect match IDs
- `python src/fetch_match_ids.py --resume`

3. Fetch match payloads
- `python src/fetch_match_details.py --resume`

## Notes

- Riot developer keys expire frequently. If requests suddenly fail with 401, regenerate key and update `.env`.
- Raw data files under `data/raw/` are intentionally gitignored.
