# rift-archetypes

## Analysis conventions
- **CS** includes jungle camps: total minions killed + neutral minions killed.
- **Season-match** every comparison: compare a player only against baselines from the same season.
- **Normalize** for game length (per-minute stats) and control for match outcome, so features describe playstyle, not whether the game was won.
- **Show comparisons as percentiles**, not basis points or raw differences.
- If a dashboard section doesn't have enough data, show an explicit "insufficient data (n=X)" message. Never render it blank.

## Testing
- Accuracy tests encode the conventions above, using a few real games whose expected values were checked by hand.
- Workflow: write or update the tests, run them and show the results, fix the pipeline until they're green, and only then regenerate the dashboard.
