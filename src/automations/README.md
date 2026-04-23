# `src/automations/` — Monthly Ranking Automation

Lightweight scheduled jobs that aggregate Monday.com board data. Currently ships:

- **`monthly_ranking.py`** — reads the AE Tracker board, filters to the current month, produces a per-advisor ranked DataFrame (multi-metric: one rank column per metric).

## Local run

```bash
source .venv/bin/activate

RANKING_SOURCE_BOARD_ID=18402779315 \
RANKING_METRICS="Appels faits,Appels répondus,RDV Book,RDV Faits,PA Vendues,\$\$\$ Recues" \
RANKING_DATE_COLUMN=Date \
RANKING_MONTHS_AGO=1 \
python -m src.automations.monthly_ranking
```

`MONDAY_API_KEY` is picked up from `.env` or the environment.

## Tests

```bash
pytest src/automations/tests/ -v
```

All tests are pure-logic (no Monday API calls).

## GitHub Actions

The workflow `.github/workflows/monthly_ranking.yml` runs `python -m src.automations.monthly_ranking` every 10 minutes (cron) and on manual dispatch.

### Required secrets (GitHub → Settings → Secrets)
- `MONDAY_API_KEY`
- `RANKING_SOURCE_BOARD_ID`

### Required variables (GitHub → Settings → Variables)
- `RANKING_METRICS` — CSV list, e.g. `Appels faits,RDV Book,$$$ Recues`
- `RANKING_DATE_COLUMN` — e.g. `Date`

### Optional variables
- `RANKING_MONTHS_AGO` — default `0` (current month). Use `1` during testing when the current month has no data yet.
- `RANKING_ADVISOR_COLUMN` — default `group_title`
- `RANKING_USE_GROUP_AS_ADVISOR` — default `true`
- `RANKING_TARGET_BOARD_ID` — reserved for a future iteration (writeback to a ranking board). Currently unused.

## Scope of this iteration

The script reads, aggregates, ranks, and logs the DataFrame. It does **not** yet write back to a target Monday board — that step is intentionally deferred until the ranking logic is validated in production runs.
