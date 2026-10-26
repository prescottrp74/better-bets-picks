# Better Bets — weekend under picks (auto)

Picks the 3 football matches least likely to see 3+ goals ("under 2.5"),
priced 1.20 or better, and publishes them to a phone-friendly page — refreshed
automatically by GitHub Actions, no PC needed.

**Your page:** `https://prescottrp74.github.io/better-bets-picks/`
(live once the repo is public and Pages is enabled — see setup below).

## How it works
- `data/Matches.csv.gz` — historical results + odds (shipped).
- Each run, `src/fetch_and_run.py` fetches the current season's results and this
  round's fixtures/odds from football-data.co.uk (GitHub's runners have open
  internet), refreshes the Dixon-Coles goals model, and ranks unders by how
  likely they are to land.
- `.github/workflows/picks.yml` runs it Fri & Sat mornings (and on demand from
  the Actions tab), then commits the new `docs/index.html` page.

## One-time setup
1. Repo must be **public** for free GitHub Pages.
2. Settings → Pages → Source: "Deploy from a branch", branch `main`, folder
   `/docs`. Save.
3. Settings → Actions → General → Workflow permissions: "Read and write".
4. Run the workflow once (Actions tab → "Weekend under picks" → Run workflow)
   to publish the first page.

## Honest note
These are the **safest fair-priced** unders, not a proven edge. Backtesting over
150k+ matches found no way to beat this market with public data (see the main
project). The model picks low-scoring games; it does not beat the bookie. Treat
it as entertainment and stake small.

## Leagues covered
Big 5 + Championship, English League 1/2 & National League, Scottish divisions,
Segunda, Serie B, Ligue 2, 2.Bundesliga, Eredivisie, Primeira, Belgian, Turkish,
Greek — 22 in total.
