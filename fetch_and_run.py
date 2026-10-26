"""
Better Bets — self-contained weekend under-2.5 picker (GitHub Actions).

Runs where there's open internet: fetches historical results (mirror) +
current-season results and this round's fixtures/odds (football-data.co.uk),
fits a time-weighted Dixon-Coles goals model, ranks the unders most likely to
land at >= 1.20, and writes a phone-friendly page to docs/index.html.

One file on purpose, so it is easy to host.
"""
import os
import sys
import json
import html
import datetime as dt
import urllib.request

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
BASE = "https://www.football-data.co.uk"
HISTORY_URL = ("https://raw.githubusercontent.com/xgabora/"
               "Club-Football-Match-Data-2000-2025/main/data/Matches.csv")
UA = {'User-Agent': 'Mozilla/5.0'}
MIN_ODDS = 1.20
TOP_N = 3

WINDOW_DAYS = 540
HALFLIFE_DAYS = 180
MIN_MATCHES = 150
MIN_APPS = 15

LEAGUES = {
    'E0': 'Premier League', 'E1': 'Championship', 'SP1': 'La Liga', 'I1': 'Serie A',
    'F1': 'Ligue 1', 'D1': 'Bundesliga', 'E2': 'Eng League One', 'E3': 'Eng League Two',
    'EC': 'Eng National League', 'SC0': 'Scot Premiership', 'SC1': 'Scot Championship',
    'SC2': 'Scot League One', 'SC3': 'Scot League Two', 'SP2': 'Segunda', 'I2': 'Serie B',
    'F2': 'Ligue 2', 'D2': '2.Bundesliga', 'N1': 'Eredivisie', 'P1': 'Primeira Liga',
    'B1': 'Belgian Pro League', 'T1': 'Turkish Super Lig', 'G1': 'Greek Super League'}
DIVS = list(LEAGUES)
MAXG = 10

# ----------------------------------------------------------------------------
# Dixon-Coles model
# ----------------------------------------------------------------------------
def _tau(x, y, lam, mu, rho):
    t = np.ones_like(lam, dtype=float)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    t[m00] = 1.0 - lam[m00] * mu[m00] * rho
    t[m01] = 1.0 + lam[m01] * rho
    t[m10] = 1.0 + mu[m10] * rho
    t[m11] = 1.0 - rho
    return t


class DixonColes:
    def __init__(self, n_teams):
        self.n_teams = n_teams
        self.params = None

    def _unpack(self, p):
        n = self.n_teams
        home = p[0]
        rho = p[1]
        att = p[2:2 + n]
        dfn = p[2 + n:2 + 2 * n]
        att = att - att.mean()
        return home, rho, att, dfn

    def _nll(self, p, hi, ai, hg, ag, w):
        home, rho, att, dfn = self._unpack(p)
        log_lam = home + att[hi] - dfn[ai]
        log_mu = att[ai] - dfn[hi]
        lam = np.exp(log_lam)
        mu = np.exp(log_mu)
        tau = np.clip(_tau(hg, ag, lam, mu, rho), 1e-10, None)
        ll = (hg * log_lam - lam - gammaln(hg + 1.0)
              + ag * log_mu - mu - gammaln(ag + 1.0) + np.log(tau))
        return -np.sum(w * ll)

    def fit(self, hi, ai, hg, ag, w, x0=None, maxiter=200):
        n = self.n_teams
        if x0 is None:
            x0 = np.concatenate([[0.25, -0.05], np.zeros(n), np.zeros(n)])
        bounds = [(-2, 2), (-0.3, 0.3)] + [(-3, 3)] * (2 * n)
        res = minimize(self._nll, x0, args=(hi, ai, hg, ag, w),
                       method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": maxiter, "ftol": 1e-7})
        self.params = res.x
        self.success = res.success
        return self

    def export_ratings(self):
        home, rho, att, dfn = self._unpack(self.params)
        return {'home': float(home), 'rho': float(rho),
                'att': {t: float(att[i]) for t, i in self._tidx.items()},
                'def': {t: float(dfn[i]) for t, i in self._tidx.items()}}


def _score_matrix_raw(log_lam, log_mu, rho):
    lam = np.exp(log_lam); mu = np.exp(log_mu)
    gx = np.arange(MAXG + 1)
    px = np.exp(gx * np.log(lam) - lam - gammaln(gx + 1.0))
    py = np.exp(gx * np.log(mu) - mu - gammaln(gx + 1.0))
    M = np.outer(px, py)
    M[0, 0] *= 1.0 - lam * mu * rho
    M[0, 1] *= 1.0 + lam * rho
    M[1, 0] *= 1.0 + mu * rho
    M[1, 1] *= 1.0 - rho
    M = np.clip(M, 0, None)
    return M / M.sum()


def prob_under_from_ratings(home, rho, att_h, def_h, att_a, def_a):
    log_lam = home + att_h - def_a
    log_mu = att_a - def_h
    M = _score_matrix_raw(log_lam, log_mu, rho)
    p_under = M[0, 0] + M[1, 0] + M[0, 1] + M[2, 0] + M[1, 1] + M[0, 2]
    return float(p_under), float(1.0 - p_under), float(np.exp(log_lam) + np.exp(log_mu))


# ----------------------------------------------------------------------------
# Ratings
# ----------------------------------------------------------------------------
def build_ratings():
    path = 'data/Matches.csv' if os.path.exists('data/Matches.csv') else 'data/Matches.csv.gz'
    df = pd.read_csv(path, low_memory=False)
    df['MatchDate'] = pd.to_datetime(df['MatchDate'], errors='coerce')
    if os.path.exists('data/new_2627.csv'):
        sup = pd.read_csv('data/new_2627.csv')
        sup['MatchDate'] = pd.to_datetime(sup['MatchDate'], errors='coerce')
        keep = ['Division', 'MatchDate', 'HomeTeam', 'AwayTeam', 'FTHome', 'FTAway']
        df = pd.concat([df, sup[keep]], ignore_index=True)
        df = df.drop_duplicates(subset=['Division', 'MatchDate', 'HomeTeam', 'AwayTeam'],
                                keep='last')
    df = df.dropna(subset=['FTHome', 'FTAway', 'MatchDate'])
    asof = df['MatchDate'].max()
    out = {'as_of': str(asof.date()), 'leagues': {}}
    for div, name in LEAGUES.items():
        d = df[df['Division'] == div]
        lo = asof - pd.Timedelta(days=WINDOW_DAYS)
        hist = d[(d['MatchDate'] <= asof) & (d['MatchDate'] >= lo)]
        if len(hist) < MIN_MATCHES:
            continue
        teams = sorted(set(hist['HomeTeam']) | set(hist['AwayTeam']))
        tidx = {t: i for i, t in enumerate(teams)}
        age = (asof - hist['MatchDate']).dt.days.values
        w = 0.5 ** (age / HALFLIFE_DAYS)
        m = DixonColes(len(teams)).fit(
            hist['HomeTeam'].map(tidx).values, hist['AwayTeam'].map(tidx).values,
            hist['FTHome'].values.astype(float), hist['FTAway'].values.astype(float), w)
        m._tidx = tidx
        apps = pd.concat([hist['HomeTeam'], hist['AwayTeam']]).value_counts()
        r = m.export_ratings()
        thin = [t for t in list(r['att']) if apps.get(t, 0) < MIN_APPS]
        for t in thin:
            r['att'].pop(t); r['def'].pop(t)
        r['name'] = name
        r['base_under_rate'] = round(float((hist['FTHome'] + hist['FTAway'] < 3).mean()), 3)
        out['leagues'][div] = r
    os.makedirs('out', exist_ok=True)
    json.dump(out, open('out/ratings.json', 'w'), indent=1)
    print(f"Ratings built as of {out['as_of']} for {len(out['leagues'])} leagues.")
    return out


# ----------------------------------------------------------------------------
# Picks
# ----------------------------------------------------------------------------
def pick(fixtures_csv, min_odds=MIN_ODDS, top_n=TOP_N, ratings_path='out/ratings.json'):
    R = json.load(open(ratings_path))
    fx = pd.read_csv(fixtures_csv)
    rows, unmatched = [], []
    for _, f in fx.iterrows():
        div = str(f['division']).strip()
        lg = R['leagues'].get(div)
        if lg is None:
            unmatched.append(f"{f['home']} v {f['away']} (league '{div}' not rated)")
            continue
        h, a = str(f['home']).strip(), str(f['away']).strip()
        if h not in lg['att'] or a not in lg['att']:
            miss = [t for t in (h, a) if t not in lg['att']]
            unmatched.append(f"{h} v {a} ({', '.join(miss)} not in {div})")
            continue
        p_under, _, eg = prob_under_from_ratings(
            lg['home'], lg['rho'], lg['att'][h], lg['def'][h], lg['att'][a], lg['def'][a])
        rows.append({'league': lg['name'], 'fixture': f"{h} v {a}",
                     'P_under': round(p_under, 3), 'fair_odds': round(1 / p_under, 2),
                     'under_odds': float(f['under_odds']), 'exp_goals': round(eg, 2)})
    res = pd.DataFrame(rows)
    flagged = pd.DataFrame()
    if len(res):
        bad = (res['exp_goals'] < 1.8) | (res['P_under'] > 0.80)
        flagged = res[bad].sort_values('P_under', ascending=False).reset_index(drop=True)
        res = res[~bad]
        res = res[res['under_odds'] >= min_odds].sort_values('P_under', ascending=False)
        res = res.reset_index(drop=True)
    return res, unmatched, flagged


# ----------------------------------------------------------------------------
# Fetching
# ----------------------------------------------------------------------------
def season_code(today):
    y = today.year
    start = y if today.month >= 7 else y - 1
    return f"{str(start)[2:]}{str(start + 1)[2:]}"


def get(url, timeout=90):
    return urllib.request.urlopen(
        urllib.request.Request(url, headers=UA), timeout=timeout).read().decode('utf-8', 'replace')


def fetch_history():
    dst = 'data/Matches.csv'
    if os.path.exists(dst) or os.path.exists(dst + '.gz'):
        print("history present, skipping download")
        return
    print("downloading history base...")
    data = urllib.request.urlopen(urllib.request.Request(HISTORY_URL, headers=UA),
                                  timeout=180).read()
    open(dst, 'wb').write(data)
    print(f"history bytes: {len(data)}")


def fetch_results(code):
    rows = ['Division,MatchDate,HomeTeam,AwayTeam,FTHome,FTAway']
    for d in DIVS:
        try:
            txt = get(f"{BASE}/mmz4281/{code}/{d}.csv")
        except Exception as e:
            print(f"  skip results {d}: {e}")
            continue
        lines = [x for x in txt.splitlines() if x.strip()]
        h = lines[0].split(',')
        need = ('Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG')
        if not all(k in h for k in need):
            continue
        ix = {k: h.index(k) for k in need}
        for ln in lines[1:]:
            c = ln.split(',')
            try:
                fh, fa = c[ix['FTHG']], c[ix['FTAG']]
                if fh == '' or fa == '':
                    continue
                p = c[ix['Date']].split('/')
                if len(p) != 3:
                    continue
                yr = p[2] if len(p[2]) == 4 else '20' + p[2]
                rows.append(f"{c[ix['Div']]},{yr}-{p[1]}-{p[0]},{c[ix['HomeTeam']]},"
                            f"{c[ix['AwayTeam']]},{fh},{fa}")
            except (IndexError, ValueError):
                continue
    open('data/new_2627.csv', 'w').write('\n'.join(rows))
    print(f"results rows: {len(rows) - 1}")


def fetch_fixtures():
    txt = get(f"{BASE}/fixtures.csv")
    lines = [x for x in txt.splitlines() if x.strip()]
    h = lines[0].split(',')
    iH = {k: h.index(k) for k in ('Div', 'Date', 'HomeTeam', 'AwayTeam')}
    iMax = h.index('Max<2.5') if 'Max<2.5' in h else None
    iAvg = h.index('Avg<2.5') if 'Avg<2.5' in h else None
    rows = ['division,date,home,away,under_odds,avg_under']
    for ln in lines[1:]:
        c = ln.split(',')
        try:
            if iMax is None or c[iMax] == '':
                continue
            p = c[iH['Date']].split('/')
            date = f"{p[2] if len(p[2])==4 else '20'+p[2]}-{p[1]}-{p[0]}" if len(p) == 3 else c[iH['Date']]
            avg = c[iAvg] if iAvg is not None and len(c) > iAvg else ''
            rows.append(f"{c[iH['Div']]},{date},{c[iH['HomeTeam']]},{c[iH['AwayTeam']]},{c[iMax]},{avg}")
        except (IndexError, ValueError):
            continue
    open('data/fixtures_latest.csv', 'w').write('\n'.join(rows))
    print(f"fixtures rows: {len(rows) - 1}")
    return 'data/fixtures_latest.csv'


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
def render_page(res, top_n, as_of, fixtures_dates):
    now = dt.datetime.utcnow().strftime('%a %d %b %Y, %H:%M UTC')
    top = res.head(top_n).to_dict('records') if len(res) else []
    cards = []
    for i, r in enumerate(top, 1):
        cards.append(f"""
    <div class="pick">
      <div class="rank">{i}</div>
      <div class="body">
        <div class="fx">{html.escape(r['fixture'])}</div>
        <div class="lg">{html.escape(r['league'])}</div>
        <div class="row"><span class="prob">{r['P_under']*100:.0f}% likely under 2.5</span>
          <span class="odds">@ {r['under_odds']}</span></div>
      </div>
    </div>""")
    if not cards:
        cards = ['<p class="empty">No qualifying fixtures this run. Check back closer to kickoff.</p>']
    others = ''
    if len(res) > top_n:
        rowsh = ''.join(
            f"<tr><td>{html.escape(r['fixture'])}</td><td>{html.escape(r['league'])}</td>"
            f"<td>{r['P_under']*100:.0f}%</td><td>{r['under_odds']}</td></tr>"
            for r in res.iloc[top_n:top_n + 12].to_dict('records'))
        others = (f'<details><summary>More candidates</summary><table>'
                  f'<tr><th>Match</th><th>League</th><th>P(under)</th><th>Best odds</th></tr>'
                  f'{rowsh}</table></details>')
    drange = f"Fixtures {fixtures_dates[0]} – {fixtures_dates[-1]}" if fixtures_dates else ""
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Better Bets — weekend unders</title>
<style>
:root {{ --bg:#0f1115; --card:#1a1d24; --line:#2a2f3a; --txt:#e8eaed; --mut:#9aa0aa;
  --accent:#4ade80; --odds:#fbbf24; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  background:var(--bg); color:var(--txt); padding:16px; max-width:560px; margin:0 auto; }}
h1 {{ font-size:1.4rem; margin:.2rem 0 0; }}
.sub {{ color:var(--mut); font-size:.82rem; margin:.3rem 0 1.2rem; line-height:1.4; }}
.pick {{ display:flex; gap:14px; background:var(--card); border:1px solid var(--line);
  border-radius:14px; padding:16px; margin-bottom:12px; align-items:center; }}
.rank {{ font-size:1.6rem; font-weight:700; color:var(--accent); width:28px; text-align:center; }}
.fx {{ font-size:1.12rem; font-weight:650; }}
.lg {{ color:var(--mut); font-size:.8rem; margin:.15rem 0 .5rem; }}
.row {{ display:flex; justify-content:space-between; align-items:baseline; }}
.prob {{ color:var(--accent); font-weight:600; font-size:.9rem; }}
.odds {{ color:var(--odds); font-weight:700; font-size:1.05rem; }}
.empty {{ color:var(--mut); }}
details {{ margin-top:10px; }} summary {{ color:var(--mut); cursor:pointer; padding:8px 0; }}
table {{ width:100%; border-collapse:collapse; font-size:.82rem; margin-top:6px; }}
th,td {{ text-align:left; padding:6px 4px; border-bottom:1px solid var(--line); }}
th {{ color:var(--mut); font-weight:500; }}
.foot {{ color:var(--mut); font-size:.72rem; margin-top:22px; line-height:1.5;
  border-top:1px solid var(--line); padding-top:14px; }}
</style></head><body>
<h1>Weekend unders</h1>
<div class="sub">Updated {now}. {drange}. Ratings as of {as_of}.<br>
The 3 matches least likely to see 3+ goals, priced 1.20 or better.</div>
{''.join(cards)}
{others}
<div class="foot">
These are the <b>safest fair-priced</b> unders, not a proven edge — the model
picks low-scoring games, it doesn't beat the bookie. A treble of the three is the
"small stake → modest return" shape, but like all betting it loses slowly on
average. Stake it as entertainment. Best (Max&lt;2.5) prices shown; shop around.
</div>
</body></html>"""


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)
    os.makedirs('data', exist_ok=True)
    os.makedirs('docs', exist_ok=True)
    os.makedirs('out', exist_ok=True)
    today = dt.date.today()
    print(f"Season code: {season_code(today)}")
    fetch_history()
    fetch_results(season_code(today))
    fx_path = fetch_fixtures()
    out = build_ratings()
    res, unmatched, flagged = pick(fx_path, MIN_ODDS, TOP_N)
    fx = pd.read_csv(fx_path)
    dates = sorted(fx['date'].dropna().unique().tolist())
    open('docs/index.html', 'w').write(render_page(res, TOP_N, out['as_of'], dates))
    if len(res):
        res.to_csv('docs/picks.csv', index=False)
    print(f"Wrote docs/index.html: {min(TOP_N, len(res))} picks, "
          f"{len(res)} qualifying, {len(flagged)} quarantined.")


if __name__ == '__main__':
    main()
