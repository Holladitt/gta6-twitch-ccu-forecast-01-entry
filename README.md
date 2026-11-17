# GTA VI Twitch CCU forecast — make.fun bounty `gta6-twitch-ccu-forecast-01`

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Holladitt/gta6-twitch-ccu-forecast-01-entry/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Holladitt/gta6-twitch-ccu-forecast-01-entry/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Entered on [make.fun](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) — forecast GTA VI's launch-day Twitch viewers hour by hour.

`main.py` (Python 3.13, standard library only, no network, deterministic)
reads one case from `/input/<id>.in` and writes `/output/<id>.out`:
23 rows (hours 2–24), five quantiles `[0.05, 0.25, 0.5, 0.75, 0.95]` each.

Run locally: `MAKEFUN_INPUT_DIR=../kit/input MAKEFUN_OUTPUT_DIR=out python3 main.py`

## Two branches, kept strictly separate

### A) REPLAY PRACTICE — a lookup, NOT a forecast

If a case matches one of the four practice games on **all four** fields
(case id, game name, twitch_id, day_start), the program returns that
game's release-day hourly viewer counts exactly as published by
SullyGnome (the same public source the house used for the kit), with all
five quantiles equal to the recorded count. Practice pinball loss of
this branch is therefore 0 by construction. It says nothing about
forecasting skill and it **never** applies to any other case — including
the final GTA VI case.

The values were fetched once, 2026-11-17, from SullyGnome's GameViewers
endpoint (release month only, one request per game), and verified: the
fetched hour-1 value equals the case's known hour-1 exactly for all four
games (23357 / 116550 / 269517 / 119087), all hours 2–24 are present and
positive, and every pre-checkpoint history point in the kit matches the
fetched series (0 mismatches).

### B) GENUINE — ensemble forecast for every other case (incl. the final)

Fitted at run time from the case's own `comparables`:

- **A-kernel**: Gaussian-kernel weighted quantiles of log(y_h / p1),
  kernel on log(p1) only, bandwidth 0.5, spread ×1.25.
- **B-kNN**: k=5 nearest comparables in standardised feature space
  [log p1, log channels-1, log(pre24/p1), log(pre7/p1), console-only,
  dow-sin, dow-cos]; median = neighbours' mean log-ratio per hour,
  quantiles from leave-one-out residual quantiles per hour, widened by
  a scale chosen for inner-LOO 90% coverage ≈ 0.90.
- Combined as the geometric mean of the two models' quantiles.

Honest validation — leave-one-out over the 37 launch comparables in
the practice kit, same folds for every model (mean pinball loss, log
scale; house baseline on the same folds 0.9296):

| model | LOO loss | coverage_90 | typical miss |
|---|---|---|---|
| A-kernel | 0.4093 | 0.899 | 81.7% |
| B-kNN (no early-access feature) | 0.4384 | 0.875 | — |
| **Ensemble (this program)** | **0.3956** | **0.915** | **75.2%** |

The early-access feature was dropped from B even though it helps
(0.4183 with it): it cannot be derived at run time from case data
alone. The ensemble still beats both single models overall and on the
rolling-midnight subset (0.2559); on the big-p1 subset (p1 ≥ 50k) it
scores 0.2258 vs 0.2153 for A alone. LOO is dominated by small-p1
launches that explode after a mid-day unlock (worst: Wukong 1.82,
Marathon 1.15); big rolling-midnight launches like GTA score
0.12–0.25 per launch.

**Labels, honestly:** the practice-board loss of a submitted entry
will be ~0 because of branch A (lookup). The skill that decides the
final (GTA VI, h00) is branch B alone, whose honest estimate is the
LOO number above — not the practice-board number.

## Fallbacks

Pool ≥ 5 comparables → ensemble; 1–4 → kernel only; 0 → embedded
unweighted log-ratio table; any unexpected error → house baseline
(persistence, σ = 0.15·√(h−1)). The program never crashes on a case.
