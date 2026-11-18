# Default run results (Yu)

Reproduction of the README's default run on branch `arman`.

| Item | Value |
|---|---|
| Date | 2026-11-18 |
| Platform | Windows 11, Python 3.10.9, fresh `.venv` from `requirements.txt` |
| Command | `python run_experiment.py` |
| Formulation | aggregated (default) |
| Solver | HiGHS (`highspy` 1.16.0) via `--solver auto` |
| Seed | 60 |
| Wall time | about 1 minute |
| Raw summary | [`run_summary.json`](run_summary.json) (copy of `output/results/run_summary.json`) |

## Model size and scenarios

- 2,001 variables (1,306 integer, 0 binary), 1,124 constraints.
- 30 generated scenarios reduced to 4 retained scenarios (2 tail, 2 central).
- Effective number of scenarios is 2.09. One scenario (S21) carries weight 0.633.
- The data pipeline gives 3 destinations, 12 periods of 15 minutes, 36 booked and 54 on-demand passengers.
- Booking feasibility passes both the Eq. (10) data cut and the exact first-stage MILP check.

| Scenario | Probability | Stratum | Aircraft |
|---|---|---|---|
| S11 | 0.033 | tail | 26 |
| S17 | 0.067 | tail | 14 |
| S21 | 0.633 | central | 23 |
| S28 | 0.267 | central | 22 |

## Solve status

- HiGHS stopped inside the requested gap: **4.88%** achieved against a **5.00%** target.
- Objective **2414.62**, dual bound 2532.50.
- This is an incumbent solution, **not a proven optimum**.
- Reconciliation: the recomputed objective matches the solver objective (difference 2.27e-12).

## First-stage decisions

| Decision | Value | Cost |
|---|---|---|
| Initial fleet | 6 | $240.00 |
| Reserved charger units (total) | 18 | $360.00 |
| Planned departures (total) | 19 | $380.00 |

## Expected recourse components

| Component | Value |
|---|---|
| Revenue | 5265.91 |
| Charging energy | 68.45 |
| Flight operating | 1094.84 |
| Emergency charging | 89.33 |
| Added departures | 172.00 |
| Cancelled departures | 30.67 |
| Unserved booked | 0.00 |
| Unserved on-demand | 130.00 |
| **Expected profit** | **3680.62** |

## Risk

| Quantity | Value |
|---|---|
| alpha | 0.90 |
| theta | 22.00 |
| VaR (zeta) | 11.00 |
| CVaR_alpha(L) | 13.00 |
| theta × CVaR penalty | 286.00 |
| Marginal penalty (booked) | 590.00 |
| Marginal penalty (on-demand) | 320.00 |

## Passenger service

| Class | Demand (expected) | Served (expected) | Rate |
|---|---|---|---|
| Booked | 36.0 | 36.0 | 100.00% |
| On-demand | 49.2 | 47.9 | 97.36% |

By scenario:

| Scenario | Prob | Booked | Booked served | On-demand | On-demand served | On-demand unserved | Service rate |
|---|---|---|---|---|---|---|---|
| S11 | 0.033 | 36 | 36 | 80 | 69 | 11 | 90.5% |
| S17 | 0.067 | 36 | 36 | 40 | 26 | 14 | 81.6% |
| S21 | 0.633 | 36 | 36 | 49 | 49 | 0 | 100.0% |
| S28 | 0.267 | 36 | 36 | 48 | 48 | 0 | 100.0% |

Unserved passengers appear only in the two tail scenarios.

## Issue found during setup (fixed in commit 2122331)

`highspy` 1.16 imports `typing_extensions` on Python 3.10 but does not declare it as a dependency. With a fresh install, `import highspy` fails. `--solver auto` then prints only `[info] skipping HIGHS -- highspy unavailable` and falls back to CBC, which was still running after more than 10 minutes on the default instance.

The fix adds `typing_extensions>=4.0` to `requirements.txt` and `pyproject.toml`. After that, HiGHS is picked automatically and the run takes about a minute, as the README says.

## Suggestions

1. **Make the solver fallback louder.** The skip message hides the real import error. Show the exception, or fail when HiGHS was expected but could not be imported.
2. **Check the scenario reduction weights.** With N_eff = 2.09 and S21 at weight 0.633, the expected values mostly reflect a single scenario.
3. **Treat objective comparisons with care.** The 4.88% gap means the result is an incumbent, not a proven optimum. Tighten `--gap` before comparing objectives across formulations or settings.

## Not run

`--formulation indexed`, `--export`, `--solver cbc` and other seeds.
