# Model notes and validation priorities

The code now implements the aircraft-indexed two-stage commitment--recourse
formulation from the paper: shared departure and charging commitments are
linked to scenario-specific charging, flights, passenger flows, additions,
cancellations, emergency capacity, and CVaR service loss.

Before treating results as publication-ready, validate the following remaining
modeling assumptions against the mathematical formulation in the paper:

1. Confirm the intended treatment of aircraft requiring zero charging periods
   and the final-period completion convention.
2. Verify the sign and interpretation of the CVaR term and the units of its risk
   weight.
3. Validate gravity-model calibration, demand adoption assumptions, and fare
   construction against observed or cited data.

The remaining items are calibration and convention checks rather than software
errors; they should be resolved before publication-quality experiments.

## Reading the results

### A negative objective is a demand-service signal

A negative objective usually means **unserved passengers dominate**, not that the
model is wrong. With `c^un = 100` per unserved passenger and a CVaR multiplier of
`lambda / (1 - alpha) = 22 / 0.10 = 220`, a handful of unserved passengers
outweighs the revenue from a small demand instance. The console report makes the
cause visible: compare `Expected unserved penalty` against `Expected revenue`.

Do **not** fix this by lowering `UNSERVED_PASSENGER_COST` — that hides the
structural shortfall rather than resolving it.

### Demand service: fleet size is the lever

The binding constraint is the **cumulative number of seats ready before each
passenger's deadline**, not charging speed. Measured levers, 5 seeds, `LoS = 2`:

| `AIRCRAFT_PASSENGER_RATIO` | mean objective | mean unserved | fully served |
|---|---|---|---|
| 0.30 | 77.0 | 1.960 | 1/5 |
| 0.60 | 275.5 | 0.613 | 3/5 |
| 0.90 | 350.2 | 0.640 | 3/5 |
| 1.20 | 459.1 | 0.067 | 4/5 |
| 1.80 | — | 0.000 | 5/5 |

Raising `CHARGING_RATE` (120 → 250 kW) and lowering `SOC_REQUIRED` (70 → 55) both
leave usable-aircraft unchanged — charging is **not** binding. Note also that
`soc_min` drives the charging *cost* term, so lowering it makes charging cheaper
without improving service.

### Scenario counts

With only 3 retained scenarios, the 0.90 CVaR tail is a **single scenario**, so
`VaR = CVaR` and the risk term reflects one outcome. Increase
`TARGET_SCENARIOS` before drawing conclusions about tail risk.

## Indexing and units

### Period indexing

Demand periods are **1-based** and must stay aligned with both the model's period
set, `range(1, n_periods + 1)`, and the 1-based `arrival_period` computed in
`model_builder.py`. An earlier revision used `for t in range(periods)`, which
keyed demand at 0…7 and silently dropped the period-0 passengers while inventing
an empty period 8. If you change the horizon, re-check that demand keys span
exactly `1..n_periods`.

### Horizon units

`HORIZON_START` and `HORIZON_END` are in **minutes** (`480` and `720`, i.e.
08:00–12:00), so `N_PERIODS = (HORIZON_END - HORIZON_START) // BIN_SIZE`.

Earlier revisions stored these in **seconds** while dividing by a `BIN_SIZE`
expressed in minutes. That mix produced 15-*second* periods and a horizon one
eighth of the intended length, so every timing result depended on which unit the
author had in mind. Two guards:

1. `N_PERIODS` must equal `(HORIZON_END - HORIZON_START) // BIN_SIZE` with both
   operands in the same unit.
2. The config comment must match the arithmetic — `720` is 12:00, and the inline
   comment currently reads 10:00.

The config, the demand time profile in `uam_data_pipeline.py` and
`scenario_generator.py` must all agree before any timing result is trusted. These
constants are imported **by value**, so a programmatic override must patch the
name in each importing module, not just `config`.

## Solver and log conventions

- The Gurobi trial licence caps a model at 2000 variables and 2000 constraints;
  the runner checks the size before building the solver and falls back. HiGHS
  (via `highspy`) is open source and unbounded, and is the practical choice.
- HiGHS logs the objective in internal minimization form, so the sign is flipped
  relative to the model. PuLP reports it in the model's own sense; the two agree.
- A run that stops on the time limit or gap tolerance still reports
  `status: Optimal` through PuLP's mapping. Check `proven_optimal` before
  interpreting an objective.

## Known code issues

`demand_model.py` is dead code — nothing imports it, and it references config
names that no longer exist (`TOTAL_TRIPS_TARGET`, `N_EVTOL`). The live demand path
is `uam_data_pipeline.py`.

