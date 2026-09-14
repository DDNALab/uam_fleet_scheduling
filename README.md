# Coordinated eVTOL Scheduling under Demand Uncertainty

This repository contains a reproducible research workflow for urban air
mobility demand estimation, stochastic scenario generation and reduction, and
risk-aware eVTOL charging and passenger scheduling in the Dallas-Fort Worth
region.

The project combines geospatial Census inputs, a gravity-based demand model,
non-homogeneous Poisson arrival scenarios, backward scenario reduction, and a
mixed-integer linear program with a CVaR risk term.

> **Research status:** This is an academic model under active validation. The
> included data and parameters support computational experiments; they should
> not be interpreted as an operational deployment plan.

## Workflow

```mermaid
flowchart LR
    A[ACS population + Census tracts] --> B[Geospatial allocation]
    B --> C[Gravity demand model]
    C --> D[Passenger and eVTOL arrivals]
    D --> E[Joint scenario generation]
    E --> F[Backward scenario reduction]
    F --> G[Conditional state-of-charge sampling]
    G --> H[Stochastic MILP + CVaR]
    H --> I[Results and interactive figures]
```

| Stage | Module |
|---|---|
| Geospatial allocation, gravity demand | `uam_data_pipeline.py` |
| Scenario generation | `scenario_generator.py` |
| Scenario reduction | `scenario_reduction.py` |
| Solver-ready structures | `model_builder.py` |
| Two-stage MILP, CVaR, solver selection | `stochastic_model.py` |
| Objective breakdown, variable dump | `solution_export.py` |
| Figures | `visualization.py` |
| Orchestration | `experiment_runner.py` |

## Repository structure

```text
coord-schedule-uam/
├── README.md                        # This file
├── literature/                      # Source papers
└── Code/
    ├── Price_EvtolPass_Sto.ipynb    # Exploratory notebook
    ├── run_experiment.py            # Source-checkout launcher
    ├── run_tests.py                 # Test launcher
    ├── pyproject.toml
    ├── requirements.txt
    ├── data/
    │   ├── raw/                     # ACS JSON and TIGER/Line tract files
    │   └── README.md                # Data dictionary and provenance notes
    ├── docs/
    │   └── MODEL_NOTES.md           # Scientific validation priorities
    ├── outputs/
    │   ├── figures/                 # Generated interactive HTML plots
    │   └── results/                 # Run summaries and solution dumps
    ├── src/coord_schedule_uam/
    │   ├── config.py                # Parameters and repository paths
    │   ├── uam_data_pipeline.py     # Census, geospatial, gravity demand
    │   ├── demand_model.py          # Gravity model helpers (unused)
    │   ├── scenario_generator.py    # Stochastic demand and arrival scenarios
    │   ├── scenario_reduction.py    # Backward reduction
    │   ├── model_builder.py         # Solver-ready scenario structures
    │   ├── stochastic_model.py      # Two-stage MILP, CVaR, solver selection
    │   ├── solution_export.py       # Objective breakdown and variable dump
    │   ├── visualization.py         # Plotly and Folium outputs
    │   └── experiment_runner.py     # End-to-end orchestration
    └── tests/                       # Fast unit and configuration tests
```

Generated outputs are ignored by Git. They can be recreated from the
documented workflow.

> **Note:** `demand_model.py` is currently dead code — nothing imports it, and it
> references config names that no longer exist (`TOTAL_TRIPS_TARGET`,
> `N_EVTOL`). The live demand path is `uam_data_pipeline.py`.

## Installation

`Code/pyproject.toml` declares `requires-python = ">=3.10"`. The validated
working environment is **Python 3.8.5** under Anaconda, which runs the pipeline
despite the declared floor — adjust the requirement or the interpreter so the two
agree before packaging.

Core dependencies are in `Code/requirements.txt`:

```text
folium, geopandas, highspy, matplotlib, numpy,
pandas, plotly, pulp, scipy, shapely
```

`highspy` provides the HiGHS MILP backend and is the recommended solver, since it
is open source and has no model-size cap. `gurobipy` is **optional** — add it only
with an unrestricted license (see [Solvers](#solvers)).

From the `Code/` directory:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS or Linux
source .venv/bin/activate
```

Install the project:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

For tests and linting tools:

```bash
python -m pip install -e ".[dev]"
```

## Running the workflow

### Directly from a source checkout

The launcher lives in `Code/` and makes `src/` importable without an editable
install. From `Code/`, run:

```bash
python run_experiment.py
```

Build the pipeline and solve, choose a reproducible seed, and pick a solver:

```bash
python run_experiment.py --seed 60 --solver highs
```

Smoke check that builds the optimization input without solving the MILP:

```bash
python run_experiment.py --skip-solve
```

Full option list:

```text
--seed SEED
--skip-solve
--solver {auto,gurobi,highs,cbc}
```

Do not launch files inside `src/coord_schedule_uam/` directly. They are package
modules and use relative imports such as `from .config import ...`.

### After installing the package

The same CLI is available as a module:

```bash
python -m coord_schedule_uam --seed 60
```

## Solvers

The MILP is solved through PuLP. Selection walks a fixed preference order and
uses the first solver that is both installed and able to hold the model:

```
SOLVER_PREFERENCE = ("GUROBI", "HIGHS", "CBC")
```

`--solver` overrides the order:

```bash
python run_experiment.py --solver highs
python run_experiment.py --solver cbc
python run_experiment.py --solver auto    # default
```

A solver that is skipped is reported with the reason, so a fallback is never
silent:

```
[info] skipping GUROBI -- size-limited license: model has 4104 vars /
       8000 constraints, trial cap is 2000 each
[info] solving with HIGHS
```

### Gurobi's size-limited license

The default Gurobi trial license caps a model at **2000 variables and 2000
constraints**. The stochastic extensive form is far larger than that, so a naive
`pulp.GUROBI()` call fails deep inside gurobipy:

```
gurobipy.GurobiError: Model too large for size-limited license;
visit https://gurobi.com/unrestricted for more information
```

`solve_model` therefore projects the model size onto the cap *before* building
the Gurobi solver and rejects it if exceeded. To actually run Gurobi you need one
of:

- an unrestricted Gurobi license (commercial or academic), or
- **HiGHS**, which is open source, unbounded, and already satisfies the
  `highspy` requirement — this is the practical choice.

**Version note.** A `gurobipy` wheel must match the installed Gurobi runtime. A
mismatch shows up as `ImportError: DLL load failed while importing gurobipy` —
for example `gurobipy` 11.x against a Gurobi 12 installation, which needs
`gurobi110.dll` while only `gurobi120.dll` exists. Install the matching wheel:

```bash
python -m pip install --upgrade "gurobipy==12.0.0"
```

To use a specific HiGHS executable rather than `highspy`, set `HIGHS_PATH`:

```powershell
$env:HIGHS_PATH = "C:\path\to\highs.exe"
python run_experiment.py --solver highs
```

## Outputs

### Console report

At the end of a solve the runner prints an objective breakdown, passenger
service totals and the CVaR block directly to stdout — no files required:

```text
================================================================
OBJECTIVE BREAKDOWN
================================================================
  Expected revenue                         631.25
  Expected charging cost                   -10.73
  Expected unserved penalty                -26.67
  Expected added-departure cost             -9.73
  Expected cancelled-departure cost         -0.00
  Expected emergency-capacity cost          -2.00
                                     ------------
  EXPECTED PROFIT                          582.12
  First-stage commitment cost               -0.00
  First-stage reservation cost              -5.00
  CVaR risk penalty                        -44.00
                                     ------------
  OBJECTIVE (model)                        533.12
  OBJECTIVE (solver)                       533.12

================================================================
PASSENGER SERVICE
================================================================
  Total demand (sum over scenarios)            31
  Served                                    29.00
  Unserved                                   2.00
  Service rate                              93.5%
  Probability-weighted unserved              0.27

  per scenario:
    scn    prob     demand   served   unserved profit
    3      0.433    10       10.00    0.00     584.60
    21     0.433    13       13.00    0.00     715.50
    29     0.133    8        6.00     2.00     140.58

================================================================
CVaR
================================================================
  L_s per scenario                   {3: 0.0, 21: 0.0, 29: 2.0}
  VaR at alpha=0.90                        2.0000
  CVaR                                     2.0000
  lambda/(1-alpha) multiplier               220.0
  CVaR penalty                            44.0000
```

`OBJECTIVE (model)` is recomputed from the solved variable values, so it
matching `OBJECTIVE (solver)` confirms the breakdown accounts for every term.

Call it directly from other code:

```python
from coord_schedule_uam.solution_export import print_solution_report
print_solution_report(model_data, objective=obj, solver="HIGHS", status="Optimal")
```

### Files

- `outputs/results/run_summary.json`: seed, scenario counts, solver status, and
  objective value.
- `outputs/results/solution_dump/`: per-variable CSVs (activation, charging,
  departures, passengers, unserved, adjustments) plus `objective_breakdown.csv`.
- `outputs/figures/1_scenario_arrivals.html`: reduced arrival profiles.
- `outputs/figures/4_reduced_scenario_analysis.html`: four-panel comparison of
  the original and reduced scenario sets.

A CSV that is locked by another program (typically open in Excel) produces a
warning and is skipped rather than aborting the run — the console report still
carries every number.

Generated outputs are intentionally ignored by Git so that commits remain
small and reviewable.

### Reading solver logs

HiGHS logs the objective in its **internal minimization form**, so the sign is
flipped relative to the model:

```text
Primal bound      -533.121319638     # true value is +533.121319638
Dual bound        -533.288150751     # true value is +533.288150751
```

This is a convention of the MPS interface, not an error. PuLP reports it in the
model's own sense, and the two agree — verified against a model with a known
optimum. Gurobi and CBC log the objective directly.

Always check `proven_optimal` before interpreting an objective. A run that stops
on the time limit or gap tolerance still reports `status: Optimal` through PuLP's
status mapping, so a large residual gap is the signal that the value is not a
true optimum:

## Configuration

Experiment parameters are centralized in
`src/coord_schedule_uam/config.py`, including:

- time-of-use electricity prices;
- battery, charging, and aircraft capacity;
- experiment horizon and time-bin size;
- fleet and passenger counts;
- scenario generation and reduction sizes;
- CVaR-related model inputs;
- study-area bounds and vertiport coordinates.

All file paths are derived from the repository root. The workflow therefore
works from any current working directory after installation.

## Testing

From `Code/`:

```bash
python run_tests.py
python -m compileall -q src
```

The test launcher in `Code/` adds `src/` to Python's import path automatically,
so an editable package installation is not required. The current tests cover
bundled-data paths, horizon configuration, time-of-use pricing, and scenario
probability preservation. Add regression tests for model outputs — in particular
that demand keys span exactly `1..n_periods` — as the formulation stabilizes.

## Data and reproducibility

The workflow reads a local 2022 ACS JSON snapshot and the Texas 2022
TIGER/Line census tract shapefile. It does not require a Census API key. See
`data/README.md` for details.

Randomness is seeded by the experiment runner. Solver versions and operating
systems can still produce small numerical differences, so record the Python
environment and solver version when reporting results.

## Known validation work

The repository layout and execution environment are productionized, but the
research formulation still needs targeted mathematical validation. See
`docs/MODEL_NOTES.md` before using results in a publication.

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

Raising `CHARGING_RATE` (120 → 250 kW) and lowering `SOC_REQUIRED` (70 → 55)
both leave usable-aircraft unchanged — charging is **not** binding. Note also
that `soc_min` drives the charging *cost* term, so lowering it makes charging
cheaper without improving service.

### Period indexing

Demand periods are **1-based** and must stay aligned with both the model's
period set, `range(1, n_periods + 1)`, and the 1-based `arrival_period` computed
in `model_builder.py`. `generate_passenger_demand` previously used
`for t in range(periods)`, which keyed demand at 0…7 and silently dropped the
period-0 passengers while inventing an empty period 8. If you change the
horizon, re-check that demand keys span exactly `1..n_periods`.

### Horizon units

`HORIZON_START` and `HORIZON_END` are currently in **minutes** (`480` and `720`,
i.e. 08:00–12:00), so `N_PERIODS = (HORIZON_END - HORIZON_START) // BIN_SIZE`.

Earlier revisions stored these in **seconds** while dividing by a `BIN_SIZE`
expressed in minutes. That mix produced 15-*second* periods and a horizon one
eighth of the intended length, so every timing result depended on which unit the
author had in mind. Two guards:

1. `N_PERIODS` must equal `(HORIZON_END - HORIZON_START) // BIN_SIZE` with both
   operands in the same unit.
2. The config comment must match the arithmetic — `720` is 12:00, and the inline
   comment currently reads 10:00.

The config, the demand time profile in `uam_data_pipeline.py` and
`scenario_generator.py` must all agree before any timing result is trusted.
Note these constants are imported **by value**, so a programmatic override must
patch the name in each importing module, not just `config`.

### Scenario counts

With only 3 retained scenarios, the 0.90 CVaR tail is a **single scenario**, so
`VaR = CVaR` and the risk term reflects one outcome. Increase
`TARGET_SCENARIOS` before drawing conclusions about tail risk.

## License

No open-source license has been selected. Add an appropriate `LICENSE` file
before distributing this repository publicly.
