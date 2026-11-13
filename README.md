# Coordinated eVTOL Scheduling under Demand Uncertainty

Research code for urban air mobility in the Dallas–Fort Worth region: Census-based
demand, correlated arrival scenarios, scenario reduction, and a two-stage
risk-averse (CVaR) MILP for charging, fleet activation and passenger service.

## Repository structure

```text
coord_schedule_UAM/
├── run_experiment.py          # main entry point
├── run_tests.py               # test suite
├── requirements.txt           # Python dependencies
├── input/                     # ACS 2022 JSON + Texas census tracts
├── output/                    # generated results (ignored by Git)
├── docs/MODEL_NOTES.md        # assumptions and open validation work
├── tests/                     # fast unit tests
└── src/                       # all code: plain modules, not a package
    ├── experiment_runner.py   # command line interface, runs the pipeline
    ├── uam_data_pipeline.py   # spatial demand: population → vertiports → OD flows
    ├── demand_model.py        # legacy temporal demand variant (unused)
    ├── economics.py           # fares, energy prices, charging duration
    ├── scenario_generator.py  # passenger and aircraft arrival scenarios
    ├── scenario_reduction.py  # tail-preserving scenario reduction
    ├── model_builder.py       # solver-ready input
    ├── stochastic_model.py    # aircraft-indexed MILP (paper §4.2)
    ├── aggregated_model.py    # duration-aggregated formulation (paper §4.3)
    ├── feasibility.py         # first-stage feasibility screening
    ├── solution_export.py     # objective breakdown and CSV export
    └── visualization.py       # HTML figures
```

## Installation

Python 3.10 or newer. From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`highspy` provides the solver and needs no licence.

## How to run

Run every command from the repository root with the environment activated.

### Step 1 — run the model

```powershell
python run_experiment.py
```

Defaults: aggregated formulation, automatic solver, seed 60. It takes about a
minute and prints the objective breakdown, passenger service and CVaR. 

`--skip-solve` builds the model and checks the data pipeline without solving.

### Step 2 — pick the formulation

The aggregated formulation is the default because it is much smaller. The indexed
one is the exact reference from paper §4.2, and is only practical on small
instances.

| Formulation | Use it for | Command |
|---|---|---|
| `aggregated` | normal runs and experiments  | `python run_experiment.py` |
| `indexed` | checking the reference model  | `python run_experiment.py --formulation indexed` |



### Step 3 — pick the solver

`--solver auto` (the default) tries Gurobi, then HiGHS, then CBC, so this step is
optional. HiGHS is the practical choice; CBC is markedly slower and returns worse
solutions on the default instance.

| Solver | Needs | Command |
|---|---|---|
| HiGHS | comes with `highspy`, no licence | `python run_experiment.py --solver highs` |
| CBC | comes with `pulp` | `python run_experiment.py --solver cbc` |
| Gurobi | `gurobipy` and a licence | `python run_experiment.py --solver gurobi` |

Combine both choices in any order, separated by spaces — `--solver` can be added
to any command above.

`python run_experiment.py --help` lists every flag, including `--seed`,
`--export`, `--gap` and `--time-limit`.


## Outputs

- Console: objective breakdown, served/unserved passengers, CVaR.
- `output/results/run_summary.json` — seed, scenarios, solver status, objective.
- With `--export`: `output/results/solution_dump/*.csv`.

## Data

The 2022 ACS snapshot and the Texas census tract shapefile in `input/` are all the
workflow needs; no Census API key is required. See `input/README.md`.
