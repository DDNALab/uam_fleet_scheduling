# Coordinated eVTOL Scheduling

This source distribution implements a two-stage stochastic UAM scheduling
workflow. First-stage decisions commit destination/time departures and reserve
charging capacity. Scenario-dependent recourse then assigns aircraft to
charging and flights, serves passengers, and prices additions, cancellations,
emergency capacity, and CVaR service loss.

See the repository-level README and `docs/MODEL_NOTES.md` for the full workflow,
data provenance, and remaining research-calibration checks.
