# Model notes and validation priorities

The package now implements the aircraft-indexed two-stage commitment--recourse
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

