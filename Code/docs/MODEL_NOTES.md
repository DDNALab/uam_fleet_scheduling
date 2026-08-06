# Model notes and validation priorities

The repository reorganization preserves the original research workflow. Two
portability defects were corrected: solver selection no longer contains a
personal Windows path, and scenario charging cost is now indexed by the scenario
currently being summed in the objective.

Before treating results as publication-ready, validate the following modeling
relationships against the mathematical formulation in the paper:

1. Link passenger-assignment variables to destination-choice and service
   indicator variables (`z` and `Y`). In the inherited formulation, those
   variables are created but are not fully coupled to passenger assignments.
2. Confirm that departure time is constrained to charging completion and that
   level-of-service constraints activate whenever passengers are assigned.
3. Confirm the intended treatment of aircraft requiring zero charging periods.
4. Verify the sign and interpretation of the CVaR term and the units of its risk
   weight.
5. Validate gravity-model calibration, demand adoption assumptions, and fare
   construction against observed or cited data.

These are model-validation questions rather than repository-layout changes, so
they were intentionally not reformulated during this organization pass.

