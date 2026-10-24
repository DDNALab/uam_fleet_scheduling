"""
Correlated scenario generation for the two-stage stochastic program.

Draws on-demand requests by period and destination, incoming aircraft by
period, and their arrival state of charge from one shared, temporally smoothed
Gaussian factor, so demand and supply move together. A log link keeps every
Poisson rate positive, and arrivals lose state of charge as demand pressure on
the fleet grows. Advance-booked demand is observed before the first-stage
decisions and is not sampled.

The factor is scaled by the deterministic ``L2`` norm of the smoothing kernel,
so every period has variance ``sigma_g^2``; trajectories are never
re-standardized to their own realized variance.

Coupling switches: ``beta_Gamma = beta_g = 0`` gives the no-coupling baseline
(arrival SoC independent of demand stress); ``beta_p = beta_e = 0`` removes the
demand/arrival dependence.
"""

from dataclasses import dataclass

import numpy as np

import config
from economics import charging_duration_periods


# ============================================================
# Scenario objects
# ============================================================

@dataclass
class Aircraft:
    """
    One stochastic incoming aircraft, i.e. one element of ``J_s``.

    Attributes
    ----------
    id : int
        Index ``j`` inside the scenario.
    arrival_period : int
        1-based modeled arrival period ``r_js``.
    arrival_time : int
        Arrival clock time in minutes after midnight (reporting only).
    initial_soc : float
        Realized arrival state of charge ``SoC_js`` in percent.
    duration : int
        Required uninterrupted charging duration ``p_js``, in periods.
    energy : float
        Energy needed to reach the departure SoC, in kWh; the charging-cost
        basis ``c^chg_jts = energy * P(t)``.
    """

    id: int
    arrival_period: int
    initial_soc: float
    duration: int
    energy: float
    arrival_time: int = 0

    def to_block(self):
        """Dictionary form consumed by the optimization models."""
        return {
            "arrival_period": int(self.arrival_period),
            "initial_soc": float(self.initial_soc),
            "duration": int(self.duration),
            "energy": float(self.energy),
            "arrival_time": int(self.arrival_time),
        }


@dataclass
class Scenario:
    """
    One realization of ``(lambda^O_tds, N^e_ts, SoC_js)``.

    Attributes
    ----------
    id : int
    probability : float
        ``pi_s``; uniform for raw scenarios, reassigned by reduction.
    ondemand_demand : dict
        ``{(t, d): int}`` realized on-demand requests.
    aircraft : list of Aircraft
        The incoming set ``J_s``.
    common_factor : numpy.ndarray
        The shared factor ``g_ts`` by period.
    arrival_counts : numpy.ndarray
        Realized incoming arrivals ``N^e_ts`` by period.
    stress : numpy.ndarray
        Realized demand-to-seat stress ratio ``Gamma_ts`` by period.
    soc_mean : numpy.ndarray
        Conditional SoC mean by period, before truncation/draw.
    """

    id: int
    probability: float
    ondemand_demand: dict
    aircraft: list
    common_factor: np.ndarray
    arrival_counts: np.ndarray = None
    stress: np.ndarray = None
    soc_mean: np.ndarray = None

    # Diagnostics attached by reduction.
    feature_vector: np.ndarray = None
    proxy_score: float = None
    stratum: str = None

    # Convenience accessors.

    @property
    def ondemand_total(self):
        """Total realized on-demand requests in the scenario."""
        return int(sum(self.ondemand_demand.values()))

    def demand_trajectory(self, periods=None, destinations=None):
        """
        ``{destination: numpy.ndarray}`` of realized on-demand requests by
        period, used by reduction features and reporting.
        """
        destinations = (
            sorted({d for (_, d) in self.ondemand_demand})
            if destinations is None
            else list(destinations)
        )

        periods = (
            sorted({t for (t, _) in self.ondemand_demand})
            if periods is None
            else list(periods)
        )

        index = {t: i for i, t in enumerate(periods)}

        out = {d: np.zeros(len(periods)) for d in destinations}

        for (t, d), value in self.ondemand_demand.items():
            if d in out and t in index:
                out[d][index[t]] += float(value)

        return out

    def duration_workload(self, periods=None):
        """
        Charging workload ``sum_{j: r_js = t} p_js`` by arrival period.

        Aggregated by arrival period rather than by aircraft, so it is
        comparable across scenarios with different aircraft counts.
        """
        periods = (
            sorted({a.arrival_period for a in self.aircraft})
            if periods is None
            else list(periods)
        )

        index = {t: i for i, t in enumerate(periods)}

        workload = np.zeros(len(periods), dtype=float)

        for aircraft in self.aircraft:
            position = index.get(aircraft.arrival_period)

            if position is not None:
                workload[position] += float(aircraft.duration)

        return workload


# ============================================================
# Common temporal factor
# ============================================================

def gaussian_smoothing_kernel(sigma):
    """
    Normalized discrete Gaussian kernel for :func:`smooth_series`.

    Implemented locally so the generator depends only on NumPy.
    """
    sigma = float(sigma)

    if sigma <= 0:
        return np.array([1.0])

    radius = max(1, int(np.ceil(4.0 * sigma)))

    offsets = np.arange(-radius, radius + 1, dtype=float)

    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)

    return kernel / kernel.sum()


def smooth_series(values, sigma):
    """Centered Gaussian smoothing with edge-replicating padding."""
    values = np.asarray(values, dtype=float)

    kernel = gaussian_smoothing_kernel(sigma)

    if kernel.size == 1:
        return values.copy()

    radius = (kernel.size - 1) // 2

    padded = np.pad(values, radius, mode="edge")

    return np.convolve(padded, kernel, mode="valid")


def sample_common_factor(
    n_periods,
    rng,
    sigma=None,
    smoothing_sigma=None,
):
    """
    Shared Gaussian temporal factor ``g_ts``, one value per period.

    Each period has mean zero and variance ``sigma_g**2``: temporal correlation
    comes from Gaussian smoothing of independent draws, scaled by the kernel's
    ``L2`` norm. Trajectories are not re-standardized, which would change the
    marginal variance and invalidate the fixed log-link correction.
    """
    sigma = (
        config.FACTOR_STD
        if sigma is None
        else float(sigma)
    )

    smoothing_sigma = (
        config.FACTOR_SMOOTHING_SIGMA
        if smoothing_sigma is None
        else float(smoothing_sigma)
    )

    if sigma < 0:
        raise ValueError("factor standard deviation cannot be negative")

    n_periods = int(n_periods)

    if n_periods < 1:
        raise ValueError("n_periods must be at least one")

    # A normalized kernel, so smoothing preserves the scale of the draws.
    kernel = gaussian_smoothing_kernel(smoothing_sigma)

    # Extra draws at both ends so every retained period sees the full kernel.
    radius = (kernel.size - 1) // 2

    raw = rng.standard_normal(
        n_periods + 2 * radius
    )

    # Linear filtering of Gaussian draws stays Gaussian.
    smoothed = np.convolve(
        raw,
        kernel,
        mode="valid",
    )

    # Unit-variance draws leave variance sum(kernel[i]**2) after filtering.
    theoretical_std = float(
        np.sqrt(np.sum(kernel ** 2))
    )

    if theoretical_std <= 0:
        raise ValueError("invalid Gaussian smoothing kernel")

    # Scale by the deterministic kernel L2 norm; never re-standardize a
    # trajectory to its own realized variance.
    factor = sigma * smoothed / theoretical_std

    return factor


# Truncated normal sampler

def sample_truncated_normal(mean, sigma, lower, upper, rng, max_draws=10_000):
    """
    Draw from ``TN[lower, upper](mean, sigma)`` by rejection.

    Rejection sampling is exact here and needs no SciPy: with the support
    ``[20, 60]``, ``sigma_s = 6`` and means near 40-50, acceptance is well above
    0.9. The fallback clips the mean only if acceptance collapses.
    """
    lower = float(lower)
    upper = float(upper)

    if upper <= lower:
        return lower

    if sigma is None or sigma <= 0:
        return float(np.clip(mean, lower, upper))

    for _ in range(max_draws):
        draw = rng.normal(mean, sigma)

        if lower <= draw <= upper:
            return float(draw)

    return float(np.clip(mean, lower, upper))


# Intensities

def log_link(base_rate, factor, beta, factor_std):
    """
    ``base_rate * exp(beta * g - 1/2 beta^2 sigma_g^2)``.

    The correction term keeps ``E[exp(beta g)] == 1``, so the mean intensity
    equals the baseline rate whatever ``beta`` is.
    """
    beta = float(beta)

    return base_rate * np.exp(beta * factor - 0.5 * beta ** 2 * factor_std ** 2)


def sample_ondemand_intensity(
    ondemand_rate,
    periods,
    destinations,
    factor,
    beta,
    factor_std=None,
):
    """
    Expected on-demand rate per (period, destination) after the factor shock.

    Uses the same factor standard deviation as the generated temporal factor.
    """
    factor_std = (
        config.FACTOR_STD
        if factor_std is None
        else float(factor_std)
    )

    return {
        (t, d): log_link(
            ondemand_rate.get((t, d), 0.0),
            factor[t - 1],
            beta,
            factor_std,
        )
        for t in periods
        for d in destinations
    }


def sample_arrival_intensity(
    arrival_rate,
    factor,
    beta,
    factor_std=None,
):
    """
    Expected incoming-aircraft rate per period after the factor shock.

    Uses the same factor standard deviation as the generated temporal factor.
    """
    factor_std = (
        config.FACTOR_STD
        if factor_std is None
        else float(factor_std)
    )

    return log_link(
        arrival_rate,
        factor,
        beta,
        factor_std,
    )


# ============================================================
# Scenario generation
# ============================================================

def generate_scenarios(
    baseline=None,
    n_scenarios=None,
    seed=None,
    parameters=None,
    booked=None,
    destinations=None,
    periods=None,
):
    """
    Generate ``n_scenarios`` raw correlated scenarios.

    Parameters
    ----------
    baseline : dict
        Output of :func:`demand_model.build_baseline_demand`.
    n_scenarios : int, optional
        Defaults to ``parameters["raw_scenarios"]``.
    seed : int, optional
        Seed for ``numpy.random.default_rng``.
    parameters : dict, optional
        Parameter block from :func:`config.build_parameters`.
    booked : dict, optional
        ``{(t, d): int}`` known bookings, overriding ``baseline["booked"]``.
        Only the level matters: booked demand enters the stress ratio.
    destinations, periods : sequence, optional
        Explicit grids, overriding the baseline.

    Returns
    -------
    list of Scenario
        Realized on-demand matrix, incoming aircraft, common factor and the
        SoC/stress diagnostics.
    """
    parameters = config.build_parameters() if parameters is None else parameters

    if baseline is None:
        baseline = {}

    destinations = list(
        baseline.get("destinations", config.ACTIVE_DESTINATIONS)
        if destinations is None
        else destinations
    )

    periods = list(
        baseline.get("periods", parameters["periods"]) if periods is None else periods
    )

    periods = [int(t) for t in periods]

    booked = baseline.get("booked", {}) if booked is None else booked

    ondemand_rate = baseline.get("ondemand_rate", {})

    if not ondemand_rate:
        ondemand_rate = {(t, d): 0.0 for t in periods for d in destinations}

    n_scenarios = int(
        parameters["raw_scenarios"] if n_scenarios is None else n_scenarios
    )

    seed = int(parameters["seed"] if seed is None else seed)

    rng = np.random.default_rng(seed)

    correlation = parameters.get("correlation", {})
    soc_parameters = parameters.get("soc", {})

    beta_demand = float(correlation.get("demand_beta", config.DEMAND_FACTOR_BETA))
    beta_arrival = float(correlation.get("arrival_beta", config.ARRIVAL_FACTOR_BETA))
    factor_std = float(correlation.get("factor_std", config.FACTOR_STD))
    smoothing_sigma = float(
        correlation.get("smoothing_sigma", config.FACTOR_SMOOTHING_SIGMA)
    )

    soc_intercept = float(soc_parameters.get("intercept", config.SOC_MEAN_INTERCEPT))
    soc_stress = float(
        soc_parameters.get("stress_coefficient", config.SOC_STRESS_COEFFICIENT)
    )
    soc_factor = float(
        soc_parameters.get("factor_coefficient", config.SOC_FACTOR_COEFFICIENT)
    )
    soc_std = float(soc_parameters.get("std", config.SOC_STD_DEVIATION))
    soc_min = float(soc_parameters.get("min", config.ARRIVAL_SOC_MIN))
    soc_max = float(soc_parameters.get("max", config.ARRIVAL_SOC_MAX))

    nmax = int(parameters.get("evtol_capacity", config.EVTOL_CAPACITY))

    # Expected incoming-aircraft intensity, derived from this instance's
    # realized baseline demand so the aircraft/passenger ratio stays exact on
    # shortened validation horizons.
    baseline_passengers = float(sum(booked.values())) + float(
        sum(ondemand_rate.values())
    )

    ratio = parameters.get("aircraft_passenger_ratio")

    if ratio is not None and baseline_passengers > 0:

        expected_aircraft = float(ratio) * baseline_passengers

    else:

        expected_aircraft = float(
            parameters.get(
                "expected_incoming_aircraft", config.EXPECTED_INCOMING_AIRCRAFT
            )
        )

    from demand_model import aircraft_time_profile

    arrival_profile = aircraft_time_profile(
        n_periods=len(periods),
        bin_size=parameters.get("bin_size", config.BIN_SIZE),
        horizon_start=parameters.get("horizon", {}).get(
            "start", config.HORIZON_START
        ),
    )

    arrival_rate = expected_aircraft * arrival_profile

    horizon_start = parameters.get("horizon", {}).get("start", config.HORIZON_START)
    bin_size = parameters.get("bin_size", config.BIN_SIZE)

    probability = 1.0 / n_scenarios

    scenarios = []

    for scenario_id in range(n_scenarios):

        # Shared Gaussian factor for this scenario.
        factor = sample_common_factor(
            len(periods), rng, sigma=factor_std, smoothing_sigma=smoothing_sigma
        )

        # On-demand requests: log-link intensity, then a Poisson draw.
        intensity = sample_ondemand_intensity(
            ondemand_rate,
            periods,
            destinations,
            factor,
            beta_demand,
            factor_std=factor_std,
        )

        ondemand_demand = {
            key: int(rng.poisson(max(0.0, value)))
            for key, value in intensity.items()
        }

        # Incoming aircraft arrivals: Poisson draw on the shifted rate.
        arrival_intensity = sample_arrival_intensity(
            arrival_rate,
            factor,
            beta_arrival,
            factor_std=factor_std,
        )

        arrival_counts = np.array(
            [int(rng.poisson(max(0.0, value))) for value in arrival_intensity],
            dtype=int,
        )

        # Demand-to-seat stress ratio, with a one-seat floor.
        stress = np.zeros(len(periods), dtype=float)

        soc_mean = np.zeros(len(periods), dtype=float)

        for position, t in enumerate(periods):

            demand_in_period = sum(
                float(booked.get((t, d), 0)) + float(ondemand_demand.get((t, d), 0))
                for d in destinations
            )

            seat_supply = max(1.0, float(arrival_counts[position]) * nmax)

            stress[position] = demand_in_period / seat_supply

            # Conditional mean arrival SoC before truncation.
            soc_mean[position] = (
                soc_intercept
                - soc_stress * stress[position]
                - soc_factor * factor[position]
            )

        # Aircraft, truncated SoC draw and required charging duration.
        aircraft = []

        aircraft_id = 0

        for position, t in enumerate(periods):

            for _ in range(int(arrival_counts[position])):

                soc = sample_truncated_normal(
                    soc_mean[position], soc_std, soc_min, soc_max, rng
                )

                duration = charging_duration_periods(soc, parameters)

                aircraft.append(
                    Aircraft(
                        id=aircraft_id,
                        arrival_period=int(t),
                        arrival_time=int(horizon_start + (t - 1) * bin_size),
                        initial_soc=float(soc),
                        duration=int(duration),
                        energy=(
                            parameters["battery_capacity"]
                            * max(0.0, parameters["soc_min"] - soc)
                            / 100.0
                        ),
                    )
                )

                aircraft_id += 1

        scenarios.append(
            Scenario(
                id=scenario_id,
                probability=probability,
                ondemand_demand=ondemand_demand,
                aircraft=aircraft,
                common_factor=factor,
                arrival_counts=arrival_counts,
                stress=stress,
                soc_mean=soc_mean,
            )
        )

    return scenarios


# ============================================================
# Diagnostics
# ============================================================

def scenario_diagnostics(scenarios):
    """
    Sample moments of the generated ensemble.

    Keeps the no-coupling baseline and intermediate coupling settings
    comparable across validation runs.
    """
    if not scenarios:
        return {}

    requests = np.array([s.ondemand_total for s in scenarios], dtype=float)

    aircraft_counts = np.array([len(s.aircraft) for s in scenarios], dtype=float)

    socs = np.array(
        [a.initial_soc for s in scenarios for a in s.aircraft],
        dtype=float,
    )

    durations = np.array(
        [a.duration for s in scenarios for a in s.aircraft],
        dtype=float,
    )

    return {
        "scenarios": len(scenarios),
        "ondemand_mean": float(requests.mean()),
        "ondemand_std": float(requests.std()),
        "aircraft_mean": float(aircraft_counts.mean()),
        "aircraft_std": float(aircraft_counts.std()),
        "soc_mean": float(socs.mean()) if socs.size else float("nan"),
        "soc_std": float(socs.std()) if socs.size else float("nan"),
        "duration_counts": {
            int(h): int((durations == h).sum()) for h in np.unique(durations)
        },
        "mean_stress": float(
            np.mean([float(np.mean(s.stress)) for s in scenarios])
        ),
    }


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    from demand_model import build_baseline_demand

    parameters = config.build_parameters()

    baseline = build_baseline_demand()

    generated = generate_scenarios(baseline, n_scenarios=25, parameters=parameters)

    for scenario in generated[:3]:
        print(
            "S%d requests=%d aircraft=%d soc_mean=%.1f"
            % (
                scenario.id,
                scenario.ondemand_total,
                len(scenario.aircraft),
                float(np.mean([a.initial_soc for a in scenario.aircraft]))
                if scenario.aircraft
                else float("nan"),
            )
        )

    print(scenario_diagnostics(generated))
