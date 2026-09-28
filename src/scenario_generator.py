"""Scenario generation for hybrid advance-booked + on-demand UAM service.

Advance bookings are known before Stage 1.  Scenario uncertainty contains
on-demand requests, incoming-aircraft counts, and arrival SoC.  A common
factor drives both request and aircraft-arrival intensities; arrival SoC is
then conditionally sampled from a truncated normal whose mean decreases with
realized demand-to-seat stress and the common factor.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import truncnorm

from .config import (
    OPTIMIZATION_SCENARIOS,
    RANDOM_SEED,
    BIN_SIZE,
    HORIZON_START,
    ARRIVAL_SOC_MIN,
    ARRIVAL_SOC_MAX,
    SOC_BASE_MEAN,
    SOC_STRESS_BETA,
    SOC_COMMON_FACTOR_BETA,
    SOC_STD,
    AIRCRAFT_PASSENGER_RATIO,
    ADVANCE_BOOKING_FRACTION,
    EVTOL_CAPACITY,
)


@dataclass
class Aircraft:
    id: int
    arrival_time: int
    initial_soc: float


@dataclass
class Scenario:
    id: int
    probability: float
    advance_bookings: dict
    on_demand_demand: dict
    aircraft: list
    common_factor: np.ndarray

    @property
    def passenger_demand(self):
        """Total realized demand; retained for diagnostics/reduction compatibility."""
        keys = set(self.advance_bookings) | set(self.on_demand_demand)
        return {
            key: int(self.advance_bookings.get(key, 0) + self.on_demand_demand.get(key, 0))
            for key in keys
        }


def generate_common_factor(n_periods, seed=None):
    if seed is not None:
        np.random.seed(seed)
    g = np.random.normal(0, 1, n_periods)
    g = gaussian_filter1d(g, sigma=1)
    g = g - np.mean(g)
    std = np.std(g)
    return g / std if std > 0 else g


def _largest_remainder_allocation(total, weights):
    """Allocate an integer total across periods while preserving the profile."""
    weights = np.asarray(weights, dtype=float)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    weights = weights / weights.sum()
    raw = total * weights
    allocated = np.floor(raw).astype(int)
    remainder = int(total - allocated.sum())
    if remainder > 0:
        order = np.argsort(-(raw - allocated))
        allocated[order[:remainder]] += 1
    return allocated


def generate_advance_bookings(expected_demand, time_profile, booking_fraction=ADVANCE_BOOKING_FRACTION):
    """Known Stage-1 booked demand B[r,d]."""
    bookings = {}
    for destination, daily_value in expected_demand.items():
        booked_total = int(round(float(daily_value) * booking_fraction))
        counts = _largest_remainder_allocation(booked_total, time_profile)
        for idx, value in enumerate(counts, start=1):
            bookings[idx, destination] = int(value)
    return bookings


def generate_on_demand_demand(
    expected_demand,
    time_profile,
    common_factor,
    booking_fraction=ADVANCE_BOOKING_FRACTION,
    beta=0.35,
):
    """Scenario-dependent same-day/on-demand requests."""
    demand = {}
    for destination, daily_value in expected_demand.items():
        residual_daily = float(daily_value) * (1.0 - booking_fraction)
        for t in range(1, len(time_profile) + 1):
            base = residual_daily * time_profile[t - 1]
            intensity = base * np.exp(beta * common_factor[t - 1] - 0.5 * beta**2)
            demand[t, destination] = int(np.random.poisson(max(intensity, 0.001)))
    return demand


def conditional_soc_mean(stress_ratio, common_factor):
    """Mean of the bounded conditional arrival-SoC distribution, in percent."""
    return (
        SOC_BASE_MEAN
        - SOC_STRESS_BETA * float(stress_ratio)
        - SOC_COMMON_FACTOR_BETA * float(common_factor)
    )


def sample_arrival_soc(stress_ratio, common_factor):
    """Draw arrival SoC from the truncated-normal model used in the manuscript."""
    mean = conditional_soc_mean(stress_ratio, common_factor)
    if SOC_STD <= 0:
        return float(np.clip(mean, ARRIVAL_SOC_MIN, ARRIVAL_SOC_MAX))
    a = (ARRIVAL_SOC_MIN - mean) / SOC_STD
    b = (ARRIVAL_SOC_MAX - mean) / SOC_STD
    return float(truncnorm.rvs(a, b, loc=mean, scale=SOC_STD))


def _total_demand_by_period(advance_bookings, on_demand_demand, n_periods):
    total = np.zeros(n_periods, dtype=float)
    for t in range(1, n_periods + 1):
        total[t - 1] = sum(
            value
            for (period, _destination), value in advance_bookings.items()
            if period == t
        ) + sum(
            value
            for (period, _destination), value in on_demand_demand.items()
            if period == t
        )
    return total


def generate_aircraft(expected_total_passengers, common_factor, total_demand_by_period,
                      beta=0.35, *, aircraft_passenger_ratio=None,
                      horizon_start=HORIZON_START):
    """Generate arrival counts first, then condition each arriving aircraft's SoC on stress."""
    periods = len(common_factor)
    # Arrival supply already grows with total passengers: do not multiply the
    # ratio by the intensity multiplier a second time in the scaling runner.
    ratio = AIRCRAFT_PASSENGER_RATIO if aircraft_passenger_ratio is None else aircraft_passenger_ratio
    if not np.isfinite(ratio) or ratio < 0:
        raise ValueError("aircraft_passenger_ratio must be finite and nonnegative")
    expected_evtols = expected_total_passengers * ratio
    base_rate = expected_evtols / periods
    rates = [
        max(base_rate * np.exp(beta * common_factor[t] - 0.5 * beta**2), 0.01)
        for t in range(periods)
    ]
    arrivals = np.random.poisson(rates)

    aircraft = []
    aircraft_id = 0
    for period, number in enumerate(arrivals):
        seat_supply = int(number) * EVTOL_CAPACITY
        stress_ratio = float(total_demand_by_period[period]) / max(1, seat_supply)
        for _ in range(int(number)):
            aircraft.append(
                Aircraft(
                    id=aircraft_id,
                    arrival_time=horizon_start + period * BIN_SIZE,
                    initial_soc=sample_arrival_soc(stress_ratio, common_factor[period]),
                )
            )
            aircraft_id += 1
    return aircraft


def generate_scenarios(
    expected_demand,
    time_profile,
    n_scenarios=OPTIMIZATION_SCENARIOS,
    seed=RANDOM_SEED,
    booking_fraction=ADVANCE_BOOKING_FRACTION,
    *, aircraft_passenger_ratio=None, horizon_start=HORIZON_START,
):
    """Generate scenarios sharing known bookings but differing in Stage-2 uncertainty."""
    np.random.seed(seed)
    if not 0 <= booking_fraction <= 1:
        raise ValueError("booking_fraction must be within [0, 1]")
    time_profile = np.asarray(time_profile, dtype=float)
    if (not len(time_profile) or not np.isfinite(time_profile).all()
            or (time_profile < 0).any() or time_profile.sum() <= 0):
        raise ValueError("time_profile must have positive finite nonnegative weights")
    time_profile = time_profile / time_profile.sum()
    n_periods = len(time_profile)
    probability = 1.0 / n_scenarios
    bookings = generate_advance_bookings(expected_demand, time_profile, booking_fraction)

    scenarios = []
    for s in range(n_scenarios):
        common_factor = generate_common_factor(n_periods)
        on_demand = generate_on_demand_demand(
            expected_demand, time_profile, common_factor, booking_fraction
        )
        total_by_period = _total_demand_by_period(bookings, on_demand, n_periods)
        aircraft = generate_aircraft(
            sum(expected_demand.values()), common_factor, total_by_period,
            aircraft_passenger_ratio=aircraft_passenger_ratio,
            horizon_start=horizon_start,
        )
        scenarios.append(
            Scenario(
                id=s,
                probability=probability,
                advance_bookings=dict(bookings),
                on_demand_demand=on_demand,
                aircraft=aircraft,
                common_factor=common_factor,
            )
        )
    return scenarios
