"""
Energy, fare and charging-duration primitives shared by both formulations.

Everything here is a pure function of data treated as known (time and cost
separability). Keeping it in one place means the aircraft-indexed model and the
duration-aggregated reformulation charge the same price for the same charging
operation.

Charging duration is a continuous energy balance rounded up to whole periods. A
charge that needs only a fraction of a period still occupies a charger-power
unit for that whole period (nonpreemptive discrete charging), so the modeled
duration is ``max(1, ceil(...))``. With the default parameters the unclamped
expression already lands in {1, 2} at 15-minute resolution; the clamp only
matters for an aircraft that arrives at or above the required departure SoC.
"""

import math

import config
# ============================================================
# Charging duration
# ============================================================

def energy_per_period_kwh(params=None):
    """Energy delivered by one charging period: ``mu * Delta`` (kWh)."""
    rate = config.CHARGING_RATE if params is None else params["charging_rate"]
    bin_size = config.BIN_SIZE if params is None else params["bin_size"]

    return rate * (bin_size / 60.0)


def charging_energy_kwh(soc, params=None):
    """``E_bar * max(0, S_req - SoC) / 100`` in kWh: the energy a charge must deliver."""
    battery = config.BATTERY_CAPACITY if params is None else params["battery_capacity"]
    soc_req = config.SOC_REQUIRED if params is None else params["soc_min"]

    return battery * max(0.0, soc_req - float(soc)) / 100.0


def charging_duration_periods(soc, params=None):
    """
    Required charging duration ``p`` in whole periods.

    ``p = ceil( E_bar (S_req - SoC)^+ / (mu Delta) )``, clamped to at least one
    period: an incoming aircraft selected for service performs one
    uninterrupted charging operation, and a charger-power unit is occupied for
    the whole period in which it is activated.

    Parameters
    ----------
    soc : float
        Arrival state of charge, in percent.
    params : dict, optional
        Parameter block carrying ``battery_capacity``, ``soc_min``,
        ``charging_rate`` and ``bin_size``. Falls back to :mod:`config`.

    Returns
    -------
    int
        Number of periods ``p_js``.
    """
    energy = charging_energy_kwh(soc, params)

    per_period = energy_per_period_kwh(params)

    if per_period <= 0:
        raise ValueError("charging_rate * bin_size must be positive")

    return max(1, int(math.ceil(energy / per_period - 1e-9)))


#: Backwards-compatible alias (older code imported this name).
charging_periods_required = charging_duration_periods


# ============================================================
# Electricity tariff
# ============================================================

def _hour_in_windows(hour, windows):
    """True when ``hour`` falls in any half-open ``[start, end)`` window."""
    for start, end in windows:
        if start <= hour < end:
            return True

    return False


def time_of_use_multiplier(hour, params=None):
    """
    Tariff multiplier ``m_k`` for a clock hour.

    ``P(t) = P_base * m_peak`` for 16 <= t < 21, ``P_base * m_off`` for
    0 <= t < 6 or 21 <= t < 24, and ``P_base * m_shoulder`` otherwise.
    """
    if params is None:
        peak = config.M_PEAK
        shoulder = config.M_SHOULDER
        off = config.M_OFF
        peak_windows = config.PEAK_HOUR_WINDOWS
        off_windows = config.OFF_HOUR_WINDOWS
    else:
        peak = params["peak_multiplier"]
        shoulder = params["shoulder_multiplier"]
        off = params["off_multiplier"]
        peak_windows = params.get("peak_hour_windows", config.PEAK_HOUR_WINDOWS)
        off_windows = params.get("off_hour_windows", config.OFF_HOUR_WINDOWS)

    if _hour_in_windows(hour, peak_windows):
        return peak

    if _hour_in_windows(hour, off_windows):
        return off

    return shoulder


def energy_price_per_kwh(hour, params=None):
    """
    Time-of-use electricity price ``P(t)`` in $/kWh.

    ``hour`` is the clock hour (fractional is fine) of the *charging start*
    period, consistent with ``c^chg_jts`` depending on ``t`` only.
    """
    if params is None:
        base = config.BASE_PRICE
    else:
        base = params["base_price"]

    return base * time_of_use_multiplier(hour, params)


def period_hour(t, params=None):
    """
    Clock hour of modeled period ``t`` (1-based), used to price charging that
    starts in ``t``.
    """
    if params is None:
        start = config.HORIZON_START
        bin_size = config.BIN_SIZE
    else:
        start = params["horizon"]["start"]
        bin_size = params["horizon"]["bin_size"]

    return (start + (int(t) - 1) * bin_size) / 60.0


def charging_cost_per_kwh(t, params=None):
    """Convenience wrapper: price of one kWh for a charge starting in period t."""
    return energy_price_per_kwh(period_hour(t, params), params)


# ============================================================
# Passenger fare
# ============================================================

def fare_for_distance(distance_miles, params=None):
    """
    Passenger fare ``rho_d = 20 + 3 Dist_d`` in dollars for the base case.

    The intercept and per-mile rate come from :mod:`config` unless ``params``
    overrides them.
    """
    if params is None:
        intercept = config.FARE_INTERCEPT
        per_mile = config.FARE_PER_MILE
    else:
        intercept = params.get("fare_intercept", config.FARE_INTERCEPT)
        per_mile = params.get("fare_per_mile", config.FARE_PER_MILE)

    return intercept + per_mile * float(distance_miles)


# ============================================================
# Flight operating cost  c^flt_d
# ============================================================

def flight_operating_cost(distance_miles, params=None):
    """
    Destination-specific flight operating cost ``c^flt_d``.

    Linear in distance, ``base + per_mile * distance_miles``, with the base and
    per-mile defaults from :mod:`config` (``FLIGHT_COST_BASE``,
    ``FLIGHT_COST_PER_MILE``).
    """
    if params is None:
        base = config.FLIGHT_COST_BASE
        per_mile = config.FLIGHT_COST_PER_MILE
    else:
        base = params.get("flight_cost_base", config.FLIGHT_COST_BASE)
        per_mile = params.get("flight_cost_per_mile", config.FLIGHT_COST_PER_MILE)

    return base + per_mile * float(distance_miles)


# ============================================================
# Geometry
# ============================================================

EARTH_RADIUS_MILES = 3958.8


def haversine_miles(lat1, lon1, lat2, lon2):
    """Great-circle distance in statute miles."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )

    return 2.0 * EARTH_RADIUS_MILES * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def destination_distances(vertiports=None, hub=None):
    """
    Road-free great-circle distance from the hub to every non-hub vertiport.

    Returns
    -------
    dict
        ``{destination_name: distance_in_miles}``
    """
    vertiports = config.VERTIPORTS_RAW if vertiports is None else vertiports
    hub = config.HUB_NAME if hub is None else hub

    hub_lat, hub_lon = vertiports[hub]

    return {
        name: haversine_miles(hub_lat, hub_lon, lat, lon)
        for name, (lat, lon) in vertiports.items()
        if name != hub
    }


def destination_fares(vertiports=None, hub=None, params=None):
    """
    Fare table ``{destination: rho_d}`` from the hub-to-destination distances.
    """
    distances = destination_distances(vertiports, hub)

    return {
        name: fare_for_distance(miles, params)
        for name, miles in distances.items()
    }
