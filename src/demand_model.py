"""
Demand construction for the case study.

Builds the baseline expected demand per period and destination from a two-peak
temporal profile, gravity destination shares and the advance-booking split, plus
the fare and distance tables. Spatial demand lives in :mod:`uam_data_pipeline`;
this module only needs NumPy so it stays importable in the modelling path.

``lambda_bar_td`` is the *total* expected demand in period ``t`` for destination
``d``. Booked demand ``B_td`` is known before the first-stage decisions; the
on-demand component is only an intensity, sampled into scenarios by
:mod:`scenario_generator`.
"""

import numpy as np

import config
from economics import destination_distances, destination_fares


# ============================================================
# Temporal profile
# ============================================================
# Two-peak Gaussian mixture plus a small background so no modeled period has a
# structurally zero intensity. Normalized so ``profile.sum() == 1``.

#: (centre_minutes, sigma_minutes, weight) for each peak of the passenger curve.
PASSENGER_PEAKS = (
    (8 * 60, 55.0, 1.00),
    (16 * 60, 70.0, 0.85),
)

PASSENGER_BASELINE = 0.08

#: Incoming-aircraft arrivals are smoother and wider than passenger demand,
#: so the supply curve does not mechanically track the demand curve.
AIRCRAFT_PEAKS = (
    (8 * 60, 90.0, 1.00),
    (16 * 60, 95.0, 1.00),
)

AIRCRAFT_BASELINE = 0.02


def _gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def time_profile(
    n_periods=None,
    bin_size=None,
    horizon_start=None,
    peaks=PASSENGER_PEAKS,
    baseline=PASSENGER_BASELINE,
):
    """
    Normalized period weights for the modeled operating window.

    Parameters
    ----------
    n_periods : int, optional
        Number of modeled periods. Defaults to ``config.N_PERIODS``.
    bin_size : int, optional
        Period length in minutes. Defaults to ``config.BIN_SIZE``.
    horizon_start : int, optional
        Window start in minutes after midnight. Defaults to
        ``config.HORIZON_START``.
    peaks : sequence of (centre, sigma, weight), optional
        Gaussian peaks in minutes after midnight.
    baseline : float, optional
        Background weight added to every period before normalization.

    Returns
    -------
    numpy.ndarray
        Length ``n_periods``, non-negative, summing to one.
    """
    n_periods = config.N_PERIODS if n_periods is None else int(n_periods)
    bin_size = config.BIN_SIZE if bin_size is None else int(bin_size)
    horizon_start = config.HORIZON_START if horizon_start is None else int(horizon_start)

    centres = horizon_start + np.arange(n_periods) * bin_size

    weights = np.full(n_periods, float(baseline), dtype=float)

    for mu, sigma, weight in peaks:
        weights += weight * _gaussian(centres, mu, sigma)

    weights = np.clip(weights, 0.0, None)

    total = weights.sum()

    if total <= 0:
        # Degenerate window (all peaks far outside the horizon): fall back to a
        # flat profile rather than returning an unusable all-zero curve.
        return np.full(n_periods, 1.0 / n_periods)

    return weights / total


def aircraft_time_profile(n_periods=None, bin_size=None, horizon_start=None):
    """
    Normalized expected incoming-aircraft intensity over the horizon.
    """
    return time_profile(
        n_periods=n_periods,
        bin_size=bin_size,
        horizon_start=horizon_start,
        peaks=AIRCRAFT_PEAKS,
        baseline=AIRCRAFT_BASELINE,
    )


# ============================================================
# Destination shares
# ============================================================

def gravity_shares(od_table, destinations=None):
    """
    Normalize a gravity table into destination shares.

    Parameters
    ----------
    od_table : pandas.DataFrame or dict
        Either a frame with ``destination`` and one of ``expected_trips`` /
        ``gravity_score`` / ``trips`` columns, or a plain
        ``{destination: weight}`` mapping.
    destinations : sequence of str, optional
        Restrict the shares to these destinations. Shares are renormalized
        over the restriction, so a subset instance still has total mass one.

    Returns
    -------
    dict
        ``{destination: share}`` summing to one.
    """
    if hasattr(od_table, "columns"):
        column = None

        for candidate in ("expected_trips", "gravity_score", "trips", "raw_gravity"):
            if candidate in od_table.columns:
                column = candidate
                break

        if column is None:
            raise ValueError(
                "od_table has none of the expected weight columns "
                "(expected_trips, gravity_score, trips, raw_gravity); "
                "found %s" % list(od_table.columns)
            )

        raw = {
            str(row["destination"]): float(row[column])
            for _, row in od_table.iterrows()
        }

    else:
        raw = {str(key): float(value) for key, value in od_table.items()}

    if destinations is not None:
        raw = {d: raw.get(d, 0.0) for d in destinations}
    else:
        raw = {d: v for d, v in raw.items() if v > 0}

    total = sum(raw.values())

    if total <= 0:
        raise ValueError("gravity table has no positive weight")

    return {d: v / total for d, v in raw.items()}


# ============================================================
# Baseline expected demand
# ============================================================

def baseline_demand_matrix(
    destination_share,
    profile,
    total_passengers=None,
    n_periods=None,
):
    """
    ``lambda_bar_td = total_passengers * share_d * profile_t``.

    Parameters
    ----------
    destination_share : dict
        ``{destination: share}`` from :func:`gravity_shares`.
    profile : numpy.ndarray
        Period weights from :func:`time_profile`.
    total_passengers : float, optional
        Expected passengers over the horizon. Defaults to
        ``config.TOTAL_PASSENGER_TRIPS``.
    n_periods : int, optional
        Overrides ``len(profile)``.

    Returns
    -------
    dict
        ``{(t, d): expected passengers}`` with 1-based periods.
    """
    total_passengers = (
        config.TOTAL_PASSENGER_TRIPS if total_passengers is None else float(total_passengers)
    )

    profile = np.asarray(profile, dtype=float)

    n_periods = len(profile) if n_periods is None else int(n_periods)

    return {
        (t, d): total_passengers * share * float(profile[t - 1])
        for d, share in destination_share.items()
        for t in range(1, n_periods + 1)
    }


# ============================================================
# Integerization of booked demand
# ============================================================

def largest_remainder(weights, target):
    """
    Integer vector of length ``len(weights)`` summing exactly to ``target``.

    Largest-remainder (Hamilton) apportionment: floor every weight, then hand
    the residual units to the largest fractional parts, so the total is matched
    exactly.
    """
    weights = np.asarray(weights, dtype=float)

    target = int(target)

    if weights.size == 0:
        return np.zeros(0, dtype=int)

    if target <= 0:
        return np.zeros(weights.size, dtype=int)

    positive = np.clip(weights, 0.0, None)

    total = positive.sum()

    if total <= 0:
        # No signal: spread the target over the periods with equal weight.
        distribution = np.full(weights.size, target / weights.size)
    else:
        distribution = positive / total * target

    counts = np.floor(distribution).astype(int)

    residual = target - int(counts.sum())

    if residual > 0:
        order = np.argsort(-(distribution - counts), kind="stable")
        counts[order[:residual]] += 1

    elif residual < 0:
        order = np.argsort(distribution - counts, kind="stable")
        for index in order[: -residual]:
            if counts[index] > 0:
                counts[index] -= 1

    return counts


def split_booked_demand(lambda_bar, booking_share=None, destinations=None):
    """
    Partition baseline demand into booked and on-demand components.

    ``B_td = phi_B lambda_bar_td`` is integerized so that both the total booked
    count and each destination's booked total match the intended values
    exactly; ``lambda_bar^O_td = (1 - phi_B) lambda_bar_td`` stays continuous
    because it is only a Poisson intensity.

    Parameters
    ----------
    lambda_bar : dict
        ``{(t, d): expected passengers}``.
    booking_share : float, optional
        ``phi_B``. Defaults to ``config.ADVANCE_BOOKING_SHARE``.
    destinations : sequence of str, optional
        Destination order. Defaults to a sorted order of the keys.

    Returns
    -------
    dict with keys
        ``booked``        : ``{(t, d): int}`` known before uncertainty
        ``ondemand_rate`` : ``{(t, d): float}`` expected on-demand intensity
        ``periods``       : sorted period list
        ``destinations``  : sorted destination list
        ``booked_total``  : realized integer booked total
        ``ondemand_total``: expected on-demand total
    """
    booking_share = (
        config.ADVANCE_BOOKING_SHARE if booking_share is None else float(booking_share)
    )

    if not 0.0 <= booking_share <= 1.0:
        raise ValueError("booking_share must lie in [0, 1], got %r" % (booking_share,))

    periods = sorted({t for (t, _) in lambda_bar})

    if destinations is None:
        destinations = sorted({d for (_, d) in lambda_bar})
    else:
        destinations = list(destinations)

    booked = {(t, d): 0 for t in periods for d in destinations}
    ondemand_rate = {(t, d): 0.0 for t in periods for d in destinations}

    # Destination totals, integerized with an exact grand total.
    destination_expectation = [
        booking_share * sum(lambda_bar.get((t, d), 0.0) for t in periods)
        for d in destinations
    ]

    grand_total = sum(destination_expectation)

    destination_targets = largest_remainder(
        destination_expectation, int(round(grand_total))
    )

    # Period split inside each destination, also exact.
    for d, destination_target in zip(destinations, destination_targets):

        weights = [
            booking_share * lambda_bar.get((t, d), 0.0)
            for t in periods
        ]

        counts = largest_remainder(weights, int(destination_target))

        for t, count in zip(periods, counts):
            booked[(t, d)] = int(count)

    for (t, d), value in lambda_bar.items():
        ondemand_rate[(t, d)] = (1.0 - booking_share) * float(value)

    return {
        "booked": booked,
        "ondemand_rate": ondemand_rate,
        "periods": periods,
        "destinations": destinations,
        "booked_total": int(sum(destination_targets)),
        "ondemand_total": float(sum(ondemand_rate.values())),
    }


# ============================================================
# Full baseline block
# ============================================================

def build_baseline_demand(
    od_table=None,
    profile=None,
    total_passengers=None,
    booking_share=None,
    destinations=None,
    n_periods=None,
    bin_size=None,
    horizon_start=None,
    vertiports=None,
):
    """
    Assemble every demand quantity that is known before the first stage.

    Builds:

    * a smooth two-peak temporal profile over the modeled window;
    * gravity destination shares scaled to the expected passenger total;
    * the ``phi_B`` / ``1 - phi_B`` split with integer booked counts;
    * the fare and distance tables.

    Parameters
    ----------
    od_table : pandas.DataFrame or dict, optional
        Gravity output. When omitted, every active destination receives an
        equal share, which is enough to exercise the model without the ACS
        pipeline installed.
    profile : numpy.ndarray, optional
        Period weights. Defaults to :func:`time_profile`.
    total_passengers, booking_share, destinations, n_periods, bin_size,
    horizon_start : optional
        Overrides for the corresponding :mod:`config` values.
    vertiports : dict, optional
        ``{name: (lat, lon)}``. Defaults to ``config.VERTIPORTS_RAW``.

    Returns
    -------
    dict
        A complete baseline block (see :func:`split_booked_demand` for the
        demand keys, plus ``lambda_bar``, ``fares``, ``distances``,
        ``profile`` and the resolved settings).
    """
    n_periods = None if n_periods is None else int(n_periods)
    bin_size = config.BIN_SIZE if bin_size is None else int(bin_size)
    horizon_start = config.HORIZON_START if horizon_start is None else int(horizon_start)

    total_passengers = (
        config.TOTAL_PASSENGER_TRIPS if total_passengers is None else float(total_passengers)
    )

    booking_share = (
        config.ADVANCE_BOOKING_SHARE if booking_share is None else float(booking_share)
    )

    if destinations is None:
        destinations = list(config.ACTIVE_DESTINATIONS)
    else:
        destinations = list(destinations)

    if profile is None:
        n_periods = config.N_PERIODS if n_periods is None else n_periods
        profile = time_profile(n_periods, bin_size, horizon_start)
    else:
        profile = np.asarray(profile, dtype=float)

        if n_periods is None:
            n_periods = len(profile)
        elif len(profile) != n_periods:
            # Truncation keeps the temporal *shape* and, crucially, the demand
            # *rate*: weights are not renormalized, so the horizon total scales
            # down with the window. A longer request is zero-padded and leaves
            # the total unchanged.
            if len(profile) > n_periods:
                profile = profile[:n_periods]
            else:
                profile = np.concatenate(
                    [profile, np.zeros(n_periods - len(profile))]
                )

    n_periods = len(profile)




    if od_table is None:
        od_table = {d: 1.0 for d in destinations}

    shares = gravity_shares(od_table, destinations=destinations)

    lambda_bar = baseline_demand_matrix(
        shares, profile, total_passengers=total_passengers, n_periods=n_periods
    )

    split = split_booked_demand(
        lambda_bar, booking_share=booking_share, destinations=destinations
    )

    distances = destination_distances(vertiports)

    fares = destination_fares(vertiports)

    return {
        "destinations": destinations,
        "periods": list(range(1, n_periods + 1)),
        "n_periods": n_periods,
        "bin_size": bin_size,
        "horizon_start": horizon_start,
        "lambda_bar": lambda_bar,
        "booked": split["booked"],
        "ondemand_rate": split["ondemand_rate"],
        "booked_total": split["booked_total"],
        "ondemand_total": split["ondemand_total"],
        "total_passengers": total_passengers,
        "booking_share": booking_share,
        "shares": shares,
        "profile": profile,
        "distances": distances,
        "fares": fares,
    }
