"""
Central configuration for the two-stage risk-averse eVTOL scheduling model.

Holds every model parameter: horizon, aircraft and chargers, demand and
booking share, fleet allocation, cost and penalty profiles, CVaR settings,
uncertainty, scenario counts, solver budget, vertiport network and paths.
Units and formulation symbols are noted beside each constant.

``build_parameters(**overrides)`` collects them into the dictionary passed
through the pipeline; overrides merge one level deep, so a sensitivity run
changes only the key it varies. ``PAPER_BASE`` (end of file) restores the
published parameter set::

    parameters = config.build_parameters(**config.PAPER_BASE)
"""

from pathlib import Path


# ============================================================
# 1. Planning horizon
# ============================================================
# Test case: 09:00-10:00, i.e. four 15-minute periods.
# ``PAPER_BASE`` (end of file) restores the published 14-hour horizon.

HORIZON_START = 9 * 60          # minutes after midnight -> 09:00

HORIZON_END = 10 * 60           # minutes after midnight -> 10:00

#: Period length in minutes. Sensitivity: 5, 10, 15.
BIN_SIZE = 15

HORIZON_MINUTES = HORIZON_END - HORIZON_START

#: Number of modeled periods ``L``; period ``t`` spans
#: ``[HORIZON_START + (t-1)*BIN_SIZE, HORIZON_START + t*BIN_SIZE)``.
N_PERIODS = HORIZON_MINUTES // BIN_SIZE

#: Alias kept for external scripts that refer to the period count as ``L``.
L_PERIODS = N_PERIODS


def period_start_minutes(t, bin_size=None, horizon_start=None):
    """Clock time (minutes after midnight) at which modeled period ``t`` starts."""
    bin_size = BIN_SIZE if bin_size is None else bin_size
    horizon_start = HORIZON_START if horizon_start is None else horizon_start

    return horizon_start + (int(t) - 1) * bin_size


def period_start_hour(t, bin_size=None, horizon_start=None):
    """Clock hour (fractional) at which modeled period ``t`` starts."""
    return period_start_minutes(t, bin_size, horizon_start) / 60.0


# ============================================================
# 2. eVTOL physical parameters
# ============================================================

BATTERY_CAPACITY = 110.0       # kWh,  E_bar
CHARGING_RATE = 120.0          # kW,   mu
SOC_REQUIRED = 70.0            # %,    S_req (required departure SoC)
EVTOL_CAPACITY = 4             # seats, n_max


def energy_to_charge_kwh(soc, params=None):
    """
    Energy in kWh needed to bring an aircraft from arrival SoC ``soc`` to
    ``S_req``::

        E(S_req - SoC)^+ = battery_capacity * max(0, S_req - SoC) / 100

    Also the per-aircraft charging cost basis.
    """
    battery = BATTERY_CAPACITY if params is None else params["battery_capacity"]
    soc_req = SOC_REQUIRED if params is None else params["soc_min"]

    return battery * max(0.0, soc_req - float(soc)) / 100.0


# ============================================================
# 3. Charging infrastructure
# ============================================================
# Identical charger units, activatable up to the physical limit; units beyond
# the reservation ``b_t`` are activated at the emergency cost.

M_FACILITIES = 35

#: Maximum departures per period (hub takeoff capacity); bounds both the
#: advance plan and realized departures.
MAX_DEPARTURES_PER_PERIOD = 17

# ============================================================
# 4. Service quality
# ============================================================
# Hard maximum wait from passenger eligibility to departure, in minutes.

LOS_MINUTES = 30


def los_periods(bin_size=None):
    """
    Number of periods in the maximum passenger wait ``LoS``.

    Rounds *up*, so a period length that does not divide ``LOS_MINUTES`` never
    shortens the realized wait below the intended limit.
    """
    bin_size = BIN_SIZE if bin_size is None else bin_size

    return max(1, -(-int(LOS_MINUTES) // int(bin_size)))


LOS_PERIODS = los_periods()


# ============================================================
# 5. Baseline demand and booking share
# ============================================================
# Expected passengers over the horizon, split into advance bookings (share
# ``phi_B``) and stochastic on-demand requests. Test case uses 20.

TOTAL_PASSENGER_TRIPS = 20         # expected passengers over the horizon

ADVANCE_BOOKING_SHARE = 0.40        # phi_B

#: Booking-share sensitivity grid.
ADVANCE_BOOKING_SHARE_RANGE = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70)


# ============================================================
# 6. Fleet allocation
# ============================================================
# Ready aircraft at the start of the horizon (``x0``), capped by ``N_fleet``,
# and the mean number of stochastic incoming aircraft per passenger.

AIRCRAFT_PASSENGER_RATIO = 0.30     # expected incoming aircraft per passenger

#: Mean stochastic incoming aircraft over the whole horizon:
#: 0.30 * 20 = 6 in the test case.
EXPECTED_INCOMING_AIRCRAFT = AIRCRAFT_PASSENGER_RATIO * TOTAL_PASSENGER_TRIPS

# EXPERIMENTAL: no calibrated value; the published base case uses 40.
# 5 keeps the test case supply-scarce so the CVaR terms actually bind.
MAX_INITIAL_FLEET = 5

#: Fleet-cap sensitivity grid.
MAX_INITIAL_FLEET_RANGE = (10, 20, 40, 60)

# EXPERIMENTAL: incremental pre-positioning cost per aircraft, incurred whether
# or not the aircraft is deployed in a realized scenario.
INITIAL_AIRCRAFT_COST = 40.0        # $ / aircraft, c_fleet

INITIAL_AIRCRAFT_COST_RANGE = (0.0, 40.0, 120.0)


# ============================================================
# 7. Charging reservation, commitment and adjustment costs
# ============================================================
# Per unit-period: reservation ``c_res_t`` and emergency capacity ``c_emg_t``;
# per departure: commitment ``c_com_dt``, add ``c_add_dt``, cancel ``c_can_dt``.

#: Uncalibrated; the profile name is an experimental factor.
COST_PROFILE = "medium"

#: Multiplier applied to the earlier development cost ratios, so the medium
#: profile below sits at 20/20/40/40/40 dollars.
COST_SCALE = 20.0

#: low 10/10/20/20/20, medium 20/20/40/40/40, high 50/50/100/100/100 ($).
COST_PROFILES = {
    "low": {
        "charging_reservation": 0.5 * COST_SCALE,     # 10.0
        "commitment": 0.5 * COST_SCALE,               # 10.0
        "added_departure": 1.0 * COST_SCALE,          # 20.0
        "cancelled_departure": 1.0 * COST_SCALE,      # 20.0
        "emergency_capacity": 1.0 * COST_SCALE,       # 20.0
    },
    "medium": {
        "charging_reservation": 1.0 * COST_SCALE,     # 20.0
        "commitment": 1.0 * COST_SCALE,               # 20.0
        "added_departure": 2.0 * COST_SCALE,          # 40.0
        "cancelled_departure": 2.0 * COST_SCALE,      # 40.0
        "emergency_capacity": 2.0 * COST_SCALE,       # 40.0
    },
    "high": {
        "charging_reservation": 2.5 * COST_SCALE,     # 50.0
        "commitment": 2.5 * COST_SCALE,               # 50.0
        "added_departure": 5.0 * COST_SCALE,          # 100.0
        "cancelled_departure": 5.0 * COST_SCALE,      # 100.0
        "emergency_capacity": 5.0 * COST_SCALE,       # 100.0
    },
}

#: Convenience aliases for the selected profile (the medium base case). In every
#: profile a reservation costs half an emergency unit-period, so the model
#: reserves only where chargers are genuinely needed.
CHARGING_RESERVATION_COST = COST_PROFILES[COST_PROFILE]["charging_reservation"]
COMMITTED_DEPARTURE_COST = COST_PROFILES[COST_PROFILE]["commitment"]
ADDED_DEPARTURE_COST = COST_PROFILES[COST_PROFILE]["added_departure"]
CANCELLED_DEPARTURE_COST = COST_PROFILES[COST_PROFILE]["cancelled_departure"]
EMERGENCY_CAPACITY_COST = COST_PROFILES[COST_PROFILE]["emergency_capacity"]


# ============================================================
# 8. Service-failure penalties
# ============================================================
# Per unserved booked (``c^un,B``) and on-demand (``c^un,O``) passenger,
# homogeneous across destinations in the base case.

UNSERVED_BOOKED_COST = 150.0        # c^un,B
UNSERVED_ONDEMAND_COST = 100.0      # c^un,O

UNSERVED_COST_RANGE = (100.0, 150.0, 250.0)


# ============================================================
# 9. Risk (CVaR) parameters
# ============================================================
# ``alpha`` is the CVaR level, ``theta`` the risk weight ($ per weighted unserved
# passenger); ``omega_B`` and ``omega_O`` weight booked vs on-demand losses.

CVAR_ALPHA = 0.90
CVAR_RISK_WEIGHT = 22.0             # theta, $ per weighted unserved passenger

CVAR_ALPHA_RANGE = (0.80, 0.90, 0.95)
CVAR_RISK_WEIGHT_RANGE = (0.0, 11.0, 22.0, 44.0)

CVAR_WEIGHT_BOOKED = 2.0            # omega_B
CVAR_WEIGHT_ONDEMAND = 1.0          # omega_O


# ============================================================
# 10. Arrival state-of-charge uncertainty
# ============================================================
# Truncated-normal arrival SoC on [20, 60] %, conditioned on the stress level
# and the common factor (coefficients in percentage points).

ARRIVAL_SOC_MIN = 20.0              # %, S_min
ARRIVAL_SOC_MAX = 60.0              # %, S_max

SOC_MEAN_INTERCEPT = 50.0           # S0       (percentage points)
SOC_STRESS_COEFFICIENT = 8.0        # beta_Gamma (percentage points per unit stress)
SOC_FACTOR_COEFFICIENT = 3.0        # beta_g     (percentage points per unit factor)
SOC_STD_DEVIATION = 6.0             # sigma_s    (percentage points)

#: No-coupling baseline: both conditional coefficients set to zero.
SOC_STRESS_COEFFICIENT_NOCOUPLING = 0.0
SOC_FACTOR_COEFFICIENT_NOCOUPLING = 0.0


# ============================================================
# 11. Common-factor correlation
# ============================================================
# Shared zero-mean temporal factor ``g_ts`` (std ``sigma_g``) entering the
# on-demand intensity (``beta_p``) and arrivals (``beta_e``) through a log link.

FACTOR_STD = 1.0                    # sigma_g
FACTOR_SMOOTHING_SIGMA = 1.0        # periods; temporal smoothing of g_ts

DEMAND_FACTOR_BETA = 0.35           # beta_p
ARRIVAL_FACTOR_BETA = 0.35          # beta_e


# ============================================================
# 12. Scenario generation and reduction
# ============================================================
# Test case: 30 raw scenarios reduced to 4; ``PAPER_BASE`` restores 100 -> 10.
# The tail stratum still gets max(2, round(0.30 * 4)) = 2 representatives.

RAW_SCENARIOS = 30
OPTIMIZATION_SCENARIOS = 4
TARGET_SCENARIOS = OPTIMIZATION_SCENARIOS     # alias used by older call sites
OUT_OF_SAMPLE_SCENARIOS = 500

RAW_SCENARIO_RANGE = (30, 100, 250)
OPTIMIZATION_SCENARIO_RANGE = (4, 8, 15)
OUT_OF_SAMPLE_SCENARIO_RANGE = (500, 1000)

RANDOM_SEED = 60

# Proxy score weights for scenario reduction.
REDUCTION_KAPPA_P = 1.0
REDUCTION_KAPPA_E = 1.0
REDUCTION_KAPPA_C = 0.05

#: At least two and roughly 30% of the retained scenarios go to the tail
#: stratum whenever the requested count permits.
TAIL_STRATUM_FRACTION = 0.30
TAIL_STRATUM_MIN = 2


# ============================================================
# 13. Solver budget
# ============================================================
# Stop at a 5% relative MIP gap or after one hour.

SOLVER_GAP_REL = 0.05
SOLVER_TIME_LIMIT = 3600


# ============================================================
# 14. Vertiport network
# ============================================================
# Test case: the DFW hub plus three nearby destinations; ``PAPER_VERTIPORTS``
# holds the six omitted ones.

HUB_NAME = "DFW Airport"

VERTIPORTS_RAW = {

    "DFW Airport":
        (32.8998, -97.0409),

    "Dallas CBD":
        (32.7767, -96.7970),

    "Dallas Love Field Airport":
        (32.84815, -96.85135),

    "Arlington Municipal Airport":
        (32.6639, -97.0943),
}

#: The six destinations omitted from the test case; merge into
#: ``VERTIPORTS_RAW`` to restore the full network.
PAPER_VERTIPORTS = {

    "Fort Worth City Center":
        (32.75550, -97.33080),

    "Mesquite Metro Airport":
        (32.7470, -96.5304),

    "Dallas Executive Airport (RBD)":
        (32.6810, -96.8690),

    "Denton Municipal Airport":
        (33.20083, -97.19778),

    "Fort Worth Meacham International Airport":
        (32.81969, -97.36319),

    "McKinney National Airport":
        (33.1753, -96.5825),
}

#: Every vertiport except the hub.
ACTIVE_DESTINATIONS = [
    name for name in VERTIPORTS_RAW if name != HUB_NAME
]

#: Gravity model parameters.
GAMMA = 1.1                 # distance-decay parameter
UAM_ADOPTION = 0.30         # adoption fraction
CBD_BOOST_INIT = 2.0        # destination-attractiveness multiplier a_j


# ============================================================
# 15. Passenger fare and electricity tariff
# ============================================================
# Fare rho_d = 20 + 3 * Dist_d ($, great-circle miles); energy price is
# P(t) = P_base * multiplier over the peak / off-peak / shoulder windows below.

FARE_INTERCEPT = 20.0       # $
FARE_PER_MILE = 3.0         # $ / mile

# P(t) = P_base * m_peak for 16 <= h < 21, m_off for 0 <= h < 6 or 21 <= h < 24,
# m_shoulder otherwise; ``h`` is the clock hour of the charging start period.

BASE_PRICE = 0.1482         # P_base, $/kWh

M_PEAK = 1.5                # m_peak
M_SHOULDER = 1.05           # m_shoulder
M_OFF = 0.6                 # m_off

#: The test window (09:00-10:00) lies entirely in the shoulder block, so the
#: time-of-use structure is inactive there.
PEAK_HOUR_WINDOWS = ((16, 21),)
OFF_HOUR_WINDOWS = ((0, 6), (21, 24))


# ============================================================
# 16. Flight operating cost c^flt_d
# ============================================================
# Destination-specific operating cost, a flat base plus a per-mile term; not
# calibrated, so it is outside the published parameter set.

FLIGHT_COST_BASE = 20
FLIGHT_COST_PER_MILE = 2


# ============================================================
# 17. Census / ACS data
# ============================================================
# Census year and the Texas bounding box used to select tracts.

CENSUS_YEAR = "2022"

BBOX = {

    "min_lon": -97.60,

    "min_lat": 32.30,

    "max_lon": -96.20,

    "max_lat": 33.30,
}


# ============================================================
# 18. Assembled parameter block
# ============================================================
# One dictionary carries every parameter into scenario generation, reduction,
# model construction and reporting; a sensitivity run overrides only its key.


def build_parameters(**overrides):
    """
    Assemble the full parameter block, with optional one-level-deep overrides.

    ``periods`` and ``los_periods`` are recomputed whenever ``horizon`` or
    ``bin_size`` is overridden, so derived grid values cannot disagree.

    Examples
    --------
    >>> p = build_parameters(booking_share=0.6, cvar={"weight": 0.0})
    >>> p["booking_share"], p["cvar"]["weight"], p["cvar"]["alpha"]
    (0.6, 0.0, 0.9)
    """
    parameters = {
        # ---- time grid -------------------------------------------------
        "periods": N_PERIODS,
        "bin_size": BIN_SIZE,
        "horizon": {
            "start": HORIZON_START,
            "end": HORIZON_END,
            "bin_size": BIN_SIZE,
        },
        "los_periods": LOS_PERIODS,

        # ---- aircraft and hub ------------------------------------------
        "battery_capacity": BATTERY_CAPACITY,
        "charging_rate": CHARGING_RATE,
        "soc_min": SOC_REQUIRED,
        "evtol_capacity": EVTOL_CAPACITY,
        "charging_facilities": M_FACILITIES,
        "max_departures": MAX_DEPARTURES_PER_PERIOD,
        "max_initial_fleet": MAX_INITIAL_FLEET,
        "initial_aircraft_cost": INITIAL_AIRCRAFT_COST,

        # ---- tariff and fares ------------------------------------------
        "base_price": BASE_PRICE,
        "peak_multiplier": M_PEAK,
        "shoulder_multiplier": M_SHOULDER,
        "off_multiplier": M_OFF,
        "peak_hour_windows": PEAK_HOUR_WINDOWS,
        "off_hour_windows": OFF_HOUR_WINDOWS,
        "fare_intercept": FARE_INTERCEPT,
        "fare_per_mile": FARE_PER_MILE,
        "flight_cost_base": FLIGHT_COST_BASE,
        "flight_cost_per_mile": FLIGHT_COST_PER_MILE,

        # ---- recourse and first-stage costs ----------------------------
        "costs": dict(COST_PROFILES[COST_PROFILE]),
        "unserved_booked": UNSERVED_BOOKED_COST,
        "unserved_ondemand": UNSERVED_ONDEMAND_COST,

        # ---- risk -------------------------------------------------------
        "cvar": {
            "alpha": CVAR_ALPHA,
            "weight": CVAR_RISK_WEIGHT,
            "weight_booked": CVAR_WEIGHT_BOOKED,
            "weight_ondemand": CVAR_WEIGHT_ONDEMAND,
        },

        # ---- demand and uncertainty ------------------------------------
        "total_passengers": TOTAL_PASSENGER_TRIPS,
        "booking_share": ADVANCE_BOOKING_SHARE,
        "aircraft_passenger_ratio": AIRCRAFT_PASSENGER_RATIO,
        "expected_incoming_aircraft": EXPECTED_INCOMING_AIRCRAFT,

        "soc": {
            "intercept": SOC_MEAN_INTERCEPT,
            "stress_coefficient": SOC_STRESS_COEFFICIENT,
            "factor_coefficient": SOC_FACTOR_COEFFICIENT,
            "std": SOC_STD_DEVIATION,
            "min": ARRIVAL_SOC_MIN,
            "max": ARRIVAL_SOC_MAX,
        },

        "correlation": {
            "factor_std": FACTOR_STD,
            "demand_beta": DEMAND_FACTOR_BETA,
            "arrival_beta": ARRIVAL_FACTOR_BETA,
            "smoothing_sigma": FACTOR_SMOOTHING_SIGMA,
        },

        # ---- scenario sampling and reduction ---------------------------
        "raw_scenarios": RAW_SCENARIOS,
        "optimization_scenarios": OPTIMIZATION_SCENARIOS,
        "out_of_sample_scenarios": OUT_OF_SAMPLE_SCENARIOS,
        "seed": RANDOM_SEED,

        "reduction": {
            "kappa_p": REDUCTION_KAPPA_P,
            "kappa_e": REDUCTION_KAPPA_E,
            "kappa_c": REDUCTION_KAPPA_C,
            "tail_fraction": TAIL_STRATUM_FRACTION,
            "tail_min": TAIL_STRATUM_MIN,
        },

        # ---- solver budget ---------------------------------------------
        "solver": {
            "gap_rel": SOLVER_GAP_REL,
            "time_limit": SOLVER_TIME_LIMIT,
        },
    }

    for key, value in overrides.items():

        if isinstance(value, dict) and isinstance(parameters.get(key), dict):
            parameters[key].update(value)
        else:
            parameters[key] = value

    # ---- keep the time grid internally consistent -----------------------
    # ``periods`` and ``los_periods`` are derived; recompute them when the
    # horizon or the period length is overridden.
    horizon_override = "horizon" in overrides
    bin_size_override = "bin_size" in overrides

    if horizon_override:

        horizon = parameters["horizon"]

        parameters["bin_size"] = int(horizon["bin_size"])

        parameters["periods"] = (
            int(horizon["end"]) - int(horizon["start"])
        ) // int(horizon["bin_size"])

        parameters["los_periods"] = los_periods(horizon["bin_size"])

    elif bin_size_override:

        parameters["horizon"]["bin_size"] = int(parameters["bin_size"])

        parameters["los_periods"] = los_periods(parameters["bin_size"])

    return parameters


# ============================================================
# 19. Paper base case (restores the published parameter set)
# ============================================================
# ``config.build_parameters(**config.PAPER_BASE)`` runs the published case; the
# destination set is module state, so merge ``PAPER_VERTIPORTS`` separately.

PAPER_BASE = {
    "horizon": {"start": 6 * 60, "end": 20 * 60, "bin_size": 15},   # 14 h
    "total_passengers": 2000,
    "max_initial_fleet": 40,
    "raw_scenarios": 100,
    "optimization_scenarios": 10,
    "out_of_sample_scenarios": 500,
}


# ============================================================
# 20. Paths
# ============================================================
# Project layout, derived from this file's location (``src/config.py``).

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "input"

RAW_DATA_DIR = DATA_DIR / "raw"

OUTPUT_DIR = PROJECT_ROOT / "output"

FIGURES_DIR = OUTPUT_DIR / "figures"

RESULTS_DIR = OUTPUT_DIR / "results"

ACS_JSON_PATH = (
    RAW_DATA_DIR /
    "acs_texas_2022_raw.json"
)

SHAPEFILE_PATH = (
    RAW_DATA_DIR /
    "tl_2022_48_tract" /
    "tl_2022_48_tract.shp"
)
