"""Central configuration for the hybrid advance-booked + on-demand UAM model."""

from pathlib import Path
import os

# Planning horizon (small development case; paper-scale case is 08:00--22:00)
HORIZON_START = 480          # 08:00
HORIZON_END = 720            # 12:00
BIN_SIZE = 15
HORIZON_MINUTES = HORIZON_END - HORIZON_START
N_PERIODS = HORIZON_MINUTES // BIN_SIZE

# eVTOL
BATTERY_CAPACITY = 110       # kWh
CHARGING_RATE = 120          # kW
SOC_REQUIRED = 70            # %
EVTOL_CAPACITY = 4

# Charging infrastructure
M_FACILITIES = 3

# Service quality
LOS_MINUTES = 30
LOS_PERIODS = LOS_MINUTES // BIN_SIZE

# Initial ready aircraft: incremental daily opportunity/pre-positioning cost.
# Synthetic development values only; vary cost and cap in the paper.
# Aircraft arrive ready before period 1; no pre-horizon charging is modeled.
INITIAL_FLEET_MAX = 12
INITIAL_FLEET_COST = 60.0

# Demand/fleet
TOTAL_PASSENGER_TRIPS = 20
AIRCRAFT_PASSENGER_RATIO = 0.5

# Hybrid-service split. This is an experimental factor, not an empirically
# calibrated DFW value. Advance bookings are known before Stage 1; the
# remaining demand arrives on demand in Stage 2.
ADVANCE_BOOKING_FRACTION = 0.40

# Network
HUB_NAME = "DFW Airport"
VERTIPORTS_RAW = {
    "DFW Airport": (32.8998, -97.0409),
    "Dallas CBD": (32.7767, -96.7970),
    "Dallas Love Field Airport": (32.84815, -96.85135),
    "Arlington Municipal Airport": (32.6639, -97.0943),
    "Fort Worth City Center": (32.75550, -97.33080),
    "Mesquite Metro Airport": (32.7470, -96.5304),
    "Dallas Executive Airport (RBD)": (32.6810, -96.8690),
    "Denton Municipal Airport": (33.20083, -97.19778),
    "Fort Worth Meacham International Airport": (32.81969, -97.36319),
    "McKinney National Airport": (33.1753, -96.5825),
}
ACTIVE_DESTINATIONS = [
    "Dallas CBD",
    "Dallas Love Field Airport",
    "Arlington Municipal Airport",
]

# Demand model
GAMMA = 1.1
UAM_ADOPTION = 0.30
CBD_BOOST_INIT = 2.0

# Electricity pricing
BASE_PRICE = 0.1482
M_PEAK = 1.5
M_SHOULDER = 1.05
M_OFF = 0.6

# Stage-1 costs
CHARGING_RESERVATION_COST = 1.0
COMMITTED_DEPARTURE_COST = 1.0

# Stage-2 costs
ADDED_DEPARTURE_COST = 2.0
CANCELLED_DEPARTURE_COST = 2.0
EMERGENCY_CAPACITY_COST = 2.0
ONDEMAND_UNSERVED_PASSENGER_COST = 100.0
BOOKED_UNSERVED_PASSENGER_COST = 150.0
FLIGHT_OPERATING_COST = 0.0

# Capacity
MAX_DEPARTURES_PER_PERIOD = 17

# CVaR. Direct passenger failures replace the old passenger-equivalent
# cancelled-flight proxy. Booked failures can be weighted more heavily.
CVaR_ALPHA = 0.90
CVaR_RISK_WEIGHT = 22
CVAR_BOOKED_WEIGHT = 2.0
CVAR_ONDEMAND_WEIGHT = 1.0

# Reproducible synthetic profiles for the scalability study.  The profiles
# increase infrastructure capacity with demand intensity (passengers/hour),
# while aircraft arrivals already scale with TOTAL demand in scenario_generator.
# Nine geographic destinations, seat capacity, LoS, booking mix, and per-unit
# economic coefficients are deliberately unchanged.
SCALING_REFERENCE_PASSENGERS = 200.0
SCALING_REFERENCE_HOURS = 4.0
SCALING_CASE_HOURS = {200: 4, 500: 6, 1000: 8, 2000: 12}


def scalability_settings(passengers, *, horizon_hours=None, mode="proportional",
                         capacity_factor=1.0):
    """Return case-specific synthetic capacity, without mutating global defaults.

    `capacity_factor` adjusts baseline capacity (not the demand or horizon).
    Use a separate fixed-capacity experiment for operational stress testing.
    The scaled departure cap is a synthetic infrastructure expansion, NOT an
    empirical maximum for a single DFW vertiport.
    """
    import math

    passengers = float(passengers)
    if not math.isfinite(passengers) or passengers <= 0:
        raise ValueError("passengers must be positive and finite")
    if not math.isfinite(capacity_factor) or capacity_factor <= 0:
        raise ValueError("capacity_factor must be positive and finite")
    if mode not in ("fixed", "proportional"):
        raise ValueError("mode must be fixed or proportional")
    if horizon_hours is None:
        if mode == "fixed":
            horizon_hours = HORIZON_MINUTES / 60.0
        else:
            key = round(passengers)
            if not math.isclose(key, passengers) or key not in SCALING_CASE_HOURS:
                raise ValueError(
                    f"No predefined horizon for {passengers:g} passengers; "
                    "provide --horizons explicitly")
            horizon_hours = SCALING_CASE_HOURS[key]
    horizon_minutes = round(float(horizon_hours) * 60)
    if (horizon_minutes <= 0 or horizon_minutes % BIN_SIZE
            or not math.isclose(horizon_minutes, float(horizon_hours) * 60)):
        raise ValueError("horizon must contain a positive whole number of 15-minute bins")
    if mode == "fixed":
        multiplier = capacity_factor
    else:
        multiplier = (passengers / (horizon_minutes / 60.0)) / (
            SCALING_REFERENCE_PASSENGERS / SCALING_REFERENCE_HOURS
        ) * capacity_factor

    return {
        "mode": mode,
        "horizon_minutes": int(horizon_minutes),
        "n_periods": int(horizon_minutes // BIN_SIZE),
        "passengers_per_hour": passengers / (horizon_minutes / 60.0),
        "capacity_multiplier": multiplier,
        "initial_fleet_max": max(1, math.ceil(INITIAL_FLEET_MAX * multiplier - 1e-10)),
        "charging_facilities": max(1, math.ceil(M_FACILITIES * multiplier - 1e-10)),
        "max_departures": max(1, math.ceil(MAX_DEPARTURES_PER_PERIOD * multiplier - 1e-10)),
    }


# Scenario generation/reduction (small development case)
RAW_SCENARIOS = 30
TARGET_SCENARIOS = 3
OPTIMIZATION_SCENARIOS = TARGET_SCENARIOS
OUT_OF_SAMPLE_SCENARIOS = 500
RANDOM_SEED = 60

# Aircraft arrival-SoC uncertainty.  The conditional mean is
#   SOC_BASE_MEAN - SOC_STRESS_BETA*h_ts - SOC_COMMON_FACTOR_BETA*g_ts,
# then sampled from a truncated normal on [ARRIVAL_SOC_MIN, ARRIVAL_SOC_MAX].
# These are experimental dependence parameters and should be varied in the
# sensitivity analysis rather than interpreted as calibrated DFW estimates.
ARRIVAL_SOC_MIN = 20.0
ARRIVAL_SOC_MAX = 60.0
SOC_BASE_MEAN = 50.0
SOC_STRESS_BETA = 8.0
SOC_COMMON_FACTOR_BETA = 3.0
SOC_STD = 6.0

# Tail-stress proxy weights used for scenario stratification.
STRESS_WEIGHT_PASSENGER = 1.0
STRESS_WEIGHT_AIRCRAFT = 1.0
STRESS_WEIGHT_CHARGING = 0.05

# Census / ACS
CENSUS_YEAR = "2022"
BBOX = {"min_lon": -97.60, "min_lat": 32.30, "max_lon": -96.20, "max_lat": 33.30}

# Portable paths for the current repository layout:
#   coord_schedule_UAM/
#       src/config.py  <-- this file
#       data/acs_texas_2022_raw.json
#       data/tl_2022_48_tract/tl_2022_48_tract.shp
#       outputs/
# These do not depend on the PowerShell working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = PROJECT_ROOT  # compatibility: retired code/ folder
DATA_DIR = Path(os.environ.get("UAM_DATA_DIR", str(PROJECT_ROOT / "data")))
RAW_DATA_DIR = DATA_DIR
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"
DIAGNOSTICS_DIR = OUTPUT_DIR / "diagnostics"
ACS_JSON_PATH = Path(os.environ.get(
    "UAM_ACS_JSON_PATH", str(DATA_DIR / "acs_texas_2022_raw.json")
))
SHAPEFILE_PATH = Path(os.environ.get(
    "UAM_SHAPEFILE_PATH",
    str(DATA_DIR / "tl_2022_48_tract" / "tl_2022_48_tract.shp")
))
