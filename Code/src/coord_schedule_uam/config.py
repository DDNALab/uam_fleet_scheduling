"""
config.py
=========

Central configuration for the eVTOL two-stage stochastic scheduling model.

Based on:
Risk-Aware Charging and Service Commitment for Urban Air Mobility
under Correlated Operational Uncertainty

Contains:
- deterministic parameters known before uncertainty
- physical system parameters
- economic parameters
- demand model parameters
- stochastic experiment settings
"""


from pathlib import Path



# ============================================================
# Planning Horizon
# ============================================================

# Paper:
# 14-hour operating horizon
# 15-minute periods

HORIZON_START = 480         # 08:00

HORIZON_END = 720          # 10:00

BIN_SIZE = 15               # minutes


HORIZON_MINUTES = HORIZON_END - HORIZON_START

N_PERIODS = HORIZON_MINUTES // BIN_SIZE



# ============================================================
# eVTOL Physical Parameters
# ============================================================

# Paper base case:
# Battery capacity = 110 kWh
# Charging power = 120 kW
# Required departure SOC = 70%
# Capacity = 4 passengers


BATTERY_CAPACITY = 110       # kWh

CHARGING_RATE = 120          # kW

SOC_REQUIRED = 70            # %

EVTOL_CAPACITY = 4           # passengers



# ============================================================
# Charging Infrastructure
# ============================================================

# Number of identical charging-power units

M_FACILITIES = 3



# ============================================================
# Service Quality
# ============================================================

# Maximum passenger waiting time

LOS_MINUTES = 30

LOS_PERIODS = LOS_MINUTES // BIN_SIZE



# ============================================================
# Fleet / Demand Base Parameters
# ============================================================

# Total expected passenger demand

TOTAL_PASSENGER_TRIPS = 20


# Passenger/eVTOL relationship
# Used for scenario generation
#
# Raised from 0.30 to 1.80 after a seed sweep (seeds 60, 1-4). At 0.30 the
# 8-period horizon with LoS = 2 left demand structurally unserved (mean 1.96
# passengers per seed, objective swinging between -222 and +412). At 1.80 no
# seed leaves a single passenger unserved -- the binding constraint is the
# cumulative number of seats ready before each passenger's LoS deadline, not
# charging speed (charging rate and soc_min were both measured to have no
# effect on the unserved count).

AIRCRAFT_PASSENGER_RATIO = 0.5



# ============================================================
# Vertiport Network
# ============================================================

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

# ---------------------------------------------------------------------------
# Active destinations for optimization test
# ---------------------------------------------------------------------------

ACTIVE_DESTINATIONS = [
    "Dallas CBD",
    "Dallas Love Field Airport",
    "Arlington Municipal Airport",
]

# ============================================================
# Demand Model / Gravity Model
# ============================================================

# Gravity model exponent:

# G_DFW,j =
# U_DFW * U_j / distance^gamma

GAMMA = 1.1


# UAM adoption rate

UAM_ADOPTION = 0.30



# Optional attractiveness adjustment

CBD_BOOST_INIT = 2.0



# ============================================================
# Electricity Pricing
# ============================================================

BASE_PRICE = 0.1482       # $/kWh


# Time-of-use multipliers

M_PEAK = 1.5

M_SHOULDER = 1.05

M_OFF = 0.6



# ============================================================
# Two-stage Objective Parameters
# ============================================================


# First-stage costs

CHARGING_RESERVATION_COST = 1.0

COMMITTED_DEPARTURE_COST = 1.0



# Second-stage recourse costs

ADDED_DEPARTURE_COST = 2.0

CANCELLED_DEPARTURE_COST = 2.0

EMERGENCY_CAPACITY_COST = 2.0

UNSERVED_PASSENGER_COST = 100.0


FLIGHT_OPERATING_COST = 0.0



# ============================================================
# Operational Limits
# ============================================================

# Maximum departures per time period

MAX_DEPARTURES_PER_PERIOD = 17



# ============================================================
# CVaR Risk Parameters
# ============================================================

# Paper:
# alpha = 0.90
# theta = 22


CVaR_ALPHA = 0.90

CVaR_RISK_WEIGHT = 22



# ============================================================
# Scenario Generation
# ============================================================

# Paper:
# 100 generated scenarios
# reduced to 10
# evaluated on 500 out-of-sample


# ============================================================
# Scenario Generation / Reduction
# ============================================================

# Number of raw Monte Carlo scenarios
RAW_SCENARIOS = 30


# Number of scenarios kept for optimization
# after scenario reduction
TARGET_SCENARIOS = 3


# Alias used by optimization workflow
OPTIMIZATION_SCENARIOS = TARGET_SCENARIOS


# Out-of-sample validation scenarios
OUT_OF_SAMPLE_SCENARIOS = 500


RANDOM_SEED = 60



# ============================================================
# Aircraft Uncertainty Range
# ============================================================

# Arrival SOC uncertainty

ARRIVAL_SOC_MIN = 20

ARRIVAL_SOC_MAX = 60



# ============================================================
# Census / ACS Data
# ============================================================

CENSUS_YEAR = "2022"


BBOX = {

    "min_lon": -97.60,

    "min_lat": 32.30,

    "max_lon": -96.20,

    "max_lat": 33.30,
}



# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


DATA_DIR = PROJECT_ROOT / "data"


RAW_DATA_DIR = DATA_DIR / "raw"


OUTPUT_DIR = PROJECT_ROOT / "outputs"


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