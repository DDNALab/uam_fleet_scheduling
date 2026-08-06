"""
config.py
=========
Central configuration for the eVTOL stochastic scheduling pipeline.
All tunable constants live here so that experiments only need to import
this module — no need to touch individual pipeline modules.
"""

from pathlib import Path


# ---------------------------------------------------------------------------
# Energy / Time-of-Use Pricing
# ---------------------------------------------------------------------------
BASE_PRICE  = 0.1482   # $/kWh — avg residential rate, Dallas (PowerChoiceTexas, 2025)
M_PEAK      = 1.5      # CoServ TOU: peak multiplier  (~20.92 ¢/kWh, 4–9 pm)
M_SHOULDER  = 1.05     # Shoulder multiplier (transitional)
M_OFF       = 0.6      # CoServ TOU: off-peak multiplier (~8.56 ¢/kWh)

# ---------------------------------------------------------------------------
# eVTOL / Battery Parameters
# ---------------------------------------------------------------------------
SOC_MIN          = 70.0    # Minimum required state of charge (%)
BATTERY_CAPACITY = 150.0   # Battery capacity (kWh)
CHARGING_RATE    = 125     # Charging power (kW)
EVTOL_CAPACITY   = 4       # Max passengers per eVTOL
# Reference: Z. Wu, Y. Zhang, Optimal eVTOL charging and passenger scheduling,
#            AIAA Aviation 2020 Forum, 2020.

# ---------------------------------------------------------------------------
# Simulation Horizon
# ---------------------------------------------------------------------------
HORIZON_START   = 480    # 8:00 AM in minutes since midnight
HORIZON_END     = 600    # 10:00 AM in minutes since midnight
BIN_SIZE        = 10     # Minutes per time bin
HORIZON_MINUTES = HORIZON_END - HORIZON_START          # 120 min
L_PERIODS       = HORIZON_MINUTES // BIN_SIZE          # 12 periods

# ---------------------------------------------------------------------------
# Demand / Fleet
# ---------------------------------------------------------------------------
N_EVTOL               = 8
TOTAL_PASSENGER_TRIPS = 50   # Will be overridden dynamically from gravity model
HUB_NAME              = "DFW Airport"

# Trips by destination (used as demand-model target shares)
TRIPS_BY_DEST = {
    "Dallas CBD":                   20,
    "Dallas Love Field Airport":    18,
    "Arlington Municipal Airport":  12,
}

# ---------------------------------------------------------------------------
# Scenario Generation / Reduction
# ---------------------------------------------------------------------------
N_SCENARIOS      = 30   # Stochastic scenarios to generate
TARGET_SCENARIOS = 3    # Representative scenarios after reduction
SEED             = 60   # Random seed for reproducibility

# ---------------------------------------------------------------------------
# Model Parameters
# ---------------------------------------------------------------------------
M_FACILITIES = 3    # Number of charging facilities
LOS          = 2    # Level of Service: max waiting time (periods)

# ---------------------------------------------------------------------------
# Census / API
# ---------------------------------------------------------------------------
CENSUS_YEAR    = "2022"

# DFW bounding box
BBOX = {
    "min_lon": -97.60, "min_lat": 32.30,
    "max_lon": -96.20, "max_lat": 33.30,
}

# UAM adoption rate
UAM_ADOPTION = 0.30

# Gravity model
CBD_BOOST_INIT  = 2.0
TOTAL_TRIPS_TARGET = 50
GAMMA           = 1.1   # Distance-decay exponent

# ---------------------------------------------------------------------------
# Vertiport Network
# ---------------------------------------------------------------------------
VERTIPORTS_RAW = {
    "Dallas CBD":                            (32.7767,  -96.7970),
    "DFW Airport":                           (32.8998,  -97.0409),
    "Dallas Love Field Airport":             (32.84815, -96.85135),
    "Fort Worth City Center":                (32.75550, -97.33080),
    "Mesquite Metro Airport":                (32.7470,  -96.5304),
    "Dallas Executive Airport (RBD)":        (32.6810,  -96.8690),
    "Denton Municipal Airport":              (33.20083, -97.19778),
    "Fort Worth Meacham International Airport": (32.81969, -97.36319),
    "Arlington Municipal Airport":           (32.6639,  -97.0943),
    "McKinney National Airport":             (33.1753,  -96.5825),
}

# Repository-relative paths. These work regardless of the current directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"

ACS_JSON_PATH = RAW_DATA_DIR / "acs_texas_2022_raw.json"
SHAPEFILE_PATH = RAW_DATA_DIR / "tl_2022_48_tract" / "tl_2022_48_tract.shp"
