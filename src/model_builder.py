"""Convert reduced Scenario objects into solver-ready hybrid-model input."""

import math

from .config import (
    HORIZON_START,
    HORIZON_END,
    BIN_SIZE,
    LOS_PERIODS,
    SOC_REQUIRED,
    BATTERY_CAPACITY,
    CHARGING_RATE,
    BASE_PRICE,
    M_PEAK,
    M_SHOULDER,
    M_OFF,
    EVTOL_CAPACITY,
    M_FACILITIES,
    MAX_DEPARTURES_PER_PERIOD,
    CHARGING_RESERVATION_COST,
    COMMITTED_DEPARTURE_COST,
    ADDED_DEPARTURE_COST,
    CANCELLED_DEPARTURE_COST,
    EMERGENCY_CAPACITY_COST,
    BOOKED_UNSERVED_PASSENGER_COST,
    ONDEMAND_UNSERVED_PASSENGER_COST,
    FLIGHT_OPERATING_COST,
    CVaR_ALPHA,
    CVaR_RISK_WEIGHT,
    CVAR_BOOKED_WEIGHT,
    CVAR_ONDEMAND_WEIGHT,
    VERTIPORTS_RAW,
    INITIAL_FLEET_MAX,
    INITIAL_FLEET_COST,
)


def build_aircraft_block(scenario):
    aircraft = {}
    for a in scenario.aircraft:
        period = (a.arrival_time - HORIZON_START) // BIN_SIZE + 1
        aircraft[a.id] = {
            "arrival_period": int(period),
            "initial_soc": float(a.initial_soc),
        }
    return aircraft


def calculate_fares():
    hub = VERTIPORTS_RAW["DFW Airport"]
    fares = {}
    for name, coord in VERTIPORTS_RAW.items():
        if name == "DFW Airport":
            continue
        lat1, lon1 = hub
        lat2, lon2 = coord
        distance_km = math.sqrt(((lat1 - lat2) * 111) ** 2 + ((lon1 - lon2) * 85) ** 2)
        fares[name] = 20 + 3 * distance_km * 0.621371
    return fares


def build_model_input(reduced_scenarios):
    if not reduced_scenarios:
        raise ValueError("At least one reduced scenario is required")

    periods = (HORIZON_END - HORIZON_START) // BIN_SIZE
    bookings = dict(reduced_scenarios[0].advance_bookings)
    for s in reduced_scenarios[1:]:
        if s.advance_bookings != bookings:
            raise ValueError("Advance bookings must be identical across scenarios (Stage-1 information)")

    model = {
        "parameters": {
            "periods": periods,
            "base_price": BASE_PRICE,
            "peak_multiplier": M_PEAK,
            "shoulder_multiplier": M_SHOULDER,
            "off_multiplier": M_OFF,
            "bin_size": BIN_SIZE,
            "los_periods": LOS_PERIODS,
            "destination_fares": calculate_fares(),
            "horizon": {
                "start": HORIZON_START,
                "end": HORIZON_END,
                "bin_size": BIN_SIZE,
            },
            "battery_capacity": BATTERY_CAPACITY,
            "charging_rate": CHARGING_RATE,
            "soc_min": SOC_REQUIRED,
            "evtol_capacity": EVTOL_CAPACITY,
            "initial_fleet_max": INITIAL_FLEET_MAX,
            "charging_facilities": M_FACILITIES,
            "max_departures": MAX_DEPARTURES_PER_PERIOD,
            "costs": {
                "charging_reservation": CHARGING_RESERVATION_COST,
                "commitment": COMMITTED_DEPARTURE_COST,
                "added_departure": ADDED_DEPARTURE_COST,
                "cancelled_departure": CANCELLED_DEPARTURE_COST,
                "emergency_capacity": EMERGENCY_CAPACITY_COST,
                "unserved_booked": BOOKED_UNSERVED_PASSENGER_COST,
                "unserved_on_demand": ONDEMAND_UNSERVED_PASSENGER_COST,
                "flight": FLIGHT_OPERATING_COST,
                "initial_fleet": INITIAL_FLEET_COST,
            },
            "cvar": {
                "alpha": CVaR_ALPHA,
                "weight": CVaR_RISK_WEIGHT,
                "booked_weight": CVAR_BOOKED_WEIGHT,
                "on_demand_weight": CVAR_ONDEMAND_WEIGHT,
            },
        },
        "advance_bookings": bookings,
        "scenarios": {},
    }

    for s in reduced_scenarios:
        model["scenarios"][s.id] = {
            "probability": float(s.probability),
            "aircraft": build_aircraft_block(s),
            "on_demand_demand": dict(s.on_demand_demand),
            "common_factor": s.common_factor,
        }

    return model
