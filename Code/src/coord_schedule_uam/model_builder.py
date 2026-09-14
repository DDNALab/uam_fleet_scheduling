"""
model_builder.py
================

Prepare reduced scenarios for the two-stage stochastic MILP.

Input:
    Reduced Scenario objects from scenario_reduction.py


Output:
    Solver-ready model data


Does NOT:
- optimize
- create decisions
- generate uncertainty

It only prepares inputs.
"""


from collections import defaultdict

from .config import BASE_PRICE, VERTIPORTS_RAW
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
    UNSERVED_PASSENGER_COST,
    FLIGHT_OPERATING_COST,

    CVaR_ALPHA,
    CVaR_RISK_WEIGHT
)



# ============================================================
# Convert aircraft
# ============================================================


def build_aircraft_block(
        scenario
):

    aircraft = {}


    for a in scenario.aircraft:

        period = (

            a.arrival_time
            -
            HORIZON_START

        ) // BIN_SIZE + 1


        aircraft[a.id] = {

            "arrival_period":
                int(period),

            "initial_soc":
                float(a.initial_soc)

        }


    return aircraft



# ============================================================
# Convert demand
# ============================================================

from .config import ACTIVE_DESTINATIONS
def build_demand_block(
        scenario
):

    demand = defaultdict(int)


    for key, value in scenario.passenger_demand.items():

        t, destination = key
        if destination not in ACTIVE_DESTINATIONS:
            continue


        demand[
            (
                int(t),
                destination
            )
        ] += int(value)


    return dict(demand)

def calculate_fares():

    hub = VERTIPORTS_RAW["DFW Airport"]

    fares = {}

    for name, coord in VERTIPORTS_RAW.items():

        if name == "DFW Airport":
            continue


        lat1, lon1 = hub
        lat2, lon2 = coord


        distance_km = math.sqrt(
            ((lat1-lat2)*111)**2 +
            ((lon1-lon2)*85)**2
        )


        distance_miles = distance_km * 0.621371


        fares[name] = 20 + 3 * distance_miles


    return fares

# ============================================================
# Main builder
# ============================================================


def build_model_input(
        reduced_scenarios
):


    periods = (

        HORIZON_END
        -
        HORIZON_START

    ) // BIN_SIZE



    model = {


        "parameters": {


            "periods":
                periods,

            "base_price":
                BASE_PRICE,

            "peak_multiplier":
                M_PEAK,

            "shoulder_multiplier":
                M_SHOULDER,

            "off_multiplier":
                M_OFF,

            "bin_size":
                BIN_SIZE,

            "los_periods":
                LOS_PERIODS,

            "destination_fares":
                    calculate_fares(),

            "horizon":
            {

                "start":
                    HORIZON_START,


                "end":
                    HORIZON_END,


                "bin_size":
                    BIN_SIZE,

            },



            "battery_capacity":
                BATTERY_CAPACITY,



            "charging_rate":
                CHARGING_RATE,



            "soc_min":
                SOC_REQUIRED,



            "evtol_capacity":
                EVTOL_CAPACITY,



            "charging_facilities":
                M_FACILITIES,



            "max_departures":
                MAX_DEPARTURES_PER_PERIOD,



            "costs":
            {

                "charging_reservation":
                    CHARGING_RESERVATION_COST,


                "commitment":
                    COMMITTED_DEPARTURE_COST,


                "added_departure":
                    ADDED_DEPARTURE_COST,


                "cancelled_departure":
                    CANCELLED_DEPARTURE_COST,


                "emergency_capacity":
                    EMERGENCY_CAPACITY_COST,


                "unserved":
                    UNSERVED_PASSENGER_COST,


                "flight":
                    FLIGHT_OPERATING_COST

            },



            "cvar":
            {

                "alpha":
                    CVaR_ALPHA,


                "weight":
                    CVaR_RISK_WEIGHT

            }

        },



        "scenarios": {}

    }



    # --------------------------------------------------------
    # Scenario conversion
    # --------------------------------------------------------


    for s in reduced_scenarios:


        model["scenarios"][s.id] = {


            "probability":
                s.probability,


            "aircraft":
                build_aircraft_block(s),



            "passenger_demand":
                build_demand_block(s),



            "common_factor":
                s.common_factor

        }



    return model