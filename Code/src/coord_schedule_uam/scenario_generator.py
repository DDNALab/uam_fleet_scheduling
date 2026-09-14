"""
scenario_generator.py
=====================

Scenario generation for the two-stage stochastic UAM model.

Creates uncertainty scenarios required by the stochastic MILP:

For each scenario s:

1. Passenger demand:
        lambda[t,d,s]

2. Aircraft availability:
        J_s

3. Aircraft arrival time

4. Aircraft initial SoC


Based on:
Risk-Aware Charging and Service Commitment for Urban Air Mobility
under Correlated Operational Uncertainty


This file does NOT:
- optimize flights
- optimize charging
- assign passengers

Those belong to stochastic_model.py
"""


import numpy as np
from dataclasses import dataclass

from scipy.ndimage import gaussian_filter1d


from .config import (
    OPTIMIZATION_SCENARIOS,
    RANDOM_SEED,

    BIN_SIZE,
    HORIZON_START,
    HORIZON_END,

    BATTERY_CAPACITY,
    SOC_REQUIRED,

    ARRIVAL_SOC_MIN,
    ARRIVAL_SOC_MAX,

    AIRCRAFT_PASSENGER_RATIO
)



# ============================================================
# Scenario object
# ============================================================


@dataclass
class Aircraft:

    id: int

    arrival_time: int

    initial_soc: float



@dataclass
class Scenario:

    id: int

    probability: float

    passenger_demand: dict
    # {(period,destination): passengers}


    aircraft: list
    # list of Aircraft


    common_factor: np.ndarray



# ============================================================
# Common uncertainty factor
# ============================================================

def generate_common_factor(
        n_periods,
        seed=None
):

    """
    Generate correlated temporal uncertainty.

    Represents:
    - weather
    - congestion
    - events

    Used to correlate:
    demand
    aircraft arrivals
    """


    if seed is not None:
        np.random.seed(seed)


    g = np.random.normal(
        0,
        1,
        n_periods
    )


    # temporal smoothing

    g = gaussian_filter1d(
        g,
        sigma=1
    )


    g = (
        g - np.mean(g)
    )


    std = np.std(g)


    if std > 0:
        g = g / std


    return g



# ============================================================
# Passenger demand uncertainty
# ============================================================


def generate_passenger_demand(
        expected_demand,
        time_profile,
        common_factor,
        beta=0.35
):

    """
    Generate:

        lambda_tds

    from expected demand.

    The common factor changes demand intensity.

    """

    demand = {}


    periods = len(time_profile)


    for destination, daily_value in expected_demand.items():


        # Demand periods are 1-based so they line up with the model's period
        # set, range(1, n_periods + 1), and with the 1-based arrival_period
        # produced in model_builder.py. time_profile and common_factor stay
        # 0-indexed, hence the t - 1 lookups.
        for t in range(1, periods + 1):


            base = (
                daily_value
                *
                time_profile[t - 1]
            )


            # log-normal perturbation

            intensity = (

                base
                *
                np.exp(
                    beta *
                    common_factor[t - 1]
                    -
                    0.5 *
                    beta**2
                )

            )


            passengers = np.random.poisson(
                max(
                    intensity,
                    0.001
                )
            )


            demand[
                (t, destination)
            ] = int(passengers)


    return demand



# ============================================================
# Aircraft arrival uncertainty
# ============================================================


def generate_aircraft(
        expected_total_passengers,
        common_factor,
        beta=0.35
):

    """
    Generate scenario-dependent aircraft set:

        J_s


    """

    periods = len(common_factor)



    # expected aircraft number

    expected_evtols = (

        expected_total_passengers
        *
        AIRCRAFT_PASSENGER_RATIO

    )



    base_rate = (

        expected_evtols
        /
        periods

    )



    aircraft_rate = []


    for t in range(periods):

        rate = (

            base_rate
            *
            np.exp(
                beta *
                common_factor[t]
                -
                0.5 *
                beta**2
            )

        )


        aircraft_rate.append(
            max(rate,0.01)
        )



    arrivals = np.random.poisson(
        aircraft_rate
    )


    aircraft=[]


    aircraft_id=0


    for period, number in enumerate(arrivals):


        for _ in range(number):


            arrival_time = (

                HORIZON_START
                +
                period *
                BIN_SIZE

            )


            soc = np.random.uniform(
                ARRIVAL_SOC_MIN,
                ARRIVAL_SOC_MAX
            )


            aircraft.append(

                Aircraft(

                    id=aircraft_id,

                    arrival_time=arrival_time,

                    initial_soc=soc

                )

            )


            aircraft_id += 1



    return aircraft



# ============================================================
# Scenario generator
# ============================================================


def generate_scenarios(
        expected_demand,
        time_profile,
        n_scenarios=OPTIMIZATION_SCENARIOS,
        seed=RANDOM_SEED
):


    """

    Main scenario generator.

    Returns:

    [
      Scenario 1,
      Scenario 2,
      ...
    ]

    """

    np.random.seed(seed)



    n_periods = len(time_profile)



    scenarios=[]


    probability = (
        1 /
        n_scenarios
    )



    for s in range(n_scenarios):


        common_factor = generate_common_factor(
            n_periods
        )


        passenger_demand = generate_passenger_demand(

            expected_demand,

            time_profile,

            common_factor

        )


        aircraft = generate_aircraft(

            sum(expected_demand.values()),

            common_factor

        )


        scenarios.append(

            Scenario(

                id=s,

                probability=probability,

                passenger_demand=passenger_demand,

                aircraft=aircraft,

                common_factor=common_factor

            )

        )



    return scenarios



# ============================================================
# Testing
# ============================================================


if __name__ == "__main__":


    # Example only

    demand = {

        "Dallas CBD": 800,

        "Dallas Love Field Airport": 500,

        "Arlington Municipal Airport": 300

    }


    periods = int(
        (HORIZON_END-HORIZON_START)
        /
        BIN_SIZE
    )


    profile = np.ones(periods)

    profile = (
        profile /
        profile.sum()
    )



    scenarios = generate_scenarios(
        demand,
        profile
    )


    for s in scenarios:

        print(
            "Scenario:",
            s.id
        )

        print(
            "Passengers:",
            sum(
                s.passenger_demand.values()
            )
        )

        print(
            "Aircraft:",
            len(s.aircraft)
        )

        print()