"""
stochastic_model.py
===================

Two-stage stochastic MILP model for eVTOL scheduling.

Based on:
Risk-Aware Charging and Service Commitment for Urban Air Mobility
under Correlated Operational Uncertainty


Architecture:

uam_data_pipeline.py
        |
scenario_generator.py
        |
scenario_reduction.py
        |
model_builder.py
        |
THIS FILE


Stage 1 decisions:
    n[d,t] : committed departures
    b[t]   : reserved charging capacity


Stage 2 decisions:
    x[j,t,s]       : aircraft charging
    z[j,d,t,s]     : aircraft departures
    xi[j,t,d,s]    : passenger assignment
    v_plus[d,t,s]  : added departures
    v_minus[d,t,s] : cancelled departures
    b_plus[t,s]    : emergency charging capacity


"""



import pulp
import numpy as np


from collections import defaultdict



# ============================================================
# Helper functions
# ============================================================


def energy_price_per_kwh(
        hour,
        params
):

    """
    Time-of-use electricity price.
    """

    base = params["base_price"]


    if 16 <= hour <= 21:

        return (
            base *
            params["peak_multiplier"]
        )


    elif 12 <= hour < 16:

        return (
            base *
            params["shoulder_multiplier"]
        )


    else:

        return (
            base *
            params["off_multiplier"]
        )



def charging_periods_required(
        soc,
        params
):

    """
    Calculate number of charging periods
    needed to reach minimum SOC.
    """


    soc_min = params["soc_min"]

    if soc >= soc_min:

        return 0


    energy_needed = (

        params["battery_capacity"]
        *
        (soc_min - soc)
        /
        100

    )


    hours = (

        energy_needed
        /
        params["charging_rate"]

    )


    periods = (

        hours
        *
        60
        /
        params["bin_size"]

    )


    return int(
        np.ceil(periods)
    )



# ============================================================
# Main model builder
# ============================================================


def build_stochastic_model(
        model_input
):


    params = model_input["parameters"]

    scenarios = model_input["scenarios"]



    periods = range(
        1,
        params["periods"] + 1
    )



    destinations = sorted(

        list({
            d
            for s in scenarios.values()
            for (t, d) in s["passenger_demand"].keys()


        })

    )



    # --------------------------------------------------------
    # Create MILP
    # --------------------------------------------------------

    model = pulp.LpProblem(

        "TwoStage_eVTOL_Stochastic_Scheduling",

        pulp.LpMaximize

    )



    # ========================================================
    # FIRST STAGE VARIABLES
    # ========================================================


    # committed departures

    n = {}

    for d in destinations:
        for t in periods:
            n[d, t] = pulp.LpVariable(f"Committed_Departure_{d}_{t}",lowBound=0,cat="Integer")





    # reserved charging capacity
    b = {}

    for t in periods:  
        b[t] = pulp.LpVariable(f"Reserved_Charging_{t}",lowBound=0,cat="Integer")

    # ========================================================
    # FIRST STAGE CONSTRAINTS
    # ========================================================



    # charging reservation limit

    for t in periods:


        model += (

            b[t]

            <=

            params["charging_facilities"]

        ), f"Charging_limit_{t}"




    # departure capacity

    for t in periods:


        model += (

            pulp.lpSum(

                n[d,t]

                for d in destinations

            )

            <=

            params["max_departures"]

        ), f"Departure_capacity_{t}"



    # ========================================================
    # SCENARIO VARIABLES STORAGE
    # ========================================================


# --------------------------------------------------------
# Stage 2 variables (PDF aircraft-indexed formulation)
# --------------------------------------------------------

    activation = {}
    charging = {}
    departures = {}
    departure_time = {}
    eligibility = {}
    passengers = {}
    unserved = {}

    added_departures = {}
    cancelled_departures = {}
    emergency_capacity = {}


    for sid, scenario in scenarios.items():

        aircraft_ids = list(
            scenario["aircraft"].keys()
        )


        # ----------------------------------------------------
        # u_js : aircraft activation
        # ----------------------------------------------------
        activation[sid] = pulp.LpVariable.dicts(
            f"Activation_s{sid}",
            aircraft_ids,
            lowBound=0,
            upBound=1,
            cat="Binary"
        )


        # ----------------------------------------------------
        # x_jts : charging start decision
        # ----------------------------------------------------
        charging[sid] = pulp.LpVariable.dicts(
            f"Charge_s{sid}",
            (
                aircraft_ids,
                periods
            ),
            lowBound=0,
            upBound=1,
            cat="Binary"
        )


        # ----------------------------------------------------
        # z_jdks : aircraft departure assignment
        # ----------------------------------------------------
        departures[sid] = pulp.LpVariable.dicts(
            f"Flight_s{sid}",
            (
                aircraft_ids,
                destinations,
                periods
            ),
            lowBound=0,
            upBound=1,
            cat="Binary"
        )


        # ----------------------------------------------------
        # F_js : departure time
        # ----------------------------------------------------
        departure_time[sid] = pulp.LpVariable.dicts(
            f"DepartureTime_s{sid}",
            aircraft_ids,
            lowBound=0,
            cat="Continuous"
        )


        # ----------------------------------------------------
        # Y_jtds : passenger eligibility
        # ----------------------------------------------------
        eligibility[sid] = pulp.LpVariable.dicts(
            f"Eligibility_s{sid}",
            (
                aircraft_ids,
                periods,
                destinations
            ),
            lowBound=0,
            upBound=1,
            cat="Binary"
        )


        # ----------------------------------------------------
        # xi_jtds : passenger assignment
        # ----------------------------------------------------
        passengers[sid] = pulp.LpVariable.dicts(
            f"Passengers_s{sid}",
            (
                aircraft_ids,
                periods,
                destinations
            ),
            lowBound=0,
            cat="Continuous"
        )


        # ----------------------------------------------------
        # eta_tds : unserved passengers
        # ----------------------------------------------------
        unserved[sid] = pulp.LpVariable.dicts(
            f"Unserved_s{sid}",
            (
                periods,
                destinations
            ),
            lowBound=0,
            cat="Continuous"
        )


        # ----------------------------------------------------
        # Recourse variables
        # ----------------------------------------------------

        added_departures[sid] = pulp.LpVariable.dicts(
            f"Added_s{sid}",
            (
                destinations,
                periods
            ),
            lowBound=0,
            cat="Integer"
        )


        cancelled_departures[sid] = pulp.LpVariable.dicts(
            f"Cancelled_s{sid}",
            (
                destinations,
                periods
            ),
            lowBound=0,
            cat="Integer"
        )


        emergency_capacity[sid] = pulp.LpVariable.dicts(
            f"EmergencyCharge_s{sid}",
            periods,
            lowBound=0,
            cat="Integer"
        )



    return {
        "model": model,

        "parameters": model_input["parameters"],

        "variables": {

            "n": n,
            "b": b,

            "activation": activation,
            "charging": charging,
            "departures": departures,
            "departure_time": departure_time,

            "eligibility": eligibility,

            "passengers": passengers,
            "unserved": unserved,

            "added_departures": added_departures,
            "cancelled_departures": cancelled_departures,

            "emergency_capacity": emergency_capacity
        },

        "destinations": destinations,

        "periods": periods,

        "scenarios": scenarios
    }
# ============================================================
# CONTINUE stochastic_model.py
# ============================================================


def add_constraints_and_objective(
        model_data
):


    model = model_data["model"]

    vars = model_data["variables"]

    params = model_data["scenarios"]

    destinations = model_data["destinations"]

    periods = model_data["periods"]
    
    M = len(periods) + 100


    n = vars["n"]

    b = vars["b"]

    activation = vars["activation"]

    charging = vars["charging"]

    departures = vars["departures"]

    passengers = vars["passengers"]

    added_departures = vars["added_departures"]

    cancelled_departures = vars["cancelled_departures"]

    emergency_capacity = vars["emergency_capacity"]



    # ========================================================
    # Scenario recourse constraints
    # ========================================================


    scenario_costs = {}



    for sid, scenario in params.items():


        aircraft = scenario["aircraft"]

        demand = scenario["passenger_demand"]
        


        # -------------------------------
        # Aircraft charging constraints
        # -------------------------------


        for j, info in aircraft.items():


            arrival_period = info["arrival_period"]

            soc = info["initial_soc"]


            charge_duration = charging_periods_required(

                soc,

                model_data["parameters"]

            )


            for t in periods:


                # cannot charge before arrival

                if t < arrival_period:


                    model += (

                        charging[sid][j][t]

                        ==

                        0

                    ), f"Before_arrival_charge_{sid}_{j}_{t}"



        # -------------------------------
        # Charging capacity
        # -------------------------------


        for t in periods:

            active_charging = []


            for j, info in aircraft.items():

                soc = info["initial_soc"]

                charge_duration = charging_periods_required(
                    soc,
                    model_data["parameters"]
                )


                for start in periods:

                    if start <= t < start + charge_duration:

                        active_charging.append(
                            charging[sid][j][start]
                        )


            model += (
                pulp.lpSum(active_charging)
                <=
                b[t]
                +
                emergency_capacity[sid][t]
            ), f"Charging_capacity_{sid}_{t}"

        # ------------------------------------------------
        # Emergency capacity limit
        # ------------------------------------------------

        for t in periods:

            model += (
                emergency_capacity[sid][t]
                <=
                model_data["parameters"]["charging_facilities"]
                -
                b[t]
            ), f"Emergency_limit_{sid}_{t}"

        # -------------------------------
        # Aircraft cannot fly while charging
        # -------------------------------


        for j in aircraft:


            for t in periods:


                model += (

                    pulp.lpSum(

                        departures[sid][j][d][t]

                        for d in destinations

                    )

                    +

                    charging[sid][j][t]

                    <=

                    1

                ), f"Charge_or_fly_{sid}_{j}_{t}"



        # ====================================================
        # Each activated aircraft performs one flight
        # ------------------------------------------------

        for j in aircraft:

            model += (
                pulp.lpSum(
                    departures[sid][j][d][k]
                    for d in destinations
                    for k in periods
                )
                ==
                activation[sid][j]
            ), f"One_flight_{sid}_{j}"

        # ------------------------------------------------
        # PDF Eq. (9)
        # Departure time definition
        # ------------------------------------------------

        for j in aircraft:

            model += (
                departure_time[sid][j]
                ==
                pulp.lpSum(
                    k *
                    departures[sid][j][d][k]
                    for d in destinations
                    for k in periods
                )
            ), f"Departure_time_{sid}_{j}"

        # ------------------------------------------------
        # PDF Eq. (10)
        # Aircraft cannot depart before charging completes
        # ------------------------------------------------

        for j, info in aircraft.items():

            soc = info["initial_soc"]

            charge_duration = charging_periods_required(
                soc,
                model_data["parameters"]
            )


            model += (
                departure_time[sid][j]
                >=
                pulp.lpSum(
                    (t + charge_duration)
                    *
                    charging[sid][j][t]
                    for t in periods
                )
            ), f"Charge_before_departure_{sid}_{j}"

        # ------------------------------------------------
        # PDF Eq. (11)
        # Hub departure capacity
        # ------------------------------------------------

        for t in periods:

            model += (
                pulp.lpSum(
                    departures[sid][j][d][t]
                    for j in aircraft
                    for d in destinations
                )
                <=
                model_data["parameters"]["max_departures"]
            ), f"Departure_capacity_{sid}_{t}"

        # ====================================================
        # Passenger capacity
        # ====================================================


        for j in aircraft:


            for d in destinations:


                for t in periods:


                    model += (

                        passengers[sid][j][d][t]

                        <=

                        model_data["parameters"]
                        ["evtol_capacity"]

                        *

                        departures[sid][j][d][t]

                    ), (
                        f"Passenger_capacity_{sid}_{j}_{d}_{t}"
                    )



        # ------------------------------------------------
        # PDF Eq. (17)
        # Demand conservation
        # ------------------------------------------------

        for t in periods:

            for d in destinations:

                demand_value = demand.get(
                    (t,d),
                    0
                )


                model += (
                    pulp.lpSum(
                        passengers[sid][j][t][d]
                        for j in aircraft
                    )
                    +
                    unserved[sid][t][d]
                    ==
                    demand_value
                ), f"Demand_balance_{sid}_{t}_{d}"

        # ------------------------------------------------
        # PDF Eq. (14)
        # Passenger capacity
        # ------------------------------------------------

        for j in aircraft:

            for t in periods:

                for d in destinations:

                    model += (
                        passengers[sid][j][t][d]
                        <=
                        model_data["parameters"]["evtol_capacity"]
                        *
                        eligibility[sid][j][t][d]
                    ), f"Passenger_capacity_{sid}_{j}_{t}_{d}"
                    
        # ------------------------------------------------
        # Eligibility link
        # ------------------------------------------------

        for j in aircraft:

            for t in periods:

                for d in destinations:

                    model += (
                        eligibility[sid][j][t][d]
                        <=
                        pulp.lpSum(
                            departures[sid][j][d][k]
                            for k in periods
                        )
                    ), f"Eligibility_link_{sid}_{j}_{t}_{d}"

        # ====================================================
        # Link Stage 1 commitment and Stage 2 operation
        # ====================================================


    for d in destinations:

        for t in periods:

            actual_departures = pulp.lpSum(
                departures[sid][j][d][t]
                for j in aircraft
            )


            model += (
                actual_departures
                ==
                n[(d,t)]
                +
                added_departures[sid][d][t]
                -
                cancelled_departures[sid][d][t]
            ), f"Commitment_link_{sid}_{d}_{t}"

        # ------------------------------------------------
        # PDF Eq. (13)
        # Cannot cancel more than committed
        # ------------------------------------------------

        for d in destinations:

            for t in periods:

                model += (
                    cancelled_departures[sid][d][t]
                    <=
                    n[(d,t)]
                ), f"Cancellation_limit_{sid}_{d}_{t}"

        # ------------------------------------------------
        # PDF Eq. (18)-(19)
        # Waiting time feasibility
        # ------------------------------------------------

        for j in aircraft:

            for t in periods:

                for d in destinations:

                    model += (
                        departure_time[sid][j]
                        >=
                        t
                        -
                        M*(1-eligibility[sid][j][t][d])
                    ), f"Wait_lower_{sid}_{j}_{t}_{d}"


                    model += (
                        departure_time[sid][j]
                        <=
                        t
                        +
                        model_data["parameters"]["los_periods"]
                        +
                        M*(1-eligibility[sid][j][t][d])
                    ), f"Wait_upper_{sid}_{j}_{t}_{d}"

        # ====================================================
        # Build scenario profit
        # ====================================================


        revenue = pulp.lpSum(

            passengers[sid][j][d][t]

            *

            model_data["parameters"]
            ["destination_fares"]
            .get(d,0)


            for j in aircraft

            for d in destinations

            for t in periods

        )


        added_cost = pulp.lpSum(

            added_departures[sid][d][t]

            *

            model_data["parameters"]
            ["costs"]
            ["added_departure"]

            for d in destinations

            for t in periods

        )


        cancel_cost = pulp.lpSum(

            cancelled_departures[sid][d][t]

            *

            model_data["parameters"]
            ["costs"]
            ["cancelled_departure"]

            for d in destinations

            for t in periods

        )


        emergency_cost = pulp.lpSum(

            emergency_capacity[sid][t]

            *

            model_data["parameters"]
            ["costs"]
            ["emergency_capacity"]

            for t in periods

        )


        scenario_costs[sid] = (

            revenue

            -

            added_cost

            -

            cancel_cost

            -

            emergency_cost

        )



    # ========================================================
    # CVaR variables
    # ========================================================
    cvar_zeta = pulp.LpVariable(
        "CVaR_zeta",
        lowBound=0,
        cat="Continuous"
    )


    cvar_rho = {}

    alpha = model_data["parameters"]["cvar"]["alpha"]

    theta = model_data["parameters"]["cvar"]["weight"]



    eta = pulp.LpVariable(

        "CVaR_eta",

        lowBound=None

    )


    rho = pulp.LpVariable.dicts(

        "CVaR_excess",

        params.keys(),

        lowBound=0

    )

    cvar_rho = {}
    scenarios = model_data["scenarios"]

    for sid in scenarios:

        cvar_rho[sid] = pulp.LpVariable(
            f"CVaR_rho_{sid}",
            lowBound=0,
            cat="Continuous"
        )

    for sid in params:


        model += (

            rho[sid]

            >=

            eta

            -

            scenario_costs[sid]

        ), f"CVaR_constraint_{sid}"



    expected_recourse = pulp.lpSum(

        params[sid]["probability"]

        *

        scenario_costs[sid]

        for sid in params

    )



    cvar = (

        eta

        +

        (

            1/(1-alpha)

        )

        *

        pulp.lpSum(

            params[sid]["probability"]

            *

            rho[sid]

            for sid in params

        )

    )



    # ========================================================
    # First-stage costs
    # ========================================================


    commitment_cost = pulp.lpSum(

        n[(d,t)]

        *

        model_data["parameters"]
        ["costs"]
        ["commitment"]

        for d in destinations

        for t in periods

    )


    reservation_cost = pulp.lpSum(

        b[t]

        *

        model_data["parameters"]
        ["costs"]
        ["charging_reservation"]

        for t in periods

    )



    # ========================================================
    # Final objective
    # ========================================================


    model += (

        expected_recourse

        -

        commitment_cost

        -

        reservation_cost

        -

        theta*cvar

    )


    return model



# ============================================================
# Solve
#
# Superseded: the live solver entry point is defined at the end of
# this module (Gurobi, via PuLP's native GUROBI interface).
# ============================================================
def add_constraints_and_objective(model_data):

    model = model_data["model"]

    vars = model_data["variables"]

    params = model_data["parameters"]

    scenarios = model_data["scenarios"]

    destinations = model_data["destinations"]

    periods = model_data["periods"]


    n = vars["n"]
    b = vars["b"]

    activation = vars["activation"]
    charging = vars["charging"]
    departures = vars["departures"]
    departure_time = vars["departure_time"]

    eligibility = vars["eligibility"]
    passengers = vars["passengers"]
    unserved = vars["unserved"]

    added_departures = vars["added_departures"]
    cancelled_departures = vars["cancelled_departures"]
    emergency_capacity = vars["emergency_capacity"]


    # Big-M for waiting constraints later
    M = len(periods) + 100


    # ========================================================
    # FIRST STAGE CONSTRAINTS
    # ========================================================


    # Eq. (3)
    # Reserved charging capacity

    for t in periods:

        model += (
            b[t]
            <=
            params["charging_facilities"]
        ), f"Charging_reservation_limit_{t}"


    # Eq. (4)
    # Departure reservation capacity

    for t in periods:

        model += (
            pulp.lpSum(
                n[d,t]
                for d in destinations
            )
            <=
            params["max_departures"]
        ), f"Departure_reservation_limit_{t}"



    # ========================================================
    # SCENARIO RECOURSE CONSTRAINTS
    # ========================================================


    for sid, scenario in scenarios.items():


        aircraft = scenario["aircraft"]
        demand = scenario["passenger_demand"]


        # ====================================================
        # CHARGING CONSTRAINTS
        # PDF Eq. (5)-(7)
        # ====================================================


        for j, info in aircraft.items():


            soc = info["initial_soc"]


            charge_duration = charging_periods_required(
                soc,
                params
            )


            # --------------------------------------------
            # Eq. (5)
            #
            # An activated aircraft charges exactly once
            # --------------------------------------------

            model += (

                pulp.lpSum(
                    charging[sid][j][t]
                    for t in periods
                )

                ==

                activation[sid][j]

            ), f"Charging_assignment_{sid}_{j}"



            # Aircraft cannot charge before arrival

            arrival = info["arrival_period"]


            for t in periods:

                if t < arrival:

                    model += (

                        charging[sid][j][t]

                        ==

                        0

                    ), f"No_charge_before_arrival_{sid}_{j}_{t}"



        # --------------------------------------------
        # Eq. (6)
        #
        # Charger occupancy
        # --------------------------------------------


        for t in periods:


            active_charging = []


            for j, info in aircraft.items():


                duration = charging_periods_required(
                    info["initial_soc"],
                    params
                )


                for start in periods:


                    if start <= t < start + duration:

                        active_charging.append(
                            charging[sid][j][start]
                        )


            model += (

                pulp.lpSum(active_charging)

                <=

                b[t]

                +

                emergency_capacity[sid][t]

            ), f"Charging_capacity_{sid}_{t}"



        # --------------------------------------------
        # Eq. (7)
        #
        # Emergency charger limit
        # --------------------------------------------


        for t in periods:

            model += (

                emergency_capacity[sid][t]

                <=

                params["charging_facilities"]

                -

                b[t]

            ), f"Emergency_capacity_limit_{sid}_{t}"



        # ====================================================
        # AIRCRAFT OPERATION
        # PDF Eq. (8)-(11)
        # ====================================================


        for j, info in aircraft.items():


            # --------------------------------------------
            # Eq. (8)
            #
            # Each active aircraft performs one flight
            # --------------------------------------------


            model += (

                pulp.lpSum(

                    departures[sid][j][d][k]

                    for d in destinations

                    for k in periods

                )

                ==

                activation[sid][j]

            ), f"Aircraft_one_flight_{sid}_{j}"



            # --------------------------------------------
            # Eq. (9)
            #
            # Departure time definition
            # --------------------------------------------


            model += (

                departure_time[sid][j]

                ==

                pulp.lpSum(

                    k *

                    departures[sid][j][d][k]

                    for d in destinations

                    for k in periods

                )

            ), f"Departure_time_definition_{sid}_{j}"



            # --------------------------------------------
            # Eq. (10)
            #
            # Cannot depart before charging finishes
            # --------------------------------------------


            duration = charging_periods_required(
                info["initial_soc"],
                params
            )


            model += (

                departure_time[sid][j]

                >=

                pulp.lpSum(

                    (t + duration)

                    *

                    charging[sid][j][t]

                    for t in periods

                )

            ), f"Charging_before_departure_{sid}_{j}"



        # --------------------------------------------
        # Eq. (11)
        #
        # Departure capacity
        # --------------------------------------------


        for t in periods:


            model += (

                pulp.lpSum(

                    departures[sid][j][d][t]

                    for j in aircraft

                    for d in destinations

                )

                <=

                params["max_departures"]

            ), f"Hub_departure_capacity_{sid}_{t}"

        # ====================================================
        # COMMITMENT CONSTRAINTS
        # PDF Eq. (12)-(13)
        # ====================================================


        for d in destinations:

            for t in periods:


                # --------------------------------------------
                # Eq. (12)
                #
                # Actual flights =
                # committed flights
                # + added flights
                # - cancelled flights
                # --------------------------------------------


                actual_departures = pulp.lpSum(

                    departures[sid][j][d][t]

                    for j in aircraft

                )


                model += (

                    actual_departures

                    ==

                    n[d,t]

                    +

                    added_departures[sid][d][t]

                    -

                    cancelled_departures[sid][d][t]

                ), f"Commitment_link_{sid}_{d}_{t}"



                # --------------------------------------------
                # Eq. (13)
                #
                # Cannot cancel more than committed
                # --------------------------------------------


                model += (

                    cancelled_departures[sid][d][t]

                    <=

                    n[d,t]

                ), f"Cancellation_limit_{sid}_{d}_{t}"





        # ====================================================
        # PASSENGER ASSIGNMENT CONSTRAINTS
        # PDF Eq. (14)-(19)
        # ====================================================



        # --------------------------------------------
        # Eq. (17)
        #
        # Demand conservation:
        #
        # Demand = served + unserved
        # --------------------------------------------


        for t in periods:

            for d in destinations:


                demand_value = demand.get(
                    (t,d),
                    0
                )


                model += (

                    pulp.lpSum(

                        passengers[sid][j][t][d]

                        for j in aircraft

                    )

                    +

                    unserved[sid][t][d]

                    ==

                    demand_value

                ), f"Demand_balance_{sid}_{t}_{d}"



        # --------------------------------------------
        # Eq. (14)
        #
        # Aircraft passenger capacity
        # --------------------------------------------


        for j in aircraft:

            for t in periods:

                for d in destinations:


                    model += (

                        passengers[sid][j][t][d]

                        <=

                        params["evtol_capacity"]

                        *

                        eligibility[sid][j][t][d]

                    ), (
                        f"Passenger_capacity_{sid}_{j}_{t}_{d}"
                    )



        # --------------------------------------------
        # Link eligibility to flight
        #
        # If aircraft does not fly to destination d,
        # it cannot serve passengers to d
        # --------------------------------------------


        for j in aircraft:

            for t in periods:

                for d in destinations:


                    model += (

                        eligibility[sid][j][t][d]

                        <=

                        pulp.lpSum(

                            departures[sid][j][d][k]

                            for k in periods

                        )

                    ), (
                        f"Eligibility_flight_link_{sid}_{j}_{t}_{d}"
                    )



        # --------------------------------------------
        # Eq. (18)-(19)
        #
        # Waiting time constraints
        #
        # If Y=1:
        #
        # t <= F <= t + LoS
        #
        # --------------------------------------------


        for j in aircraft:

            for t in periods:

                for d in destinations:


                    # Flight cannot depart before passenger arrival


                    model += (

                        departure_time[sid][j]

                        >=

                        t

                        -

                        M *

                        (
                            1 -
                            eligibility[sid][j][t][d]
                        )

                    ), (
                        f"Waiting_lower_{sid}_{j}_{t}_{d}"
                    )



                    # Passenger cannot wait beyond LoS


                    model += (

                        departure_time[sid][j]

                        <=

                        t

                        +

                        params["los_periods"]

                        +

                        M *

                        (
                            1 -
                            eligibility[sid][j][t][d]
                        )

                    ), (
                        f"Waiting_upper_{sid}_{j}_{t}_{d}"
                    )

    # ========================================================
    # SCENARIO PROFIT
    # ========================================================

    scenario_profit = {}


    for sid, scenario in scenarios.items():

        aircraft = scenario["aircraft"]
        demand = scenario["passenger_demand"]


        # --------------------------------------------
        # Revenue from served passengers
        # --------------------------------------------

        revenue = pulp.lpSum(

            passengers[sid][j][t][d]

            *

            params["destination_fares"].get(
                d,
                0
            )

            for j in aircraft

            for t in periods

            for d in destinations

        )


        # --------------------------------------------
        # Charging cost
        # --------------------------------------------

        charging_energy = {}

        for j, info in aircraft.items():

            charging_energy[j] = (
                params["battery_capacity"]
                *
                max(
                    0,
                    params["soc_min"]
                    -
                    info["initial_soc"]
                )
                /
                100
            )
        charging_cost = pulp.lpSum(

            charging[sid][j][t]
            *
            charging_energy[j]
            *
            energy_price_per_kwh(
                (
                    params["horizon"]["start"]
                    +
                    (t-1)*params["bin_size"]
                )/60,
                params
            )

            for j in aircraft
            for t in periods

        )


        # --------------------------------------------
        # Flight operating cost
        # --------------------------------------------

        flight_cost = pulp.lpSum(

            departures[sid][j][d][t]

            *

            params["costs"]["flight"]

            for j in aircraft

            for d in destinations

            for t in periods

        )


        # --------------------------------------------
        # Emergency charging cost
        # --------------------------------------------

        emergency_cost = pulp.lpSum(

            emergency_capacity[sid][t]

            *

            params["costs"]["emergency_capacity"]

            for t in periods

        )


        # --------------------------------------------
        # Unserved passenger penalty
        # --------------------------------------------

        unserved_cost = pulp.lpSum(

            unserved[sid][t][d]

            *

            params["costs"]["unserved"]

            for t in periods

            for d in destinations

        )


        # --------------------------------------------
        # Added / cancelled commitment costs
        # --------------------------------------------

        added_cost = pulp.lpSum(

            added_departures[sid][d][t]

            *

            params["costs"]["added_departure"]

            for d in destinations

            for t in periods

        )


        cancelled_cost = pulp.lpSum(

            cancelled_departures[sid][d][t]

            *

            params["costs"]["cancelled_departure"]

            for d in destinations

            for t in periods

        )


        scenario_profit[sid] = (

            revenue

            -

            charging_cost

            -

            flight_cost

            -

            emergency_cost

            -

            unserved_cost

            -

            added_cost

            -

            cancelled_cost

        )



    # ========================================================
    # CVaR FORMULATION
    # PDF Eq. (20)-(22)
    # ========================================================


    alpha = params["cvar"]["alpha"]

    theta = params["cvar"]["weight"]



    # VaR variable

    zeta = pulp.LpVariable(
        "CVaR_zeta",
        lowBound=None,
        cat="Continuous"
    )


    # Excess loss variables

    rho = pulp.LpVariable.dicts(

        "CVaR_rho",

        scenarios.keys(),

        lowBound=0,

        cat="Continuous"

    )



    for sid, scenario in scenarios.items():
        demand = scenario["passenger_demand"]


        # --------------------------------------------
        # Scenario loss
        #
        # L_s =
        # unserved passengers
        # +
        # cancelled flights penalty
        # --------------------------------------------


        scenario_loss = (

            pulp.lpSum(

                unserved[sid][t][d]

                for t in periods

                for d in destinations

            )

            +

            params["evtol_capacity"]

            *

            pulp.lpSum(

                cancelled_departures[sid][d][t]

                for d in destinations

                for t in periods

            )

        )



        # Eq. (20)

        model += (

            rho[sid]

            >=

            scenario_loss

            -

            zeta

        ), f"CVaR_excess_{sid}"



    # ========================================================
    # EXPECTED PROFIT
    # ========================================================


    expected_profit = pulp.lpSum(

        scenarios[sid]["probability"]

        *

        scenario_profit[sid]

        for sid in scenarios

    )



    # ========================================================
    # CVaR TERM
    # ========================================================


    cvar = (

        zeta

        +

        (

            1/(1-alpha)

        )

        *

        pulp.lpSum(

            scenarios[sid]["probability"]

            *

            rho[sid]

            for sid in scenarios

        )

    )



    # ========================================================
    # FIRST STAGE COSTS
    # ========================================================


    commitment_cost = pulp.lpSum(

        n[d,t]

        *

        params["costs"]["commitment"]

        for d in destinations

        for t in periods

    )



    reservation_cost = pulp.lpSum(

        b[t]

        *

        params["costs"]["charging_reservation"]

        for t in periods

    )



    # ========================================================
    # FINAL OBJECTIVE
    # ========================================================


    model += (

        expected_profit

        -

        commitment_cost

        -

        reservation_cost

        -

        theta*cvar

    )


    return model



# ============================================================
# Solve
# ============================================================


#: Solver preference order used when solver="auto".
#: Gurobi first (strongest), then HiGHS (open source, no size cap), then
#: CBC as a last resort. GLPK is deliberately absent: it does not scale to
#: a model of this size.
SOLVER_PREFERENCE = ("GUROBI", "HIGHS", "CBC")


def _make_solver(
        name,
        model,
        time_limit,
        gap_rel,
        threads,
        warm_start,
        mip_focus,
        cuts,
        presolve,
        log_path,
        verbose
):
    """
    Instantiate one PuLP solver, or return (None, reason) if unavailable.

    Gurobi's trial ("size-limited") license caps the model at 2000 variables
    and 2000 constraints, so `available()` returning True is not enough -- we
    also project the model size onto that cap and reject Gurobi if it is
    exceeded, rather than letting the solve die deep inside gurobipy.
    """

    name = name.upper()

    # ------------------------------------------------------------
    # Gurobi
    # ------------------------------------------------------------

    if name == "GUROBI":

        solver = pulp.GUROBI(

            msg=verbose,

            timeLimit=time_limit,

            gapRel=gap_rel,

            warmStart=warm_start,

            logPath=log_path,

            MIPFocus=mip_focus,

            Cuts=cuts,

            Presolve=presolve
        )

        if not solver.available():

            return None, "gurobipy unavailable or not licensed"

        nvars = len(model.variables())

        ncons = sum(
            len(c) if isinstance(c, list) else 1

            for c in model.constraints.values()
        )

        # Trial licenses allow at most 2000 rows and 2000 columns.
        if nvars > 2000 or ncons > 2000:

            return None, (
                "size-limited license: model has %d vars / %d constraints, "
                "trial cap is 2000 each" % (nvars, ncons)
            )

        if threads:

            solver.solver_params["Threads"] = threads

        return solver, None

    # ------------------------------------------------------------
    # HiGHS (open source, no size cap, in-process via highspy)
    # ------------------------------------------------------------

    if name == "HIGHS":

        solver = pulp.HiGHS(

            msg=verbose,

            timeLimit=time_limit,

            gapRel=gap_rel,

            threads=threads or None,

            logPath=log_path
        )

        if not solver.available():

            return None, "highspy unavailable"

        return solver, None

    # ------------------------------------------------------------
    # CBC (last resort)
    # ------------------------------------------------------------

    if name == "CBC":

        solver = pulp.PULP_CBC_CMD(

            msg=verbose,

            timeLimit=time_limit,

            gapRel=gap_rel,

            threads=threads or None
        )

        if not solver.available():

            return None, "cbc unavailable"

        return solver, None

    return None, "unknown solver %r" % name


def solve_model(
        model,
        time_limit=600,
        gap_rel=0.02,
        threads=0,
        warm_start=False,
        mip_focus=1,
        cuts=2,
        presolve=2,
        log_path=None,
        verbose=True,
        solver="auto"
):
    """
    Solve the two-stage stochastic MILP.

    solver : "auto" (default) walks SOLVER_PREFERENCE and uses the first
             solver that is both available and able to hold this model.
             Pass "gurobi", "highs" or "cbc" to force one.

    time_limit : max seconds before returning the best solution found
    gap_rel    : relative MIP gap tolerance (0.02 = stop within 2% of optimal)
    threads    : thread count (0 = let the solver choose)
    warm_start : seed a MIP start (Gurobi only)
    mip_focus  : Gurobi MIPFocus; 0 balanced, 1 feasible fast,
                 2 prove optimality, 3 bound movement
    cuts       : Gurobi cut aggressiveness (0-3)
    presolve   : Gurobi presolve level (0-2)
    log_path   : optional file to write the solver log to
    verbose    : show the solver log
    """

    if solver == "auto":

        candidates = SOLVER_PREFERENCE

    else:

        candidates = (solver,)

    chosen = None

    skip_reasons = []

    for name in candidates:

        inst, reason = _make_solver(

            name, model, time_limit, gap_rel, threads, warm_start,

            mip_focus, cuts, presolve, log_path, verbose
        )

        if inst is not None:

            chosen = (name, inst)

            break

        skip_reasons.append("%s: %s" % (name, reason))

        print("[info] skipping %s -- %s" % (name, reason))

    if chosen is None:

        raise RuntimeError(
            "No usable MILP solver found. Tried: %s"
            % "; ".join(skip_reasons)
        )

    solver_name, solver = chosen

    print("[info] solving with %s" % solver_name)

    model.solve(solver)

    # ------------------------------------------------------------
    # Recover a trustworthy status.
    #
    # PuLP's HiGHS backend maps HiGHS' optional/limit statuses onto
    # LpStatusOptimal, so a run that merely ran out of time can be reported
    # as "Optimal". Query HiGHS directly and, if it stopped early, report
    # the honest bound and gap instead of a false optimality claim.
    # ------------------------------------------------------------

    status = pulp.LpStatus[model.status]

    proven_optimal = (status == "Optimal")

    bound = None

    if solver_name == "HIGHS":

        try:

            import highspy

            hs = model.solverModel.getModelStatus()

            proven_optimal = (hs == highspy.HighsModelStatus.kOptimal)

            bound = model.solverModel.getObjectiveValue()

            if not proven_optimal:

                status = (
                    "TimeLimit" if hs == highspy.HighsModelStatus.kTimeLimit
                    else "NotOptimal(%s)" % hs.name
                )

        except Exception:

            pass

    try:

        objective = pulp.value(model.objective)

    except Exception:

        objective = None

    if not proven_optimal:

        print(
            "\n[warning] %s did not prove optimality (%s). "
            "The returned solution is the best found, NOT a proved optimum; "
            "increase time_limit or relax gap_rel to close the gap.\n"
            % (solver_name, status)
        )

    return {

        "status":
            status,

        "objective":
            objective,

        "solver":
            solver_name,

        "proven_optimal":
            proven_optimal
    }