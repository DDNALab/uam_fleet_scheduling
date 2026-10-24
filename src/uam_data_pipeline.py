"""
Deterministic spatial demand inputs for the coordinated UAM model.

Builds what is known before uncertainty: the vertiport network, ACS population
per census tract, the population assigned to each vertiport, gravity-based
origin-destination demand from the DFW hub, and a within-day demand profile.

Paths come from ``config``: the ACS summary JSON, the census tract shapefile
clipped to ``BBOX``, and the raw vertiport coordinates. All three are required;
there is no synthetic fallback if a file is missing.

The output feeds the first-stage commitment decisions ``n[d, t]`` and ``b[t]``.
Scenarios and stochastic demand, aircraft and SoC uncertainty live in
:mod:`scenario_generator`.
"""


import json
import math

import geopandas as gpd
import numpy as np
import pandas as pd

from config import (
    ACS_JSON_PATH,
    SHAPEFILE_PATH,
    BBOX,
    VERTIPORTS_RAW,
    HUB_NAME,
    UAM_ADOPTION,
    GAMMA,
    TOTAL_PASSENGER_TRIPS,
    BIN_SIZE,
    HORIZON_START,
    HORIZON_END,
)



# ============================================================
# 1. Vertiport network
# ============================================================

def load_vertiports():

    df = pd.DataFrame({

        "name": list(VERTIPORTS_RAW.keys()),

        "lat": [
            x[0]
            for x in VERTIPORTS_RAW.values()
        ],

        "lon": [
            x[1]
            for x in VERTIPORTS_RAW.values()
        ]

    })


    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(
            df.lon,
            df.lat
        ),
        crs="EPSG:4326"
    )


    return gdf



# ============================================================
# 2. Census tracts
# ============================================================

def load_census_tracts():

    bbox = (

        BBOX["min_lon"],
        BBOX["min_lat"],
        BBOX["max_lon"],
        BBOX["max_lat"]

    )


    return (
        gpd.read_file(
            SHAPEFILE_PATH,
            bbox=bbox
        )
        .to_crs(4326)
    )



# ============================================================
# 3. ACS population
# ============================================================

def load_population():

    with open(
        ACS_JSON_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        acs = json.load(f)


    columns = acs[0]

    rows = acs[1:]


    df = pd.DataFrame(
        rows,
        columns=columns
    )


    df["GEOID"] = (
        df["state"]
        +
        df["county"]
        +
        df["tract"]
    )


    df["population"] = (
        df["B01003_001E"]
        .astype(int)
    )


    return df[
        [
            "GEOID",
            "population"
        ]
    ]



# ============================================================
# 4. Population merge
# ============================================================

def merge_population(
        tracts,
        population
):

    result = tracts.merge(
        population,
        on="GEOID",
        how="left"
    )


    # Tracts without an ACS population record are dropped.
    return result.dropna(
        subset=["population"]
    )



# ============================================================
# 5. Nearest-vertiport assignment
# ============================================================

def assign_nearest_vertiport(
        tracts,
        vertiports
):

    # Project both layers to EPSG:3857 so ``distance`` is in metres.
    verts = (
        vertiports
        .to_crs(3857)
    )


    tracts_proj = (
        tracts
        .to_crs(3857)
    )


    tracts_proj["centroid"] = (
        tracts_proj
        .geometry
        .centroid
    )


    assignments=[]


    for point in tracts_proj["centroid"]:

        idx = (
            verts.distance(point)
            .idxmin()
        )

        assignments.append(
            vertiports.loc[idx,"name"]
        )


    tracts_proj["nearest_vertiport"] = assignments


    return tracts_proj



# ============================================================
# 6. Market potential
# ============================================================

def calculate_market_potential(
        tracts
):

    """
    Market potential per vertiport, ``U_j = sum_i a P_i``: the UAM adoption
    share times the population of each tract assigned to the vertiport.
    """

    tracts["uam_population"] = (
        tracts["population"]
        *
        UAM_ADOPTION
    )


    return (

        tracts
        .groupby(
            "nearest_vertiport"
        )
        ["uam_population"]
        .sum()

    )



# ============================================================
# 7. Gravity model
# ============================================================

def haversine_km(
        lat1,
        lon1,
        lat2,
        lon2
):

    R = 6371

    p1 = math.radians(lat1)

    p2 = math.radians(lat2)

    dlat = math.radians(lat2-lat1)

    dlon = math.radians(lon2-lon1)


    a = (
        math.sin(dlat/2)**2
        +
        math.cos(p1)
        *
        math.cos(p2)
        *
        math.sin(dlon/2)**2
    )


    return (
        2*R*
        math.atan2(
            math.sqrt(a),
            math.sqrt(1-a)
        )
    )



def gravity_demand(
        market,
        vertiports
):

    """
    Gravity score ``U_hub * U_j / d**GAMMA`` for every non-hub destination,
    scaled so the flows sum to ``TOTAL_PASSENGER_TRIPS``.
    """

    hub_value = market[HUB_NAME]


    hub = vertiports[
        vertiports.name == HUB_NAME
    ].iloc[0]


    flows=[]


    # Only hub-origin flows are modelled, so the hub is never a destination.
    for destination, Uj in market.items():


        if destination == HUB_NAME:
            continue


        dest = vertiports[
            vertiports.name == destination
        ].iloc[0]


        distance = haversine_km(

            hub.lat,
            hub.lon,

            dest.lat,
            dest.lon
        )


        score = (
            hub_value
            *
            Uj
            /
            distance**GAMMA
        )


        flows.append({

            "origin": HUB_NAME,

            "destination": destination,

            "distance_km": distance,

            "gravity_score": score

        })


    df = pd.DataFrame(flows)


    df["expected_trips"] = (

        TOTAL_PASSENGER_TRIPS
        *
        df.gravity_score
        /
        df.gravity_score.sum()

    )


    return df



# ============================================================
# 8. Time distribution
# ============================================================

def gaussian(x,mu,sigma):

    return np.exp(
        -0.5*
        ((x-mu)/sigma)**2
    )



def passenger_time_profile():

    # Two Gaussian peaks at 08:00 and 16:00 (sigma 60 and 80 min) on a 0.05 floor.
    periods = np.arange(
        HORIZON_START,
        HORIZON_END,
        BIN_SIZE
    )


    morning = gaussian(
        periods,
        8*60,
        60
    )


    afternoon = gaussian(
        periods,
        16*60,
        80
    )


    profile = (
        0.05
        +
        morning
        +
        afternoon
    )


    return (
        pd.DataFrame({

            "period": periods,

            "weight":
                profile/profile.sum()

        })
    )



# ============================================================
# 9. Pipeline
# ============================================================

# Chains the steps above; the returned dict holds every intermediate table.

def build_uam_data():

    vertiports = load_vertiports()


    tracts = load_census_tracts()


    population = load_population()


    tracts = merge_population(
        tracts,
        population
    )


    tracts = assign_nearest_vertiport(
        tracts,
        vertiports
    )


    market = calculate_market_potential(
        tracts
    )


    od_demand = gravity_demand(
        market,
        vertiports
    )


    time_profile = passenger_time_profile()



    return {

        "vertiports":
            vertiports,

        "tracts":
            tracts,

        "market":
            market,

        "od_demand":
            od_demand,

        "time_profile":
            time_profile

    }



if __name__ == "__main__":

    data = build_uam_data()

    print(
        data["od_demand"]
    )