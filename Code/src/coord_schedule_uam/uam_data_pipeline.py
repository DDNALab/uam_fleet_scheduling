"""
uam_data_pipeline.py
====================

Deterministic UAM demand preparation pipeline.

Creates the information available BEFORE uncertainty:

1. Vertiport network
2. ACS population allocation
3. UAM market potential
4. Gravity-based OD demand
5. Expected passenger demand profile
6. Destination distances and fares


Used as input for:
- Stage 1 commitment decisions:
      n[d,t]
      b[t]

Not responsible for:
- scenarios
- stochastic demand
- aircraft uncertainty
- SoC uncertainty
"""


import json
import math

import geopandas as gpd
import numpy as np
import pandas as pd

from shapely.geometry import Polygon

from .config import (
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
# 1. Vertiports
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
# 4. Merge population
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


    return result.dropna(
        subset=["population"]
    )



# ============================================================
# 5. Assign census tracts to nearest vertiport
# ============================================================

def assign_nearest_vertiport(
        tracts,
        vertiports
):

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
# 6. Calculate U_j market potential
# ============================================================

def calculate_market_potential(
        tracts
):

    """
    PDF:

    U_j = sum_i a P_i

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

    Computes:

    G_DFW,j =
    U_DFW U_j / d^gamma

    """

    hub_value = market[HUB_NAME]


    hub = vertiports[
        vertiports.name == HUB_NAME
    ].iloc[0]


    flows=[]


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
# 9. Complete pipeline
# ============================================================

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