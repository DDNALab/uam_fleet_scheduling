"""
Demand model.py
===============
Gravity model for UAM demand generation.
Produces OD flows from DFW hub and scenario-ready demand structure.
Includes: passenger + eVTOL arrival sampling, destination sampling,
          distance/fare lookups, and flow map visualization.
"""

import math
import geopandas as gpd
import numpy as np
import pandas as pd

from .config import (
    GAMMA, CBD_BOOST_INIT, TOTAL_TRIPS_TARGET, HUB_NAME,
    HORIZON_START, HORIZON_END, BIN_SIZE,
    TOTAL_PASSENGER_TRIPS, N_EVTOL,
)


# ---------------------------------------------------------------------------
# A) Gravity model (core OD generation)
# ---------------------------------------------------------------------------
def build_gravity_flows(verts_gdf, uam_by_vertiport):
    """
    Compute raw gravity-based flows from DFW to all destinations.
    """
    vp = verts_gdf.merge(
        uam_by_vertiport.rename("uam_pop"),
        left_on="name",
        right_index=True,
        how="inner"
    ).copy()

    vp_proj = vp.to_crs(epsg=3857)
    dfw     = vp_proj[vp_proj["name"] == HUB_NAME].iloc[0]
    dfw_idx = dfw.name
    P_dfw   = float(dfw["uam_pop"])

    flows = []
    for idx, row in vp_proj.iterrows():
        if idx == dfw_idx:
            continue
        P_j     = float(row["uam_pop"])
        dist_km = float(dfw.geometry.distance(row.geometry) / 1000.0)
        dist_km = max(dist_km, 1e-6)
        raw     = (P_dfw * P_j) / (dist_km ** GAMMA)
        flows.append({
            "origin":         HUB_NAME,
            "destination":    row["name"],
            "distance_km":    dist_km,
            "uam_pop_origin": P_dfw,
            "uam_pop_dest":   P_j,
            "raw_gravity":    raw,
        })

    return pd.DataFrame(flows), vp_proj


# ---------------------------------------------------------------------------
# B) Calibration: CBD dominance + scaling
# ---------------------------------------------------------------------------
def calibrate_gravity(flows_df):
    """
    Enforce:
    1. CBD dominance
    2. Total trips normalization
    """
    flows_df            = flows_df.copy()
    cbd_boost           = CBD_BOOST_INIT
    flows_df["raw_adj"] = flows_df["raw_gravity"]
    flows_df.loc[flows_df["destination"] == "Dallas CBD", "raw_adj"] *= cbd_boost

    for _ in range(20):
        top_dest = flows_df.loc[flows_df["raw_adj"].idxmax(), "destination"]
        if top_dest == "Dallas CBD":
            break
        cbd_boost *= 1.5
        flows_df["raw_adj"] = flows_df["raw_gravity"]
        flows_df.loc[flows_df["destination"] == "Dallas CBD", "raw_adj"] *= cbd_boost

    total_raw         = flows_df["raw_adj"].sum()
    K                 = TOTAL_TRIPS_TARGET / total_raw if total_raw > 0 else 0.0
    flows_df["trips"] = (flows_df["raw_adj"] * K).round(2)

    return flows_df, cbd_boost


# ---------------------------------------------------------------------------
# C) Visualization (flow map only)
# ---------------------------------------------------------------------------
def plot_flows_map(flows_df, vp_proj, verts_gdf):
    """
    Visualize OD flows from DFW hub.
    """
    import folium

    m = folium.Map(location=[32.85, -97.0], zoom_start=10)

    for _, r in verts_gdf.iterrows():
        folium.Marker(
            location=[r.geometry.y, r.geometry.x],
            popup=r["name"],
            tooltip=r["name"],
            icon=folium.Icon(color="blue", icon="plane", prefix="fa")
        ).add_to(m)

    for _, row in flows_df.iterrows():
        trips = float(row["trips"])
        if trips < 50:
            continue
        o    = vp_proj[vp_proj["name"] == row["origin"]].geometry.iloc[0]
        d    = vp_proj[vp_proj["name"] == row["destination"]].geometry.iloc[0]
        o_ll = gpd.GeoSeries([o], crs=3857).to_crs(4326).iloc[0]
        d_ll = gpd.GeoSeries([d], crs=3857).to_crs(4326).iloc[0]
        folium.PolyLine(
            locations=[[o_ll.y, o_ll.x], [d_ll.y, d_ll.x]],
            color="black",
            weight=min(20, trips / 100.0),
            opacity=0.65,
            tooltip=f"{row['origin']} → {row['destination']}: {trips:.1f}"
        ).add_to(m)

    return m


# ---------------------------------------------------------------------------
# D) Time-distribution + sampling helpers  (from simulation snippet)
# ---------------------------------------------------------------------------
def haversine_miles(lat1, lon1, lat2, lon2):
    R        = 3958.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi     = math.radians(lat2 - lat1)
    dlambda  = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def make_time_pdf_passengers(x_minutes):
    """
    Bimodal histogram-like demand:
      - strong morning peak  ~ 08:00
      - strong afternoon peak ~ 16:00
      - small baseline everywhere
    """
    m1, s1, w1 = 8 * 60,  55, 1.00
    m2, s2, w2 = 16 * 60, 70, 0.85
    baseline   = 0.08
    y = baseline + w1 * gaussian(x_minutes, m1, s1) + w2 * gaussian(x_minutes, m2, s2)
    y = np.clip(y, 0, None)
    return y / y.sum()


def make_time_pdf_evtols(x_minutes):
    """
    Smooth bimodal curve for eVTOL scheduling:
      - smoother, wider peaks
    """
    m1, s1, w1 = 8 * 60,  90, 1.00
    m2, s2, w2 = 16 * 60, 95, 1.00
    baseline   = 0.02
    y = baseline + w1 * gaussian(x_minutes, m1, s1) + w2 * gaussian(x_minutes, m2, s2)
    y = np.clip(y, 0, None)
    return y / y.sum()


def sample_arrival_times(n, bin_edges, bin_mid, pdf_mid):
    """
    Sample arrival times (minutes since midnight) from a binned PDF.
      - discrete bin selection by pdf_mid
      - uniform draw within the selected bin
    """
    probs = pdf_mid / pdf_mid.sum()
    idx   = np.random.choice(len(bin_mid), size=n, p=probs)
    low   = bin_edges[idx]
    high  = bin_edges[idx + 1]
    return np.random.uniform(low, high)


def sample_destinations(n, trips_by_dest):
    """
    Sample n destination names weighted by trip counts.
    """
    dests   = list(trips_by_dest.keys())
    weights = np.array([trips_by_dest[d] for d in dests], dtype=float)
    weights = weights / weights.sum()
    return np.random.choice(dests, size=n, p=weights)


# ---------------------------------------------------------------------------
# E) Scenario data builder  (arrivals + destinations + fares)
# ---------------------------------------------------------------------------
def build_scenario_data(flows_df, vertiports):
    """
    Build all arrival and destination arrays for the DFW scenario.

    Parameters
    ----------
    flows_df   : calibrated OD DataFrame (from build_demand_model)
    vertiports : dict  {name: (lat, lon)}

    Returns
    -------
    dict with:
        passenger_arrivals       – np.ndarray (minutes since midnight)
        passenger_dest_names     – np.ndarray (destination name per passenger)
        passenger_dest_distances – np.ndarray (miles)
        passenger_dest_fares     – np.ndarray ($)
        evtol_arrivals           – np.ndarray (minutes since midnight)
        BINS                     – bin edges used
        BIN_MID                  – bin centres used
    """
    hub_coord = vertiports[HUB_NAME]

    # Time bins
    BINS    = np.arange(HORIZON_START, HORIZON_END + BIN_SIZE, BIN_SIZE)
    BIN_MID = (BINS[:-1] + BINS[1:]) / 2

    # Distance + fare for every non-hub vertiport
    DESTINATION_DATA = {}
    for name, coord in vertiports.items():
        if name == HUB_NAME:
            continue
        dist = haversine_miles(hub_coord[0], hub_coord[1], coord[0], coord[1])
        fare = round(20 + 3 * dist, 2)
        DESTINATION_DATA[name] = {"distance": dist, "fare": fare}

    # Build TRIPS_BY_DEST from calibrated flows
    TRIPS_BY_DEST = dict(zip(flows_df["destination"], flows_df["trips"]))

    # Validate all destinations exist in the network
    missing = [d for d in TRIPS_BY_DEST if d not in DESTINATION_DATA]
    if missing:
        raise ValueError(
            f"TRIPS_BY_DEST contains destinations not in network "
            f"(excluding hub): {missing}"
        )

    # Passenger arrivals — bimodal PDF
    pdf_pax            = make_time_pdf_passengers(BIN_MID)
    passenger_arrivals = sample_arrival_times(TOTAL_PASSENGER_TRIPS, BINS, BIN_MID, pdf_pax)

    # Passenger destinations + per-passenger distance / fare
    passenger_dest_names     = sample_destinations(TOTAL_PASSENGER_TRIPS, TRIPS_BY_DEST)
    passenger_dest_distances = np.array(
        [DESTINATION_DATA[d]["distance"] for d in passenger_dest_names]
    )
    passenger_dest_fares = np.array(
        [DESTINATION_DATA[d]["fare"] for d in passenger_dest_names]
    )

    # eVTOL arrivals — smooth bimodal PDF
    pdf_evtol      = make_time_pdf_evtols(BIN_MID)
    evtol_arrivals = sample_arrival_times(N_EVTOL, BINS, BIN_MID, pdf_evtol)

    return {
        "passenger_arrivals":       passenger_arrivals,
        "passenger_dest_names":     passenger_dest_names,
        "passenger_dest_distances": passenger_dest_distances,
        "passenger_dest_fares":     passenger_dest_fares,
        "evtol_arrivals":           evtol_arrivals,
        "BINS":                     BINS,
        "BIN_MID":                  BIN_MID,
    }


# ---------------------------------------------------------------------------
# F) Main demand pipeline (used by scenario generator)
# ---------------------------------------------------------------------------
def build_demand_model(verts_gdf, uam_by_vertiport, vertiports=None):
    """
    Full demand pipeline:
    gravity → calibration → scenario data → outputs

    Parameters
    ----------
    verts_gdf        : GeoDataFrame of vertiports
    uam_by_vertiport : Series of UAM-eligible population per vertiport name
    vertiports       : dict {name: (lat, lon)}
                       If provided, scenario arrays (arrivals, destinations,
                       distances, fares) are included in the output dict.

    Returns
    -------
    dict with keys:
        flows, vp_proj, total_trips, cbd_boost
        + all keys from build_scenario_data() if vertiports is provided
    """
    flows_df, vp_proj   = build_gravity_flows(verts_gdf, uam_by_vertiport)
    flows_df, cbd_boost = calibrate_gravity(flows_df)

    result = {
        "flows":       flows_df,
        "vp_proj":     vp_proj,
        "total_trips": float(flows_df["trips"].sum()),
        "cbd_boost":   cbd_boost,
    }

    if vertiports is not None:
        result.update(build_scenario_data(flows_df, vertiports))

    return result
