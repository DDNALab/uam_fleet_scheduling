"""
Data loading.py
===============
Vertiport definitions, census tracts, ACS population retrieval,
and UAM population allocation by nearest vertiport.
"""

import json

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

from .config import ACS_JSON_PATH, BBOX, UAM_ADOPTION, VERTIPORTS_RAW


# ---------------------------------------------------------------------------
# A) Vertiport definitions
# ---------------------------------------------------------------------------
def load_vertiports():
    """
    Load vertiports from config into GeoDataFrame.
    """
    verts = pd.DataFrame({
        "name": list(VERTIPORTS_RAW.keys()),
        "lat": [v[0] for v in VERTIPORTS_RAW.values()],
        "lon": [v[1] for v in VERTIPORTS_RAW.values()]
    })

    verts_gdf = gpd.GeoDataFrame(
        verts,
        geometry=gpd.points_from_xy(verts["lon"], verts["lat"]),
        crs="EPSG:4326"
    )

    return verts_gdf


# ---------------------------------------------------------------------------
# B) Census tract loading
# ---------------------------------------------------------------------------
def load_census_tracts(shapefile_path):
    """
    Load only census tracts intersecting the DFW study bounds.

    Applying the bounding box in the file reader avoids loading and processing
    the entire statewide shapefile for every experiment.
    """
    bbox = (
        BBOX["min_lon"],
        BBOX["min_lat"],
        BBOX["max_lon"],
        BBOX["max_lat"],
    )
    tracts = gpd.read_file(shapefile_path, bbox=bbox).to_crs(epsg=4326)
    return tracts


# ---------------------------------------------------------------------------
# C) ACS population loading
# ---------------------------------------------------------------------------
def fetch_acs_population(json_path=ACS_JSON_PATH):
    """
    Load ACS 2022 population by census tract (Texas) from local JSON file.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        acs = json.load(f)

    columns = acs[0]
    rows = acs[1:]

    pop = pd.DataFrame(rows, columns=columns)

    pop["GEOID"] = pop["state"] + pop["county"] + pop["tract"]
    pop["population"] = pop["B01003_001E"].astype(int)

    return pop[["GEOID", "population"]]


# ---------------------------------------------------------------------------
# D) Merge population into tracts
# ---------------------------------------------------------------------------
def merge_population(tracts, pop):
    """
    Attach ACS population to census tracts.
    """
    tracts_pop = tracts.merge(pop, on="GEOID", how="left")
    tracts_pop = tracts_pop.dropna(subset=["population"])
    return tracts_pop


# ---------------------------------------------------------------------------
# E) Spatial filtering (DFW bounding box)
# ---------------------------------------------------------------------------
def clip_to_bbox(tracts_pop):
    """
    Clip tracts to DFW study region.
    """
    min_lon = BBOX["min_lon"]
    min_lat = BBOX["min_lat"]
    max_lon = BBOX["max_lon"]
    max_lat = BBOX["max_lat"]

    rect = Polygon([
        (min_lon, min_lat),
        (min_lon, max_lat),
        (max_lon, max_lat),
        (max_lon, min_lat)
    ])

    return tracts_pop[tracts_pop.intersects(rect)].copy()


# ---------------------------------------------------------------------------
# F) Assign nearest vertiport + UAM allocation
# ---------------------------------------------------------------------------
def assign_nearest_vertiport(tracts_in_rect, verts_gdf):
    """
    Assign each tract centroid to nearest vertiport.
    """
    verts_proj = verts_gdf.to_crs(epsg=3857)
    centroids_proj = tracts_in_rect.to_crs(epsg=3857)
    centroids_proj["centroid"] = centroids_proj.geometry.centroid
    centroids_proj = centroids_proj.set_geometry("centroid")

    nearest_idx = centroids_proj.geometry.apply(
        lambda p: verts_proj.distance(p).idxmin()
    )

    centroids_proj["nearest_vertiport"] = nearest_idx.apply(
        lambda i: verts_gdf.loc[i, "name"]
    )

    centroids_proj["uam_population"] = centroids_proj["population"] * UAM_ADOPTION

    return centroids_proj


# ---------------------------------------------------------------------------
# G) Aggregate demand by vertiport
# ---------------------------------------------------------------------------
def compute_uam_by_vertiport(centroids_proj):
    """
    Aggregate UAM demand per vertiport.
    """
    return (
        centroids_proj.groupby("nearest_vertiport")["uam_population"]
        .sum()
        .sort_values(ascending=False)
    )


# ---------------------------------------------------------------------------
# H) Full pipeline wrapper (used by main.py)
# ---------------------------------------------------------------------------
def load_all_data(shapefile_path):
    """
    End-to-end data pipeline.
    """
    verts_gdf = load_vertiports()
    tracts = load_census_tracts(shapefile_path)
    pop = fetch_acs_population()

    tracts_pop = merge_population(tracts, pop)
    tracts_in_rect = clip_to_bbox(tracts_pop)

    centroids_proj = assign_nearest_vertiport(tracts_in_rect, verts_gdf)
    uam_by_vert = compute_uam_by_vertiport(centroids_proj)

    return {
        "vertiports": verts_gdf,
        "tracts": tracts_in_rect,
        "centroids": centroids_proj,
        "uam_by_vertiport": uam_by_vert,
        "vertiports_dict": {row["name"]: (row["lat"], row["lon"]) for _, row in verts_gdf.iterrows()}
    }
