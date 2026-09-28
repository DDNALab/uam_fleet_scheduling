"""Deterministic UAM demand preparation from local Census tract and ACS files.

Data files are located relative to the project root by src.config:
  PROJECT_ROOT/data/tl_2022_48_tract/tl_2022_48_tract.shp
  PROJECT_ROOT/data/acs_texas_2022_raw.json
No network access or data downloads are performed here.
"""

import json
import math

import geopandas as gpd
import numpy as np
import pandas as pd

from .config import (
    ACS_JSON_PATH, SHAPEFILE_PATH, BBOX, VERTIPORTS_RAW, HUB_NAME,
    UAM_ADOPTION, GAMMA, TOTAL_PASSENGER_TRIPS, BIN_SIZE,
    HORIZON_START, HORIZON_END,
)


def load_vertiports():
    df = pd.DataFrame({
        'name': list(VERTIPORTS_RAW),
        'lat': [x[0] for x in VERTIPORTS_RAW.values()],
        'lon': [x[1] for x in VERTIPORTS_RAW.values()],
    })
    return gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.lon, df.lat), crs='EPSG:4326'
    )


def load_census_tracts():
    if not SHAPEFILE_PATH.is_file():
        raise FileNotFoundError(
            f'Census shapefile not found: {SHAPEFILE_PATH}. '
            'Run: python -m src.diagnostics.check_data_paths'
        )
    bbox = (BBOX['min_lon'], BBOX['min_lat'], BBOX['max_lon'], BBOX['max_lat'])
    return gpd.read_file(SHAPEFILE_PATH, bbox=bbox).to_crs(4326)


def load_population():
    if not ACS_JSON_PATH.is_file():
        raise FileNotFoundError(
            f'ACS population file not found: {ACS_JSON_PATH}. '
            'Run: python -m src.diagnostics.check_data_paths'
        )
    with ACS_JSON_PATH.open('r', encoding='utf-8') as f:
        acs = json.load(f)
    df = pd.DataFrame(acs[1:], columns=acs[0])
    df['GEOID'] = df['state'] + df['county'] + df['tract']
    df['population'] = pd.to_numeric(df['B01003_001E'], errors='coerce')
    return df[['GEOID', 'population']]


def merge_population(tracts, population):
    return tracts.merge(population, on='GEOID', how='left').dropna(subset=['population'])


def assign_nearest_vertiport(tracts, vertiports):
    verts = vertiports.to_crs(3857)
    projected = tracts.to_crs(3857).copy()
    projected['centroid'] = projected.geometry.centroid
    projected['nearest_vertiport'] = [
        vertiports.loc[verts.distance(point).idxmin(), 'name']
        for point in projected['centroid']
    ]
    return projected


def calculate_market_potential(tracts):
    tracts = tracts.copy()
    tracts['uam_population'] = tracts['population'] * UAM_ADOPTION
    return tracts.groupby('nearest_vertiport')['uam_population'].sum()


def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2-p1, math.radians(lon2-lon1)
    a = (math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2)
    return 2*radius*math.atan2(math.sqrt(a), math.sqrt(1-a))


def gravity_demand(market, vertiports):
    hub_value = market[HUB_NAME]
    hub = vertiports.loc[vertiports.name == HUB_NAME].iloc[0]
    flows = []
    for destination, potential in market.items():
        if destination == HUB_NAME:
            continue
        dest = vertiports.loc[vertiports.name == destination].iloc[0]
        distance = haversine_km(hub.lat, hub.lon, dest.lat, dest.lon)
        score = hub_value*potential / distance**GAMMA
        flows.append({
            'origin': HUB_NAME, 'destination': destination,
            'distance_km': distance, 'gravity_score': score,
        })
    if not flows:
        raise ValueError('No OD destinations have tract/ACS market potential in BBOX')
    df = pd.DataFrame(flows)
    if df.gravity_score.sum() <= 0:
        raise ValueError('Gravity weights sum to zero: check ACS and tract merge')
    df['expected_trips'] = TOTAL_PASSENGER_TRIPS*df.gravity_score/df.gravity_score.sum()
    return df


def gaussian(x, mu, sigma):
    return np.exp(-0.5*((x-mu)/sigma)**2)


def passenger_time_profile(*, start_minutes=HORIZON_START, end_minutes=None,
                           bin_minutes=BIN_SIZE):
    """Normalized 08:00/16:00 demand profile across ANY configured time horizon.

    Defaults preserve the original four-hour development instance.  An extended
    horizon adds bins instead of stretching the old sixteen-period profile.
    """
    if end_minutes is None:
        end_minutes = HORIZON_END
    if not all(isinstance(x, (int, np.integer)) for x in
               (start_minutes, end_minutes, bin_minutes)):
        raise ValueError("profile times must be integer minutes")
    if bin_minutes <= 0 or end_minutes <= start_minutes or (
            end_minutes - start_minutes) % bin_minutes:
        raise ValueError("horizon must contain a positive whole number of time bins")
    periods = np.arange(start_minutes, end_minutes, bin_minutes)
    morning = gaussian(periods, 8 * 60, 60)
    afternoon = gaussian(periods, 16 * 60, 80)
    profile = 0.05 + morning + afternoon
    return pd.DataFrame({'period': periods, 'weight': profile / profile.sum()})


def build_uam_data():
    vertiports = load_vertiports()
    tracts = load_census_tracts()
    population = load_population()
    tracts = merge_population(tracts, population)
    tracts = assign_nearest_vertiport(tracts, vertiports)
    market = calculate_market_potential(tracts)
    return {
        'vertiports': vertiports, 'tracts': tracts,
        'market': market, 'od_demand': gravity_demand(market, vertiports),
        'time_profile': passenger_time_profile(),
    }


if __name__ == '__main__':
    print(build_uam_data()['od_demand'].to_string(index=False))
