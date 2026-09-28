"""Validate new project-relative Census/ACS paths, optionally load the data.

From .../coord_schedule_UAM:
    python -m src.diagnostics.check_data_paths
    python -m src.diagnostics.check_data_paths --load
"""
import argparse
import json
from pathlib import Path

from src import config


def check_paths():
    print('PROJECT_ROOT:    ', config.PROJECT_ROOT)
    print('SOURCE_DIR:      ', config.PROJECT_ROOT / 'src')
    print('DATA_DIR:        ', config.DATA_DIR)
    print('ACS_JSON_PATH:   ', config.ACS_JSON_PATH)
    print('SHAPEFILE_PATH:  ', config.SHAPEFILE_PATH)
    print('RESULTS_DIR:     ', config.RESULTS_DIR)
    missing = []
    if not (config.CODE_DIR / 'src' / 'config.py').is_file():
        missing.append(f'Missing src/config.py under {config.CODE_DIR}')
    if not config.ACS_JSON_PATH.is_file():
        missing.append(f'ACS JSON not found: {config.ACS_JSON_PATH}')
    shp = config.SHAPEFILE_PATH
    for suffix in ('.shp', '.dbf', '.shx'):
        companion = shp.with_suffix(suffix)
        if not companion.is_file():
            missing.append(f'Shapefile component not found: {companion}')
    if missing:
        for msg in missing:
            print('ERROR:', msg)
        folder = config.DATA_DIR / 'tl_2022_48_tract'
        if folder.is_dir():
            print('Available shapefiles:', *[str(p) for p in folder.rglob('*.shp')], sep='\n  ')
        raise SystemExit(1)
    print('PASS: Files exist at the configured paths.')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--load', action='store_true',
                        help='Also read ACS/tract data and build spatial OD demand')
    args = parser.parse_args()
    check_paths()
    if args.load:
        # Import GeoPandas lazily so the path-only test has no GIS dependencies.
        from src.uam_data_pipeline import build_uam_data
        output = build_uam_data()
        print('PASS: Geospatial join and OD-demand pipeline loaded.')
        print(output['od_demand'][['destination', 'expected_trips']].to_string(index=False))
        print('Total OD demand:', round(output['od_demand'].expected_trips.sum(), 4))
        print('Modeled periods:', len(output['time_profile']))


if __name__ == '__main__':
    main()
