# Data

This directory contains the fixed inputs used by the workflow.

## `raw/acs_texas_2022_raw.json`

American Community Survey 2022 five-year population estimates for Texas census
tracts. The workflow reads total population (`B01003_001E`) and constructs the
tract GEOID from the state, county, and tract fields.

## `raw/tl_2022_48_tract/`

Texas 2022 TIGER/Line census tract shapefile and its required sidecar files.
Keep the `.shp`, `.shx`, `.dbf`, `.prj`, and `.cpg` files together.

The repository uses these files locally and does not require a Census API key.
Review the relevant U.S. Census Bureau attribution and redistribution guidance
before distributing a public copy.

