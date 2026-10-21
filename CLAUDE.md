# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Jupyter notebooks and a thermal spring database supporting Luijendijk et al. (2020, GRL, https://doi.org/10.1029/2020GL090134): using thermal springs to quantify deep groundwater flow and its thermal footprint in the Alps, with a comparison to North American orogens. There is no build system, test suite or package configuration. Work is done by running notebooks from the repository root, since all file paths are relative to it (except `additional_notebooks/`, which uses `../` paths).

Dependencies are listed in `environment.yml` (conda-forge): numpy, pandas, scipy, matplotlib, geopandas, shapely, pyproj, rasterio, rasterstats, cartopy, pillow. Raster IO uses rasterio only; GDAL Python bindings (`osgeo`), earthpy and chardet are not needed. The code targets current versions (tested with numpy 2.2 and 2.5, pandas 2.2 and 3.0, matplotlib 3.10 and 3.11, geopandas 1.1 and 1.2); avoid APIs removed in these versions, such as `np.int`, `matplotlib.cm.get_cmap`, `mpl_toolkits.axes_grid`, `{'init': 'epsg:...'}` CRS dicts and chained assignment like `df.loc[i]['col'] = value`. Under pandas 3, arrays from `Series.values` are read-only, so copy them before modifying in place.

Run a notebook headless with:

```
jupyter nbconvert --to notebook --execute --inplace <notebook>.ipynb
```

Figures are written to `fig/`, which is not tracked in git and must exist before running the figure notebooks. Running notebooks overwrites tracked files in `data/`; to check a change for regressions, run the notebook and compare the regenerated CSV against `git show HEAD:<file>`, then restore with `git checkout -- data`. The random points in `data/random_points_in_alps.*` are resampled on every run and will always differ.

## Processing pipeline

The notebooks form a chain where each step writes a CSV consumed by the next. Run in this order after changing upstream data or criteria:

1. `GIS_analysis_spring_data.ipynb`: downloads the spring data from PANGAEA (dataset 916086, tab-delimited text with a metadata header ending in `*/`) to `data/PANGAEA_916086_thermal_springs_alps.tab` (gitignored) if not present, renames the PANGAEA columns to the short names used by all notebooks (`long`, `lat`, `flow_rate_mean`, `temp_mean`, ...) and converts EC from mS/m to S/m. It then keeps natural springs inside `GIS_data/outline_alps_final.shp`, and attaches watershed (HydroBASINS `eu_pfaf_gamma.shp`) and raster statistics (DEM, recharge, surface temperature, lithology, water table) using `rasterstats.zonal_stats`. Writes `data/thermal_springs_alps_with_geospatial_data.csv` (+ `.shp`) and the same for `data/random_points_in_alps` (used for data bias tests). The notebook is written for students: all paths and thresholds are in one parameter cell (`raster_datasets` lists each raster with its column name and whether to compute watershed statistics), the steps use three documented helpers (`sample_raster`, `find_watersheds` via `gp.sjoin`, `calculate_watershed_stats` via `rasterstats.zonal_stats`), and springs and random points are processed together in the `datasets` dict. `sample_raster` deliberately uses the nearest cell centre with numpy rounding (not `rasterio.sample`), because several springs lie exactly on cell boundaries and the two methods pick different cells.
2. `calculate_heat_flux.ipynb`: filters out non-thermal springs (spring temperature must exceed the minimum watershed surface temperature by `temp_cutoff_thermal_springs`, following White 1957; rejects go to `data/non_thermal_springs.csv`), then computes contributing area (discharge / recharge), recharge and circulation temperatures (including SiO2 geothermometers from `lib/various_functions.py`), and spring heat flux. Writes `data/thermal_springs_alps_with_HF_estimates.csv`, `..._clean.csv`, and `data/summary_thermal_springs_data.csv`.
3. Downstream consumers of the step 2 outputs:
   * `make_figures.ipynb`: manuscript figures.
   * `map_thermal_springs.ipynb`: maps of spring temperature, discharge and heat flux, combined with histograms via PIL. Needs `fig/contributing_area_HF_and_thermal_footprint.png` from `make_figures.ipynb`, so run that first.
   * `compare_orogens_v2.ipynb`: compares Alps summary with North American springs (`data/springs_database_North_America_Grant_Ferguson_mod.csv`, Ferguson & Grasby 2011); writes `data/comparison_heat_flux_orogens.csv`.
4. Independent background heat flow analyses: `background_heat_flow_Alps.ipynb` (from `GIS_data/global_hf_eu_v3.csv`, writes `data/background_heat_flow_alps.csv`) and `background_heat_flow_NA.ipynb` (from `data/heat_flow_NA_merged.csv`, produced by `additional_notebooks/merge_HF_datasets.ipynb`, whose own inputs are not in the repository).

The North America notebooks use `GIS_data/etopo2022_bed_NA_5min.tif` (ETOPO 2022, clipped to -170/-50/20/75 and resampled to 5 arcmin) for hillshading and `GIS_data/heat_flow_NA_nat_neighbour.tif` (interpolated background heat flow, no CRS set, used for zonal statistics per orogen).

The spring database is published on PANGAEA (https://doi.org/10.1594/PANGAEA.916089, child datasets 916086 for the compiled spring data and 916088 for the heat flux estimates). `data/thermal_springs_alps.csv` is the original compilation with full references; it is no longer read by any notebook. It differs from the PANGAEA version in a few rounded flow rates (up to 0.08 percent), a few decimal-comma values, and the absence on PANGAEA of the DOI, link and helium columns.

## Shared code in `lib/`

* `mapping_functions.py`: hillshading (`set_shade`, `hillshade`, and `plot_shaded_dem`, which draws the shaded rgb image with `imshow`), the cartopy map layout `make_map_figure`, and image concatenation/cropping helpers used by the map notebooks. Hollow colour-mapped scatter markers need explicit edge colours (`set_edgecolor(cmap(norm(z)))`); `set_facecolor('none')` alone makes them invisible.
* `various_functions.py`: SiO2 geothermometers.

Rasters are read with rasterio in the notebooks. `GIS_analysis_spring_data.ipynb` defines its own `read_raster_file` that returns `(array, dimensions, origin, cellsize, nodata, crs)`, where origin is the upper left corner and the y cellsize is negative; downstream cells index rasters using these values.

## Known caveats

* `calculate_heat_flux.ipynb` detects the encoding of its input CSV with `chardet`; the spring data and outputs of `GIS_analysis_spring_data.ipynb` are UTF-8 with BOM (`utf-8-sig`).
* `google_earth_files_springs/` contains KML exports of the spring locations.
