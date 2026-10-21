# Thermal springs Alps notebooks

[![Run notebooks](https://github.com/ElcoLuijendijk/thermal_springs_alps/actions/workflows/run_notebooks.yml/badge.svg)](https://github.com/ElcoLuijendijk/thermal_springs_alps/actions/workflows/run_notebooks.yml)

This directory contains a series of notebooks that were used to analyse a database of thermal springs in the Alps (Luijendijk et al., 2020, https://doi.org/10.1029/2020GL090134).


## Description

* [GIS_analysis_spring_data.ipynb](GIS_analysis_spring_data.ipynb) parses the spring database to include only natural springs that are inside the Alps. The notebook also calculates some spatial statistics of the springs and the watersheds in which these springs are located using the hydrobasins dataset and several raster files that are located in the directory [GIS_data](GIS_data). It also calculates the distance of each spring to the nearest major fault and the uncertainty of the contributing area of each spring. The results are saved as [data/thermal_springs_alps_with_geospatial_data.csv](data/thermal_springs_alps_with_geospatial_data.csv). The same analysis is carried out for 5000 random points in the Alps, and the springs are compared with the random points using statistical tests ([data/comparison_springs_and_random_points.csv](data/comparison_springs_and_random_points.csv)), including the elevation of the major faults ([data/elevation_of_faults.csv](data/elevation_of_faults.csv)) and a comparison of the distance to faults with random points at the same elevation and position in the valley ([data/fault_distance_elevation_matched.csv](data/fault_distance_elevation_matched.csv)). Finally, the notebook tests how sensitive the results are to the size of the watersheds ([data/sensitivity_watershed_level.csv](data/sensitivity_watershed_level.csv)).
* [calculate_heat_flux.ipynb](calculate_heat_flux.ipynb) takes the output from the GIS_analysis notebook, removes non-thermal springs and calculates the recharge temperature, circulation temperature, spring heat flux and thermal footprint. The uncertainty of the net heat flux is estimated using a Monte Carlo analysis. The results are saved as a new .csv file with ``_with_HF_estimates.csv`` appended to the filename, and a summary of the results is saved to [data/summary_thermal_springs_data.csv](data/summary_thermal_springs_data.csv).
* [make_figures.ipynb](make_figures.ipynb) was used to make all the figures for the manuscript. The figures are saved to the subdirectory fig.
* [map_thermal_springs.ipynb](map_thermal_springs.ipynb) can be used to make a map of the thermal spring discharge and temperatures. The map is also saved to the subdirectory fig. Run make_figures.ipynb first, since this notebook combines the map with one of the figures made by make_figures.ipynb.
* [background_heat_flow_Alps.ipynb](background_heat_flow_Alps.ipynb) was used to analyze the background heat flow in the Alps using data from the global heat flow database.
* [background_heat_flow_NA.ipynb](background_heat_flow_NA.ipynb) was used to make a map of the background heat flow in North America.
* [compare_orogens_v2.ipynb](compare_orogens_v2.ipynb) was used to compare the spring discharge and heat flux of the Alps with published data for springs in North America (Ferguson & Grasby 2011, Geofluids).
* [additional_notebooks/merge_HF_datasets.ipynb](additional_notebooks/merge_HF_datasets.ipynb) was used to merge the SMU and global heat flow databases for North America into [data/heat_flow_NA_merged.csv](data/heat_flow_NA_merged.csv). The input files for this notebook are not included in this repository because of their size.

The notebooks should be run from the main directory of this repository, in the order listed above. Parameters that are used by more than one notebook are defined in [lib/parameters.py](lib/parameters.py). The figures are saved in the subdirectory fig, which is not included in the repository and needs to be created before running the notebooks that make figures. The script [run_all.py](run_all.py) creates this directory and runs all notebooks in the correct order, see below.

The Alps thermal spring database has been published at PANGAEA (https://doi.org/10.1594/PANGAEA.916089). [GIS_analysis_spring_data.ipynb](GIS_analysis_spring_data.ipynb) downloads the compiled spring data (https://doi.org/10.1594/PANGAEA.916086) to the subdirectory data the first time it is run, so an internet connection is needed for the first run. The original compilation, which contains the full references, is located in [data/thermal_springs_alps.csv](data/thermal_springs_alps.csv) but is not used by the notebooks.

The elevation data for North America in [GIS_data/etopo2022_bed_NA_5min.tif](GIS_data/etopo2022_bed_NA_5min.tif) were derived from the ETOPO 2022 global relief model (NOAA National Centers for Environmental Information, https://doi.org/10.25921/fd45-gt74), clipped to North America and resampled to a resolution of 5 arcminutes.


## Dependencies

The notebooks require a number of python packages: numpy, pandas, scipy, matplotlib, geopandas, shapely, pyproj, rasterio, rasterstats, cartopy and pillow. The notebooks have been tested with Python 3.13 and 3.15, numpy 2.2 and 2.5, pandas 2.2 and 3.0, matplotlib 3.10 and 3.11, geopandas 1.1 and 1.2, shapely 2.1 and 2.2, rasterio 1.4 and 1.5, and cartopy 0.25 and 0.26.

The easiest way to install all dependencies is to use conda (or mamba) with the environment file [environment.yml](environment.yml), which installs the packages from the conda-forge channel. Create the environment by running the following command from the main directory of this repository:

```
conda env create -f environment.yml
```

Activate the environment and start Jupyter Lab:

```
conda activate thermal_springs_alps
jupyter lab
```

To run all notebooks in the correct order without opening them in Jupyter, use:

```
python run_all.py
```

The executed notebooks are saved in the directory ``executed_notebooks``. Use ``python run_all.py --inplace`` to save the results in the notebooks themselves, or add the names of notebooks to run only these notebooks, for example ``python run_all.py calculate_heat_flux make_figures``.

All notebooks are also run automatically on GitHub after each change to the repository, see the badge at the top of this page. The figures and data produced by the latest run can be downloaded from the [GitHub actions page](https://github.com/ElcoLuijendijk/thermal_springs_alps/actions/workflows/run_notebooks.yml).

If you have updated the environment file, you can update an existing environment using ``conda env update -f environment.yml --prune``. To remove the environment, use ``conda env remove -n thermal_springs_alps``.


## Version history

The version of the notebooks that was used for the published paper (Luijendijk et al. 2020) is available as release [v2.0](https://github.com/ElcoLuijendijk/thermal_springs_alps/releases/tag/v2.0). Version 3.0 contains the following changes:

**Changes that affect the results:**

* The spring data are now downloaded from PANGAEA instead of read from a local csv file. The PANGAEA dataset contains slightly rounded flow rates for a few springs, which changes the heat flux of five springs by less than 0.1 percent.
* One data point that is a gallery and borehole (spring 353) is no longer included as a thermal spring. The number of thermal springs is 310 instead of 311 and the median temperature 22.65 instead of 22.80 °C.
* Raster cells without data are now excluded from the watershed statistics. This corrects the minimum surface temperature and the minimum recharge temperature of one spring (spring 222).
* The contributing area is now also calculated for springs with only a minimum and maximum flow rate (12 thermal springs).
* The total heat flux of the thermal springs is the same as in version 2.0: 111 (84 to 146) MW.

**New analyses:**

* The range of the contributing area, using the highest and lowest groundwater recharge in the watershed of each spring.
* The distance of each spring to the nearest major fault.
* A comparison of the springs with 5000 random points in the Alps using statistical tests. The springs are located closer to major faults than random points (median distance 9.0 km instead of 14.5 km). Major faults in the Alps tend to follow valleys and are located at lower elevations than the average location in the Alps, like the springs. However, the springs are also closer to faults than random points at the same elevation and the same height above the lowest point of the watershed (median distance 14.1 km).
* A Monte Carlo analysis of the uncertainty of the net heat flux. The total net heat flux of the thermal springs is 109 (103 to 114) MW (median and 5th to 95th percentile) if the errors of the springs are independent, and 108 (96 to 122) MW if the errors are fully correlated. This range is narrower than the minimum and maximum estimate (84 to 146 MW), because the minimum and maximum estimate combine a high recharge temperature with a high recharge elevation, which is physically not consistent.
* A test of the sensitivity of the results to the size of the watersheds.

**Other changes:**

* The code works with current versions of numpy, pandas, matplotlib, geopandas and rasterio, and pyGISlib was replaced by rasterio. A conda environment file was added.
* Two bugs in the maps were fixed: the shaded relief was not displayed correctly and the spring markers were not visible.
* The elevation data for North America (ETOPO1) were replaced by ETOPO 2022, and the missing raster with background heat flow in North America was added.
* GIS_analysis_spring_data.ipynb and calculate_heat_flux.ipynb were restructured and documented to make them easier to use.
* The script run_all.py runs all notebooks, and all notebooks are run automatically on GitHub after each change.


## Authors
* **Elco Luijendijk**, <elco.luijendijk-at-posteo.de>


## Reference

The notebooks were used to complete the analysis and make figures for the following manuscript: 

Luijendijk, E., Winter, T., Köhler, S., Ferguson, G., von Hagke, C., & Scibek, J. (2020). Using thermal springs to quantify deep groundwater flow and its thermal footprint in the Alps and a comparison with North American orogens. Geophysical Research Letters, 47, e2020GL090134. https://doi.org/10.1029/2020GL090134 

Please cite this publication if you publish any work based on these notebooks or the thermal spring database.


## License
This project is licensed under the GNU lesser general public license (LGPL v3). See the [LICENSE.txt](LICENSE.txt) file for details.

