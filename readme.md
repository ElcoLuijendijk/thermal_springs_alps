# Thermal springs Alps notebooks

This directory contains a series of notebooks that were used to analyse a database of thermal springs in the Alps (Luijendijk et al., 2020, https://doi.org/10.1029/2020GL090134).


## Description

* [GIS_analysis_spring_data.ipynb](GIS_analysis_spring_data.ipynb) parses the spring database to include only natural springs that are inside the Alps. The notebook also calculates some spatial statistics of the springs and the watersheds in which these springs are located using the hydrobasins dataset and several raster files that are located in the directory [GIS_data](GIS_data). The results are saved as a new .csv file, which has the same filename of the original dataset but with ``_with_geospatial_data.csv`` appended to the filename.
* [calculate_heat_flux.ipynb](calculate_heat_flux.ipynb) takes the output from the GIS_analysis notebook, removes non-thermal springs and calculates the contributing area, recharge temperature, circulation temperature and spring heat flux. The results are saved as a new .csv file with ``_with_HF_estimates.csv`` appended to the filename, and a summary of the results is saved to [data/summary_thermal_springs_data.csv](data/summary_thermal_springs_data.csv).
* [make_figures.ipynb](make_figures.ipynb) was used to make all the figures for the manuscript. The figures are saved to the subdirectory fig.
* [map_thermal_springs.ipynb](map_thermal_springs.ipynb) can be used to make a map of the thermal spring discharge and temperatures. The map is also saved to the subdirectory fig. Run make_figures.ipynb first, since this notebook combines the map with one of the figures made by make_figures.ipynb.
* [background_heat_flow_Alps.ipynb](background_heat_flow_Alps.ipynb) was used to analyze the background heat flow in the Alps using data from the global heat flow database.
* [background_heat_flow_NA.ipynb](background_heat_flow_NA.ipynb) was used to make a map of the background heat flow in North America.
* [compare_orogens_v2.ipynb](compare_orogens_v2.ipynb) was used to compare the spring discharge and heat flux of the Alps with published data for springs in North America (Ferguson & Grasby 2011, Geofluids).
* [additional_notebooks/merge_HF_datasets.ipynb](additional_notebooks/merge_HF_datasets.ipynb) was used to merge the SMU and global heat flow databases for North America into [data/heat_flow_NA_merged.csv](data/heat_flow_NA_merged.csv). The input files for this notebook are not included in this repository because of their size.

The notebooks should be run from the main directory of this repository. The subdirectory fig is not included in the repository and needs to be created before running the notebooks that make figures.

The Alps thermal spring database has been published at PANGAEA (https://doi.org/10.1594/PANGAEA.916089). [GIS_analysis_spring_data.ipynb](GIS_analysis_spring_data.ipynb) downloads the compiled spring data (https://doi.org/10.1594/PANGAEA.916086) to the subdirectory data the first time it is run, so an internet connection is needed for the first run. The original compilation, which contains the full references, is located in [data/thermal_springs_alps.csv](data/thermal_springs_alps.csv) but is not used by the notebooks.

The elevation data for North America in [GIS_data/etopo2022_bed_NA_5min.tif](GIS_data/etopo2022_bed_NA_5min.tif) were derived from the ETOPO 2022 global relief model (NOAA National Centers for Environmental Information, https://doi.org/10.25921/fd45-gt74), clipped to North America and resampled to a resolution of 5 arcminutes.


## Dependencies

The notebooks require a number of python packages: numpy, matplotlib, pandas, geopandas, scipy, rasterio, rasterstats, shapely, cartopy, pyproj, pillow, chardet and earthpy.


## Authors
* **Elco Luijendijk**, <elco.luijendijk-at-posteo.de>


## Reference

The notebooks were used to complete the analysis and make figures for the following manuscript: 

Luijendijk, E., Winter, T., Köhler, S., Ferguson, G., von Hagke, C., & Scibek, J. (2020). Using thermal springs to quantify deep groundwater flow and its thermal footprint in the Alps and a comparison with North American orogens. Geophysical Research Letters, 47, e2020GL090134. https://doi.org/10.1029/2020GL090134 

Please cite this publication if you publish any work based on these notebooks or the thermal spring database.


## License
This project is licensed under the GNU lesser general public license (LGPL v3). See the [LICENSE.txt](LICENSE.txt) file for details.

