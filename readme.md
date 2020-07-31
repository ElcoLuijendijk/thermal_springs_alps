# Thermal springs Alps notebooks

This directory contains a series of notebooks that were used to analyse a newly compiled database of thermal springs in the Alps for an upcoming manuscript (Luijendijk et al., 2020, https://doi.org/10.31223/osf.io/364dj).


## Description

* [GIS_analysis_spring_data.ipynb](GIS_analysis_spring_data.ipynb) parses the spring database to include only natural springs that are inside the Alps. The notebook also calculates some spatial statistics of the springs and the watersheds in which these springs are located using the hydrobasins dataset and several raster files that are located in the directory [GIS_data](GIS_data). The results are be saved as a new .csv file, which has the same filename of the original dataset but with ``_with_geospatial_data.csv`` appended to the filename
* [analyze_spring_heat_flux_new](analyze_spring_heat_flux_new.ipynb) takes the output from the GIS_analysis notebook and calculates the contributing area, recharge temperature, circulation temperature and spring heat flux. The results will be saved as a new .csv file with ``_with_HF_estimates.csv`` appended to the filename
* [make_figures.ipynb](make_figures.ipynb) was used to make all the figures for the manuscript. The figures are saved to the subdirectory [fig](fig).
* [map_thermal_springs.ipynb](map_thermal_springs.ipynb) can be used to make a map of the thermal spring discharge & temperatures. The map is also saved to the subdirectory [fig](fig).
* [analyze_background_heat_flow.ipynb](analyze_background_heat_flow.ipynb) was used to analyze the background heat flow in the Alps using data from the global heat flow database.
* [compare_springs_with_NA.ipynb](compare_springs_with_NA.ipynb) was used to compare the spring discharge and heat flux of the Alps with published data for springs in North America (Ferguson & Grasby 2011, Geofluids).

The alps thermal spring database is located in the subdirectory data: [data/thermal_springs_alps_and_surroundings.csv](data/thermal_springs_alps_and_surroundings.csv)


## Dependencies

The notebooks requires a number of python packages: numpy, matplotlib, pandas, geopandas, scipy, rasterio, rasterstats, shapely 


## Authors
* **Elco Luijendijk**, <elco.luijendijk-at-posteo.de>


## Reference

The notebooks were used to complete the analysis and make figures for the following manuscript: 

Luijendijk, E., Winter, T., Köhler, S., Ferguson, G., von Hagke, C., & Scibek, J. (2020). Using thermal springs to quantify deep groundwater flow and its thermal footprint in the Alps and North American orogens. EarthArxiv. https://doi.org/10.31223/osf.io/364dj

Please cite this publication if you publish any work based on these notebooks or the thermal spring database.


## License
This project is licensed under the GNU lesser general public license (LGPL v3). See the [LICENSE.txt](LICENSE.txt) file for details.

