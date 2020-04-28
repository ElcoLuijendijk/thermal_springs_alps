# Thermal springs Alps notebooks

This directory contains a series of notebooks that were used to analyze a newly compiled database of thermal springs in the Alps for an upcoming manuscript (Luijendijk et al., submitted).


## Description

* [GIS_analysis_spring_data.ipynb](GIS_analysis_spring_data.ipynb) parses the spring database to include only natrual springs that are inside the Alps. The notebook also claculates some spatial statistics o fthe springs and the watersheds in which these springs are located using the hydrobasins dataset and several raster files that are located in the directory [GIS_data](GIS_data). The results are be saved as a new .csv file, which has the same filename of the original dataset but with ``_with_geospatial_data.csv`` appended to the filename
* [analyze_spring_heat_flux_new](analyze_spring_heat_flux_new.ipynb) takes the output from the GIS_analysis notebook and calculates the contributing area, recharge temperature and spring heat flux. The results will be saved as a new .csv file with ``_with_HF_estimates.csv`` appended to the filename
* [make_figures.ipynb](make_figures.ipynb) was used to make all the figures for the manuscript. The figures are saved to the subdirectory [fig](fig).
* [map_thermal_springs.ipynb](map_thermal_springs.ipynb) can be used to make a map of the thermal spring discharge & temperatures. The map is also saved to the subdirectory [fig](fig).
* [analyze_background_heat_flow.ipynb](analyze_background_heat_flow.ipynb) was used to analyze the background heat flow in the Alps using data from the global heat flow database.
* [compare_springs_with_NA.ipynb](compare_springs_with_NA.ipynb) was used to compare the spring discharge and heat flux of the Alps with published data for springs in North America (Ferguson & Grasby 2011, Geofluids).

The alps thermal spring database is located in the subdirectory data: [data/thermal_springs_alps_and_surroundings.ipynb](data/thermal_springs_alps_and_surroundings.ipynb)


## Dependencies

The notebooks requires a number of python packages: numpy, matplotlib, pandas, geopandas, scipy, rasterio, rasterstats, shapely 


## Authors
* **Elco Luijendijk**, <elco.luijendijk-at-posteo.de>


## Reference

The notebooks were used to complete the analysis and make figures for the following manuscript: 

Elco Luijendijk, Theis Winter, Saskia Köhler, Grant Ferguson, Christoph von Hagke, Jacek Scibek. Using thermal springs to quantify deep groundwater flow and its thermal footprint in the Alps and North American orogens. Submitted for review to Geophysical Research Letters.

Please cite this if you use these notebooks


## License
This project is licensed under the GNU lesser general public license (LGPL v3). See the [LICENSE.txt](LICENSE.txt) file for details.

