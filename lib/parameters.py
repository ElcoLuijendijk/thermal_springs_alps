"""
Parameters that are used by more than one notebook

Changing a value here changes it in all notebooks that import this file.
"""

# number of seconds in a year
year = 365.25 * 24 * 3600

# minimum difference (degrees C) between the spring temperature and the minimum surface temperature in the
# watershed of the spring for a spring to be classified as a thermal spring, following White (1957)
temp_cutoff_thermal_springs = 5.0

# encoding of the csv files with the spring data
encoding = 'utf-8-sig'

# shapefile with the outline of the Alps
alps_outline_file = 'GIS_data/outline_alps_final.shp'

# csv file with the results of GIS_analysis_spring_data.ipynb, which is the input of calculate_heat_flux.ipynb
geospatial_data_file = 'data/thermal_springs_alps_with_geospatial_data.csv'
