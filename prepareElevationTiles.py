import geopandas as gpd
import pdb

# Path to the shapefile
shapefile_path = '/home/sujaynair/gtopo30.shp'

# Load the shapefile
data = gpd.read_file(shapefile_path)

# Perform analysis on the loaded shapefile
# Example: Print the number of features in the shapefile
num_features = len(data)
pdb.set_trace()