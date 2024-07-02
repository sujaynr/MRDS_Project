import geopandas as gpd
import fiona
import numpy as np
import pdb

gdb_path = '/home/sujaynair/GDB/Qfaults_2020_WGS84.gdb'


layers = fiona.listlayers(gdb_path)
print("Layers in the geodatabase:")
for layer in layers:
    print(layer)


# for layer in layers:
#     gdf = gpd.read_file(gdb_path, layer=layer)
#     print(f"\nContents of layer {layer}:")
#     print(gdf.head())

faults = gpd.read_file(gdb_path, layer='Qfaults_2020')
# print(faults.info())
pdb.set_trace()
print(np.ndim(faults))


'''
Unique Locations:
array(['California', 'Arizona', 'Utah', 'Nevada', 'Oklahoma', 'Kansas',
       'Oregon', 'Idaho', 'New Mexico', 'Colorado', 'Texas', 'Wyoming',
       'Hawaii', 'Alaska', 'Washington', 'WA Offshore', 'Montana'],
      dtype=object)
'''