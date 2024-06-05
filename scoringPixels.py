# SOON TO BE DEPRECIATED
import os
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import multivariate_normal
import fsspec
import pdb
pd.options.mode.copy_on_write = True

def plot_histogram(df, column_name):
    plt.figure(figsize=(10, 6))
    df[column_name].hist(bins=30, edgecolor='black')
    plt.title(f'Histogram of {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.savefig(f'histogram_{column_name}.png')
    plt.show()

file_path = '/Users/sujaynair/Documents/mrds-csv/mrds.csv'
df = pd.read_csv(file_path)
shape = df.shape
colnames = df.columns.tolist()
# print("Shape of the DataFrame:", shape)
# print("Column names:", colnames)

df['region'] = df['region'].fillna('NA')

''' THIS SECTION DOES ONLY FOR ALL A SCORE SAMPLES
dfA = df[df['score'] == 'A']

dfA_NA = dfA[dfA['region'] == 'NA']
values = {"commod1": "", "commod2": "", "commod3": ""}
dfA_NA.loc[:, ['commod1', 'commod2', 'commod3']] = dfA_NA[['commod1', 'commod2', 'commod3']].fillna(value=values)

dfA_NA.loc[:, 'commod1'] = dfA_NA['commod1'].astype(str)
dfA_NA.loc[:, 'commod2'] = dfA_NA['commod2'].astype(str)
dfA_NA.loc[:, 'commod3'] = dfA_NA['commod3'].astype(str)

dfA_NA.loc[:, 'commodities'] = dfA_NA.apply(lambda x: ','.join(filter(None, [x['commod1'], x['commod2'], x['commod3']])), axis=1)
dfA_NA = dfA_NA.assign(commodities=dfA_NA['commodities'].str.split(',')).explode('commodities')
dfA_NA = dfA_NA[dfA_NA['commodities'] != '']

local_map_path = "/Users/sujaynair/Downloads/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp"
gdf = gpd.read_file(local_map_path)

north_america = gdf[gdf['CONTINENT'] == 'North America']

# Filter for Zinc
zinc_df = dfA_NA[dfA_NA['commodities'].str.strip() == 'Zinc']

fig, ax = plt.subplots(1, 1, figsize=(20, 15))

x_min, x_max = -170, -50
y_min, y_max = 0, 80
grid_size = 500
x = np.linspace(x_min, x_max, grid_size)
y = np.linspace(y_min, y_max, grid_size)
xv, yv = np.meshgrid(x, y)

z = np.zeros_like(xv)
for _, row in zinc_df.iterrows():
    mu = [row['longitude'], row['latitude']]
    sigma = [[3, 0], [0, 3]]
    rv = multivariate_normal(mu, sigma)
    z += rv.pdf(np.dstack((xv, yv)))

ax.imshow(np.rot90(z), cmap='magma', extent=[x_min, x_max, y_min, y_max], alpha=0.7)

north_america.plot(ax=ax, color='none', edgecolor='black')

plt.title('Brute-Force Gaussian Splatting of Zinc in North America', fontsize=20)
plt.xlabel('Longitude', fontsize=15)
plt.ylabel('Latitude', fontsize=15)
plt.grid(True)
plt.show()
'''
'''#THIS IS JUST TO VISUALIZE LOCATIONS OF ELEMENTS
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import fsspec
import pdb

pd.options.mode.copy_on_write = True

def plot_histogram(df, column_name):
    plt.figure(figsize=(10, 6))
    df[column_name].hist(bins=30, edgecolor='black')
    plt.title(f'Histogram of {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.savefig(f'histogram_{column_name}.png')
    plt.show()

file_path = '/Users/sujaynair/Documents/mrds-csv/mrds.csv'
df = pd.read_csv(file_path)


df = df.sample(frac=0.1, random_state=1)  # Adjust the fraction as needed

shape = df.shape
colnames = df.columns.tolist()

df['region'] = df['region'].fillna('NA')

values = {"commod1": "", "commod2": "", "commod3": ""}
df.loc[:, ['commod1', 'commod2', 'commod3']] = df[['commod1', 'commod2', 'commod3']].fillna(value=values)

df.loc[:, 'commod1'] = df['commod1'].astype(str)
df.loc[:, 'commod2'] = df['commod2'].astype(str)
df.loc[:, 'commod3'] = df['commod3'].astype(str)

df.loc[:, 'commodities'] = df.apply(lambda x: ','.join(filter(None, [x['commod1'], x['commod2'], x['commod3']])), axis=1)
df = df.assign(commodities=df['commodities'].str.split(',')).explode('commodities')
df = df[df['commodities'] != '']

local_map_path = "/Users/sujaynair/Downloads/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp"
gdf = gpd.read_file(local_map_path)

north_america = gdf[gdf['CONTINENT'] == 'North America']

zinc_df = df[df['commodities'].str.strip() == 'Zinc']

fig, ax = plt.subplots(1, 1, figsize=(20, 15))

north_america.plot(ax=ax, color='lightgrey', edgecolor='black')


ax.scatter(zinc_df['longitude'], zinc_df['latitude'], color='red', s=10, alpha=0.6, label='Zinc Occurrences')

plt.title('Locations of Zinc Occurrences in North America', fontsize=20)
plt.xlabel('Longitude', fontsize=15)
plt.ylabel('Latitude', fontsize=15)
plt.legend()
plt.show()
'''
df = df.sample(frac=0.1, random_state=1)

values = {"commod1": "", "commod2": "", "commod3": ""}
df[['commod1', 'commod2', 'commod3']] = df[['commod1', 'commod2', 'commod3']].fillna(value=values)
df[['commod1', 'commod2', 'commod3']] = df[['commod1', 'commod2', 'commod3']].astype(str)
df['commodities'] = df.apply(lambda x: ','.join(filter(None, [x['commod1'], x['commod2'], x['commod3']])), axis=1)
df = df.assign(commodities=df['commodities'].str.split(',')).explode('commodities')
# print(df['commodities'].explode().unique())
df = df[df['commodities'] != '']

local_map_path = "/Users/sujaynair/Downloads/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp"
gdf = gpd.read_file(local_map_path)
north_america = gdf[gdf['CONTINENT'] == 'North America']
elem = 'Lead'
elem_df = df[df['commodities'].str.strip() == elem]

score_amplitude = {'A': 1, 'B': 0.8, 'C': 0.6, 'D': 0.4, 'E': 0.2}

x_min, x_max = -125, -66.5
y_min, y_max = 24.5, 49.5
grid_size = 20
x = np.linspace(x_min, x_max, grid_size + 1)
y = np.linspace(y_min, y_max, grid_size + 1)
xv, yv = np.meshgrid(x, y)
z = np.zeros((grid_size, grid_size))
pixel_size_x = (x_max - x_min) / grid_size
pixel_size_y = (y_max - y_min) / grid_size

for i in range(grid_size):
    for j in range(grid_size):
        lon_min = x_min + j * pixel_size_x
        lon_max = lon_min + pixel_size_x
        lat_max = y_max - i * pixel_size_y
        lat_min = lat_max - pixel_size_y

        print(f"Pixel [{i}, {j}] boundaries: Longitude ({lon_min}, {lon_max}), Latitude ({lat_min}, {lat_max})")

        for _, row in elem_df.iterrows():
            if lon_min <= row['longitude'] < lon_max and lat_min <= row['latitude'] < lat_max:
                print(f"Yes: ({row['score']}) quality resource at {row['site_name']}, {row['county']}, {row['state']} at ({row['longitude']}, {row['latitude']}) is in pixel [{i}, {j}]")
                amplitude = score_amplitude.get(row['score'])
                z[i, j] += amplitude


z = z / np.max(z)
z = np.flipud(z)

fig, ax = plt.subplots(1, 1, figsize=(20, 15))
norm = Normalize(vmin=0, vmax=1)
cax = ax.imshow(z, cmap='viridis', extent=[x_min, x_max, y_min, y_max], origin='lower', alpha=0.7, norm=norm)
cbar = fig.colorbar(ScalarMappable(norm=norm, cmap='magma'), ax=ax, orientation='vertical')
cbar.set_label('Normalized Intensity')

north_america.plot(ax=ax, color='none', edgecolor='black')

plt.title('Brute-Force Gaussian Splatting of Zinc in the Continental US', fontsize=20)
plt.xlabel('Longitude', fontsize=15)
plt.ylabel('Latitude', fontsize=15)
plt.show()