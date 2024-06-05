# /Users/sujaynair/anaconda3/envs/dataAnalysis/bin/pip3 install X
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
# df = df.sample(frac=0.1, random_state=1)

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
elem = 'Diamond'
elem_df = df[df['commodities'].str.strip() == elem]

score_amplitude = {'A': 1, 'B': 0.8, 'C': 0.6, 'D': 0.4, 'E': 0.2}

x_min, x_max = -125, -66.5
y_min, y_max = 24.5, 49.5
grid_size = 30
x = np.linspace(x_min, x_max, grid_size + 1)
y = np.linspace(y_min, y_max, grid_size + 1)
xv, yv = np.meshgrid(x, y)
z = np.zeros((grid_size, grid_size))
pixel_size_x = (x_max - x_min) / grid_size
pixel_size_y = (y_max - y_min) / grid_size

z_A = np.zeros((grid_size, grid_size))
z_B = np.zeros((grid_size, grid_size))
z_C = np.zeros((grid_size, grid_size))
z_D = np.zeros((grid_size, grid_size))
z_E = np.zeros((grid_size, grid_size))

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
                ''' SCORING BASED ON WHATS IN THE PIXEL
                amplitude = score_amplitude.get(row['score'])
                z[i, j] += amplitude
                '''
                if row['score'] == 'A':
                    z_A[i, j] += 1
                    z_B[i, j] += 1
                    z_C[i, j] += 1
                    z_D[i, j] += 1
                    z_E[i, j] += 1
                elif row['score'] == 'B':
                    z_B[i, j] += 1
                    z_C[i, j] += 1
                    z_D[i, j] += 1
                    z_E[i, j] += 1
                elif row['score'] == 'C':
                    z_C[i, j] += 1
                    z_D[i, j] += 1
                    z_E[i, j] += 1
                elif row['score'] == 'D':
                    z_D[i, j] += 1
                    z_E[i, j] += 1
                elif row['score'] == 'E':
                    z_E[i, j] += 1

score_layers = {'A': z_A, 'B': z_B, 'C': z_C, 'D': z_D, 'E': z_E}
output_dir = f'LayerPlots/{elem}'
os.makedirs(output_dir, exist_ok=True)
for score, z in score_layers.items():
    z = np.flipud(z)
    norm = Normalize(vmin=0, vmax=np.max(z))
    
    fig, ax = plt.subplots(1, 1, figsize=(20, 15))
    cax = ax.imshow(z, cmap='viridis', extent=[x_min, x_max, y_min, y_max], origin='lower', alpha=0.7, norm=norm)
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap='magma'), ax=ax, orientation='vertical')
    cbar.set_label('Count of Observations')

    north_america.plot(ax=ax, color='none', edgecolor='black')

    ax.set_title(f'Observations of Score {score} in the Continental US ({elem} 100%)', fontsize=20)
    ax.set_xlabel('Longitude', fontsize=15)
    ax.set_ylabel('Latitude', fontsize=15)

    plt.savefig(os.path.join(output_dir, f'Layer_{score}_{elem}100%.png'))
    plt.close()
