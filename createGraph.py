import pdb
import h5py
import pandas as pd
import networkx as nx
import os
import pickle

# Load data from pickle files
file1 = "/home/sujaynair/MRDS_Project/RaCA_DATA/ICLRDataset_RaCAFullDataset_AA_v1.pkl"
file2 = "/home/sujaynair/MRDS_Project/RaCA_DATA/ICLRDataset_RaCAFullDatasetAux_AA_v1.pkl"
mineral_file = '/home/sujaynair/MRDS_Project/prepared_data_TILES/mineralDataWithCoords.h5'
output_file = '/home/sujaynair/MRDS_Project/prepared_data_TILES/all_graphs.pkl'

full = pd.read_pickle(file1)
aux = pd.read_pickle(file2)

pdb.set_trace()

# Load coordinates from HDF5 file
with h5py.File(mineral_file, 'r') as f:
    coords = f['coordinates'][:]

# Function to find sites within the n-th square and create a complete graph
def find_sites_and_create_graph(coords, sites, n):
    """
    Finds all sites within the n-th square and creates a complete graph of those sites.
    
    Parameters:
    coords (np.array): An array of shape (10k, 2) with lat/long of the bottom-left points of the squares.
    sites (pd.DataFrame): A DataFrame containing the site data with columns including 'rcasiteid_y', 'lat', and 'long'.
    n (int): The index of the square to check.
    
    Returns:
    networkx.Graph: A complete graph of sites within the n-th square.
    tuple: The coordinates of the bottom-left and top-right corners of the square.
    """
    # Define the size of the square in degrees (approximation, as 1 degree latitude ~ 69 miles)
    square_size_in_degrees = 50 / 69.0
    
    # Get the bottom-left corner of the n-th square
    bottom_left_lat = coords[n, 0]
    bottom_left_long = coords[n, 1]
    
    # Calculate the top-right corner of the square
    top_right_lat = bottom_left_lat + square_size_in_degrees
    top_right_long = bottom_left_long + square_size_in_degrees
    
    # Find sites within the bounds of the square
    sites_in_square = sites[
        (sites['lat'] >= bottom_left_lat) &
        (sites['lat'] <= top_right_lat) &
        (sites['long'] >= bottom_left_long) &
        (sites['long'] <= top_right_long)
    ]
    
    # Add row and column information, considering [0,0] as the top-left corner
    sites_in_square['row'] = ((top_right_lat - sites_in_square['lat']) // (square_size_in_degrees / 50)).astype(int)
    sites_in_square['col'] = ((sites_in_square['long'] - bottom_left_long) // (square_size_in_degrees / 50)).astype(int)
    
    # Columns to exclude from node attributes
    columns_to_exclude = {'rcapid', 'rcasiteid_x', 'sample_id', 'lat', 'long', 'region', 'landuse', 'group', 'rcasiteid_y', 'hzn_desgn', 'texture', 'soc'}
    
    # Print the sites in the square along with their scan data
    print(f"Sites in square {n}:")
    for site_id in sites_in_square['rcasiteid_y'].unique():
        site_data = sites_in_square[sites_in_square['rcasiteid_y'] == site_id].iloc[0]
        print(site_id, site_data[['lat', 'long', 'row', 'col']].to_dict(), site_data.to_dict())
    
    # Create a complete graph
    G = nx.Graph()
    
    # Add nodes with attributes, excluding specified columns
    for site_id in sites_in_square['rcasiteid_y'].unique():
        site_data = sites_in_square[sites_in_square['rcasiteid_y'] == site_id].iloc[0]
        node_attributes = {k: v for k, v in site_data.to_dict().items() if k not in columns_to_exclude}
        G.add_node(site_id, **node_attributes)
    
    # Connect all nodes to form a complete graph
    for node1 in G.nodes():
        for node2 in G.nodes():
            if node1 != node2:
                G.add_edge(node1, node2)
    
    return G, (bottom_left_lat, bottom_left_long, top_right_lat, top_right_long)

# Create a dictionary to store all graphs
all_graphs = {}

# Iterate through the first 10 squares and save the graphs
for n in range(10):
    if {'lat', 'long', 'rcasiteid_y'}.issubset(full.columns):
        graph, square_coords = find_sites_and_create_graph(coords, full, n)
        all_graphs[n] = graph
    else:
        print("Required columns are not available in the dataset.")
        break

# Serialize the dictionary of graphs to a binary file using pickle
with open(output_file, 'wb') as f:
    pickle.dump(all_graphs, f)




# To load the graphs from the file:
# with open(output_file, 'rb') as f:
#     loaded_graphs = pickle.load(f)
# Now you can access each graph by its key, e.g., loaded_graphs[0] for the first graph.

# units? why complete? why not 4x? samples?