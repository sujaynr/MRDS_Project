import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
import pickle
import numpy as np
import matplotlib.pyplot as plt

class SimpleGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.bn3 = nn.BatchNorm1d(output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        print(f'Input x shape: {x.shape}')
        print(f'Edge index shape: {edge_index.shape}')
        
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        
        x = global_mean_pool(x, batch)
        
        return x

def load_graphs(pickle_file_path):
    with open(pickle_file_path, 'rb') as f:
        loaded_graphs = pickle.load(f)
    return loaded_graphs

def create_graph_data(graph):
    features = []
    edges = []
    rows = []
    cols = []

    node_to_idx = {node: idx for idx, node in enumerate(graph.nodes())}

    for node, data in graph.nodes(data=True):
        features.append([data[str(i)] for i in range(365, 2501)] +
                        [data['TOP'], data['BOT'], data['fragvolc'], data['Bulkdensity'],
                         data['adod'], data['c_tot_ncs'], data['caco3'],
                         data['n_tot_ncs'], data['s_tot_ncs']])
        rows.append(data['row'])
        cols.append(data['col'])

    for edge in graph.edges():
        edges.append([node_to_idx[edge[0]], node_to_idx[edge[1]]])

    if not features or not edges:
        return None

    edge_index = np.array(edges).T

    x = torch.tensor(features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)
    data.row = torch.tensor(rows, dtype=torch.long)
    data.col = torch.tensor(cols, dtype=torch.long)
    data.batch = torch.zeros(x.size(0), dtype=torch.long)  # Set batch to zeros for single graph
    return data

def aggregate_to_grid(data_list, output_dim):
    grid = torch.zeros((50, 50, output_dim))
    for data in data_list:
        print(f'Data shape: x={data.x.shape}, row={data.row.shape}, col={data.col.shape}')
        for node in range(len(data.x)):
            row, col = data.row[node], data.col[node]
            if row >= 50 or col >= 50:
                print(f"Index out of bounds: row={row}, col={col}")
                continue
            print(f"Adding node {node} with features {data.x[node]} to grid position ({row}, {col})")
            grid[row][col] += data.x[node]
    grid = grid.sum(dim=-1, keepdim=True)
    return grid

def main(num_runs):
    pickle_file_path = '/home/sujaynair/MRDS_Project/prepared_data_TILES/all_graphs.pkl'
    input_dim = 2145
    hidden_dim = 128
    output_dim = 64

    all_graphs = load_graphs(pickle_file_path)

    graphs = [create_graph_data(graph) for graph in all_graphs.values()]
    graphs = [graph for graph in graphs if graph is not None]
    loader = DataLoader(graphs, batch_size=1, shuffle=True)

    grids = []

    for run in range(num_runs):
        model = SimpleGNN(input_dim, hidden_dim, output_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        def encode():
            model.eval()
            encoded_graphs = []
            with torch.no_grad():
                for data in loader:
                    out = model(data)
                    data.x = out
                    encoded_graphs.append(data)
            return encoded_graphs

        encoded_graphs = encode()

        grid = aggregate_to_grid(encoded_graphs, output_dim)
        grids.append(grid)

        plt.figure()
        plt.imshow(grid.squeeze().numpy())
        plt.colorbar()
        plt.title(f'Grid Run {run+1}')
        plt.savefig(f'/home/sujaynair/MRDS_Project/prepared_data_TILES/grid_run_{run+1}.png')
        plt.close()

    return grids

if __name__ == "__main__":
    num_runs = 3
    grids = main(num_runs)
    for i, grid in enumerate(grids):
        print(f'Final grid shape for run {i+1}: {grid.shape}')
    pdb.set_trace()
