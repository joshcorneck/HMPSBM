#%%
import pandas as pd
import networkx as nx
import numpy as np
import pickle

# Load the data
df_raw = pd.read_csv("data/FAO/raw_data/fao_trade_multiplex.txt", 
                     header=None, names=['Layer', 'Out_Node', 'In_Node', 'Weight'],
                     sep='\s+')  

# Get the edge count and density of each layer 
num_layers = 364
num_nodes = 214
layer_density = np.zeros((num_layers, ))
edge_count = np.zeros((num_layers, ))
for i, layer in enumerate(range(1, num_layers + 1)):
    df_layer = df_raw[df_raw['Layer'] == layer][['Out_Node', 'In_Node']]
    
    G = nx.DiGraph()
    G.add_nodes_from(range(1, num_nodes + 1))
    G.add_edges_from(df_layer.itertuples(index=False, name=None))

    adj_matrix = nx.to_numpy_array(G, nodelist=range(1, num_nodes + 1))
    layer_density[i] = adj_matrix.mean()
    edge_count[i] = adj_matrix.sum()
    
decreasing_edge_count = np.argsort(edge_count)[::-1]

# Select the top 20 layers
df_top_20 = df_raw[df_raw['Layer'].isin(decreasing_edge_count[:20] + 1)] # +1 because layers are 1-indexed

node_out_counts = df_top_20.groupby(['Out_Node']).size().reset_index(name='Count').rename(columns={'Out_Node': 'Node'})
node_in_counts = df_top_20.groupby(['In_Node']).size().reset_index(name='Count').rename(columns={'In_Node': 'Node'})

# Get nodes missing from node_in_counts (count of 0)
all_nodes = set(range(1, num_nodes + 1))
existing_in_nodes = set(node_in_counts['Node'])
missing_in_nodes = all_nodes - existing_in_nodes

# Get adjust node_in_counts to include with missing nodes with count of 0
missing_in_node_df = pd.DataFrame(list(missing_in_nodes), columns=['Node'])
missing_in_node_df['Count'] = 0
node_in_counts = pd.concat([node_in_counts, missing_in_node_df], ignore_index=True)
node_in_counts = node_in_counts.sort_values(by='Node').reset_index(drop=True)

total_counts = pd.DataFrame({'Node' : range(1, num_nodes + 1)})
total_counts['Count'] = node_in_counts['Count'] + node_out_counts['Count']

# Get nodes that have a total number of connections across the layers greater than 20
node_list = np.array(total_counts[total_counts['Count'] > 20].Node)
with open('data/FAO/processed_data/network_node_list.pkl', 'wb') as handle:
    pickle.dump(node_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

layer_list = df_top_20.Layer.unique()

num_nodes = len(node_list)
num_layers = len(layer_list)

# Create maps from raw enumeration to network enumeration
raw_to_network_nodes = {int(r) : k for k, r in enumerate(node_list)}
raw_to_network_layers = {int(r) : k for k, r in enumerate(layer_list)}
network_to_raw_nodes = {k : int(r) for k, r in enumerate(node_list)}
network_to_raw_layers = {k : int(r) for k, r in enumerate(layer_list)}

# Restrict to only the nodes in node_list
df_top_20 = df_top_20[(df_top_20['In_Node'].isin(node_list)) & (df_top_20['Out_Node'].isin(node_list))]
df_top_20['In_Node'] = df_top_20['In_Node'].map(raw_to_network_nodes)
df_top_20['Out_Node'] = df_top_20['Out_Node'].map(raw_to_network_nodes)
df_top_20['Layer'] = df_top_20['Layer'].map(raw_to_network_layers)

# Construct adjacency matrices
adj_tensor = []
for i, layer in enumerate(range(num_layers)):
    df_layer = df_top_20[df_top_20['Layer'] == layer][['Out_Node', 'In_Node']]
    
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(df_layer.itertuples(index=False, name=None))

    adj_matrix = nx.to_numpy_array(G, nodelist=range(num_nodes))
    adj_tensor.append(adj_matrix)
    
adj_tensor = np.stack(adj_tensor, axis=0)

np.save('data/FAO/processed_data/adjacency_tensor.npy', adj_tensor)

with open('data/FAO/processed_data/network_to_raw_nodes.pkl', 'wb') as handle:
    pickle.dump(network_to_raw_nodes, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('data/FAO/processed_data/raw_to_network_nodes.pkl', 'wb') as handle:
    pickle.dump(raw_to_network_nodes, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('data/FAO/processed_data/network_to_raw_layers.pkl', 'wb') as handle:
    pickle.dump(network_to_raw_layers, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('data/FAO/processed_data/raw_to_network_layers.pkl', 'wb') as handle:
    pickle.dump(raw_to_network_layers, handle, protocol=pickle.HIGHEST_PROTOCOL)