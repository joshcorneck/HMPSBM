import numpy as np

import yaml
import os
import random
import pickle

from sklearn.preprocessing import StandardScaler

from src.network_simulator import MultiplexSimulator
from src.variational_bayes import VariationalBayes

with open(f"config_5.yaml", "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

array_index = int(os.getenv("PBS_ARRAYID", 0))

# Set seed
random.seed(array_index)
np.random.seed(array_index)

print(f"...Running script for PBS_ARRAYID: {array_index}...")

##~~ 
# STEP 1: SIMUALTE A NETWORK
##~~
# Unchanged
num_nodes = int(config[int(array_index // 50)]['num_nodes'])
num_layers = int(config[int(array_index // 50)]['num_layers'])
num_global_groups = 3
max_num_layer_groups = 3
features_length = 3

if num_nodes == 500:
    global_group_sizes = [200, 200, 100]
elif num_nodes == 250:
    global_group_sizes = [100, 100, 50]
else:
    global_group_sizes = [40, 40, 20]
    
print(f"Number of nodes: {num_nodes}")
print(f"Global group sizes: {global_group_sizes}")

rho_matrix = np.array(
    [[0.9, 0.5, 0.2],
     [0.4, 0.7, 0.05],
     [0.2, 0.01, 0.6]]
)
num_CAVI_its = 10
num_mc = 10000
lr_theta = 0.5
lr_sigma = 0.1
max_ELBO_phi_dec = 10
max_num_grad_steps = 30
num_adj_samps = 1
M_w = 3
M_z = 3

print(f"Number of samples: {num_adj_samps}")

features = np.zeros((num_nodes, features_length))
means = [
    [5, 5, 5],   # Mean for group 1
    [0, 0, 0],
    [-5, -5, -5] # Mean for group 2
]
covariances = [
    [[1, 0, 0], 
     [0, 1, 0], 
     [0, 0, 1]], # Covariance for group 1
    [[1, 0, 0], 
     [0, 1, 0], 
     [0, 0, 1]] # Covariance for group 2
]
# Generate features for each group
start_idx = 0
for i, (mean, cov) in enumerate(zip(means, covariances)):
    end_idx = start_idx + global_group_sizes[i]
    features[start_idx:end_idx, :] = (
        np.random.multivariate_normal(mean, cov, size=global_group_sizes[i])
        )
    start_idx = end_idx
    
# 0 mean, 1 variance scaling
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Layer groups
layer_groups = np.load(f"output/layer_groups/layer_groups_{array_index}.npy")

# Simulate a network
MS = MultiplexSimulator(num_nodes=num_nodes, num_layers=num_layers, 
                        num_glob_groups=num_global_groups,
                        max_num_layer_groups=max_num_layer_groups,
                        features=features, specify=True, 
                        rho_matrix=rho_matrix, layer_groups=layer_groups,
                        num_adj_samps=num_adj_samps)
MS.sample_network()

print("...Network simulated...")

##~~ 
# STEP 2: RUN VB
##~~
VB = VariationalBayes(num_nodes=num_nodes, num_layers=num_layers,
                      adj_tensor=MS.adjacency_tensor, features=features,
                      M_w=M_w, M_z=M_z, num_fp_its=3, num_adj_samps=num_adj_samps)
VB.run_VB_scheme(n_CAVI_its=num_CAVI_its, num_mc=num_mc, 
                 max_num_grad_steps=max_num_grad_steps, 
                 max_ELBO_phi_dec=max_ELBO_phi_dec, 
                 lr_theta=lr_theta, lr_sigma=lr_sigma)

###
# STEP 4 - SAVE NETWORK AND INFERRED PARAMETERS
###
def save_to_pickle(obj, file_name, base_path=None):
	if base_path is None:
		with open(f'{file_name}', 'wb') as f:
			pickle.dump(obj, f)
	else:
		with open(f'{base_path}/{file_name}', 'wb') as f:
    			pickle.dump(obj, f)

save_to_pickle(VB.phi_w_best, f'output/phi_w/phi_w_{array_index}.pkl')
save_to_pickle(VB.phi_z_best, f'output/phi_z/phi_z_{array_index}.pkl')
save_to_pickle(VB.alpha_rho_best, f'output/alpha_rho/alpha_rho_{array_index}.pkl')
save_to_pickle(VB.beta_rho_best, f'output/beta_rho/beta_rho_{array_index}.pkl')

save_to_pickle(VB.ELBO_store_full, f'output/ELBO/ELBO_full_{array_index}.pkl')