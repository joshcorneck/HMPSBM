import numpy as np
import pandas as pd

import yaml
import os
import pickle

from src.variational_bayes import VariationalBayes

with open(f"config_FAO.yaml", "r") as file:
    config = yaml.safe_load(file)

array_index = int(os.getenv("PBS_ARRAYID", 0))

np.random.seed(array_index)

print(f"Running script for PBS_ARRAYID: {array_index}")

##~~ 
# STEP 1: LOAD AND PROCESS DATA
##~~
adj_tensor = np.load(f'processed_data/adjacency_tensor.npy')

# Network parameters
num_nodes = int(adj_tensor.shape[1])
num_layers = int(adj_tensor.shape[0])
M_w = int(config['simulations'][array_index]['M_w'])
M_z = int(config['simulations'][array_index]['M_z'])

print(f"...M_w: {M_w}...")
print(f"...M_z: {M_z}")

##~~ 
# STEP 2: RUN VB
##~~
num_CAVI_its = 25
num_mc = 10000
lr_theta = 0.5
lr_sigma = 0.1
max_ELBO_phi_dec = 10
max_num_grad_steps = 30
num_adj_samps = 1

feature_bool = True
intercept_bool = True

if feature_bool:
    features = np.load("data/FAO/processed_data/features.npy")
    if intercept_bool:
        ones = np.ones((features.shape[0], 1))
        features = np.hstack((ones, features))
else:
    features = np.ones((num_nodes, 1))

VB = VariationalBayes(num_nodes=num_nodes, num_layers=num_layers,
                      adj_tensor=adj_tensor, features=features,
                      M_w=M_w, M_z=M_z, num_fp_its=2, num_adj_samps=num_adj_samps,
                      uniform_w=True)
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

if feature_bool:
    save_to_pickle(VB.phi_w_best, f'output_features/phi_w/phi_w_{array_index}.pkl')
    save_to_pickle(VB.phi_z_best, f'output_features/phi_z/phi_z_{array_index}.pkl')
    save_to_pickle(VB.alpha_rho_best, f'output_features/alpha_rho/alpha_rho_{array_index}.pkl')
    save_to_pickle(VB.beta_rho_best, f'output_features/beta_rho/beta_rho_{array_index}.pkl')
    save_to_pickle(VB.theta_phi_best, f'output_features/theta_phi/theta_phi_{array_index}.pkl')
    save_to_pickle(VB.ELBO_store_full, f'output_features/ELBO/ELBO_full_{array_index}.pkl')
else:
    save_to_pickle(VB.phi_w_best, f'output/phi_w/phi_w_{array_index}.pkl')
    save_to_pickle(VB.phi_z_best, f'output/phi_z/phi_z_{array_index}.pkl')
    save_to_pickle(VB.alpha_rho_best, f'output/alpha_rho/alpha_rho_{array_index}.pkl')
    save_to_pickle(VB.beta_rho_best, f'output/beta_rho/beta_rho_{array_index}.pkl')
    save_to_pickle(VB.ELBO_store_full, f'output/ELBO/ELBO_full_{array_index}.pkl')
