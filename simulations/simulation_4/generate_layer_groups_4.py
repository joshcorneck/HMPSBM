#%%
import numpy as np
import yaml
import os

array_index = int(os.getenv("PBS_ARRAYID", 0))
 
gamma1 = [1, 0, 0]
gamma2 = [0, 1, 0]
gamma3 = [0, 0, 1]

with open(f"config_4.yaml", "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
    
num_nodes = int(config[int(array_index // 50)]['num_nodes'])

if num_nodes == 500:
    global_group_sizes = [200, 200, 100]
elif num_nodes == 250:
    global_group_sizes = [100, 100, 50]
else:
    global_group_sizes = [40, 40, 20]

num_layers = 10

def sample_layer_groups(gammas, global_group_sizes):
    num_layer_groups = len(gammas[0])
    num_glob_groups = len(global_group_sizes)
    num_nodes = np.array(global_group_sizes).sum()
    layer_groups = np.zeros((num_layers, num_nodes))

    for layer in range(num_layers):
        samples = []
        for global_group in range(num_glob_groups):
            samples.append(
                np.random.choice(num_layer_groups,
                                 p=gammas[global_group],
                                 size=global_group_sizes[global_group])
            )
        layer_groups[layer,:] = (
            np.concatenate(samples)
        )

    return layer_groups

print("Sampling layer groups")
layer_groups = sample_layer_groups(gammas=[gamma1, gamma2, gamma3],     
                                   global_group_sizes=global_group_sizes)
print("Layer groups sampled")
layer_groups = layer_groups.astype(int)

array_index = int(os.getenv("PBS_ARRAYID", 0))
np.save(f"output/layer_groups/layer_groups_{array_index}.npy", layer_groups)
