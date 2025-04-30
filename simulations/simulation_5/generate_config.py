#%%
import yaml 

num_nodes = [100, 250, 500]
num_samps = [1, 3, 5]

# Create the dictionary with integer keys
config_dict = {i: {"num_nodes": a, "num_samps": b} for i, (a, b) in enumerate([(a, b) for a in num_nodes for b in num_samps], start=0)}

# Write the YAML file
with open("simulations/simulation_4/config_4.yaml", "w") as f:
    yaml.dump(config_dict, f, default_flow_style=False)

print("Config file generated: config.yaml")




# %%
