# HMPSBM

This repository contains the supporting Python code for the paper "*Simultaneous global and local clustering in
multiplex networks with covariate information*" by Joshua Corneck, Ed Cohen, James Martin, Lekha Patel, Kurtis Shuler and Francesco Sanna Passino. This repository contains the following directories:

- `analyses` contains code for reproducing the simulations studies in the paper and for plotting the resulting output.
- `data` contains scripts to process the FAO data used in this paper, code to process the data, and scripts for inference on the network and for plotting the resulting output.
- `src` contains the code for simulating from the HMPSBM and for the inference procedure.

The file `requirements.txt` contains all relevant packages that need to be installed, and these can be installed by running

```sh
pip install -r requirements.txt
```

## General use of the code

The file `example.ipynb` contains example code for using the classes contained within the files in `src`.

## Reproducing the simulation studies in the paper

In the directory `simulations/run_files` can be found `.sh` scripts for reproducing the simulation studies from the paper. For example, to reproduce the results from simulation 1, you can run the following commands:

```sh
cd simulations/run_files
chmod +x run_simulation_study_1.sh
./run_simulation_study_1.sh
```

The resulting output is saved `simulations/output` and the plots in the paper can be reproduced using `simulations/analysis_1.ipynb`.

## FAO trade network

`data/FAO/raw_data` contains all the raw data described in the paper. The scripts `data/FAO/process_FAO.py` and `data/FAO/process_FAO_features.py` can then be run
to process this raw data, which is then saved into `data/FAO/processed_data`. Note that this folder is already populated.