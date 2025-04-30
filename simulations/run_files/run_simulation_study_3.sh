#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "Script is running from: ${SCRIPT_DIR}"

# Get the parent directory and then the simulation directory
PARENT_DIR="$(dirname "${SCRIPT_DIR}")"
SIMULATION_DIR="${PARENT_DIR}/simulation_3"
OUTPUT_DIR="${SIMULATION_DIR}/output"

# Set Python path to include one level back from the parent directory
PYTHON_PATH_DIR="$(dirname "${PARENT_DIR}")"
export PYTHONPATH="${PYTHON_PATH_DIR}"

echo "Parent directory: ${PARENT_DIR}"
echo "Simulation directory: ${SIMULATION_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "PYTHONPATH set to: ${PYTHONPATH}"

# Create output directories if they don't exist
mkdir -p "${OUTPUT_DIR}/progress_output"
mkdir -p "${OUTPUT_DIR}/o_files"
mkdir -p "${OUTPUT_DIR}/e_files"
mkdir -p "${OUTPUT_DIR}/alpha_rho"
mkdir -p "${OUTPUT_DIR}/beta_rho"
mkdir -p "${OUTPUT_DIR}/ELBO"
mkdir -p "${OUTPUT_DIR}/layer_groups"
mkdir -p "${OUTPUT_DIR}/phi_w"
mkdir -p "${OUTPUT_DIR}/phi_z"

# Changing working directory so things save correctly
cd "${SIMULATION_DIR}"


# For local testing, saved as 5. To run the full set of simulations,
# set from 0-99.
for ARRAY_ID in {0..5}; do
    echo "Running simulation with ID: ${ARRAY_ID}"

    # Run the simulations with named arguments
    echo "Running generate_layer_groups_3.py..."
    python3 -u "${SIMULATION_DIR}/generate_layer_groups_3.py" --index ${ARRAY_ID} > "${OUTPUT_DIR}/progress_output/generate_layer_groups_${ARRAY_ID}.out" 2> "${OUTPUT_DIR}/e_files/generate_error_${ARRAY_ID}.log"
    
    echo "Running simulation_3.py..."
    python3 -u "${SIMULATION_DIR}/simulation_3.py" --index ${ARRAY_ID} > "${OUTPUT_DIR}/progress_output/simulation_3_output_${ARRAY_ID}.out" 2> "${OUTPUT_DIR}/e_files/simulation_error_${ARRAY_ID}.log"
    
    echo "Completed simulation ${ARRAY_ID}"
done

echo "All simulations completed"