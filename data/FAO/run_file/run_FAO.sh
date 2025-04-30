#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "Script is running from: ${SCRIPT_DIR}"

# Get the parent directory and then the simulation directory
PARENT_DIR="$(dirname "${SCRIPT_DIR}")"
SIMULATION_DIR="${PARENT_DIR}/FAO"
OUTPUT_DIR="${SIMULATION_DIR}/output_features"

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

# Match the PBS array indices from the original script (18-26)
for ARRAY_ID in {0..26}; do
    echo "Running FAO with ID: ${ARRAY_ID}"
    
    # Run the simulation
    echo "Running run_vb_FAO.py..."
    python3 -u "${SIMULATION_DIR}/run_vb_FAO.py" --index ${ARRAY_ID} > "${OUTPUT_DIR}/progress_output/FAO_output_${ARRAY_ID}.out" 2> "${OUTPUT_DIR}/e_files/FAO_error_${ARRAY_ID}.log"
    
    echo "Completed FAO with ID: ${ARRAY_ID}"
done

echo "All FAO simulations completed"