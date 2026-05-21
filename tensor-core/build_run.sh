#!/bin/bash

##################### SLURM (do not change) v  #####################
#SBATCH --export=ALL
#SBATCH --job-name="tut03"
#SBATCH --nodes=1
#SBATCH --output="tut03.%j.%N.out"
#SBATCH -t 00:15:00
##################### SLURM (do not change) ^  #####################
export SLURM_CONF=/etc/slurm/slurm.conf

echo "----- Building tut03 -----"
cmake -S . -B $(pwd)/build -DCMAKE_CUDA_ARCHITECTURES=89  -DCMAKE_BUILD_TYPE=Release
cmake --build $(pwd)/build -- -j8
make -j8

echo "---- Running tut03 ----"
$(pwd)/build/tensor-core-demo --json tensor-core-demo_results.json


