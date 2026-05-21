#!/bin/bash

##################### SLURM (do not change) v  #####################
#SBATCH --export=ALL
#SBATCH --job-name="tut01"
#SBATCH --nodes=1
#SBATCH --output="tut01.%j.%N.out"
#SBATCH -t 00:15:00
##################### SLURM (do not change) ^  #####################


cmake -S . -B $(pwd)/build -DCMAKE_BUILD_TYPE=Release
cmake --build $(pwd)/build -- -j8
make -j8

$(pwd)/build/tut01

$(pwd)/build/test/vec_op_test
