#!/bin/bash

##################### SLURM (do not change) v  #####################
#SBATCH --export=ALL
#SBATCH --job-name="lab-04"
#SBATCH --nodes=1
#SBATCH --output="lab-04.%j.%N.out"
#SBATCH -t 00:45:00
##################### SLURM (do not change) ^  #####################

# Above are SLURM directives for job scheduling on a cluster,
export SLURM_CONF=/etc/slurm/slurm.conf


echo "----- Building -----"
# Do not change below, it is fixed for everyone
SHAREDDIR=/home/coe4sp4/

# Source Intel MKL environment
source /opt/intel/oneapi/setvars.sh --force


#cmake -S . -B $(pwd)/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=${SHAREDDIR}/libpfm4/ -DPROFILING_ENABLED=ON -DUSE_MKL=ON -DOPENMP=ON
cmake -S . -B $(pwd)/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=${SHAREDDIR}/libpfm4/ -DPROFILING_ENABLED=ON -DUSE_MKL=ON -DOPENMP=ON -DGPU_ENABLED=ON
cmake --build $(pwd)/build -- -j8


echo "---- Copying Matrices-----"
mkdir -p $(pwd)/data/
cp -r /home/coe4sp4/matrices/* $(pwd)/data/

echo "---- Running CPU ----"
mkdir -p $(pwd)/logs
$(pwd)/build/lab04-cpu --benchmark_out="$(pwd)/logs/lab04-cpu.json" --benchmark_out_format=json --benchmark_perf_counters="L1-dcache-loads"

echo "---- Running GPU----"
echo "Note. to run the GPU part, you will need to first enable GPU by add -DGPU_ENABLED=ON"
$(pwd)/build/lab04-gpu --json "$(pwd)/logs/lab04_gpu.json"

echo "---- Running Tests ----"
$(pwd)/build/test/sptrsv_omp_test
$(pwd)/build/test/sptrsv_gpu_test

echo "---- Plotting ----"
python3 -m venv $(pwd)/venv
source $(pwd)/venv/bin/activate
pip install -r $(pwd)/script/requirements.txt
mkdir -p $(pwd)/plots
python3 $(pwd)/script/plot_parallel.py $(pwd)/logs/lab04-cpu.json
python3 $(pwd)/script/plot_gpu.py $(pwd)/logs/lab04_gpu.json