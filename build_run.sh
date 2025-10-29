#!/bin/bash

##################### SLURM (do not change) v  #####################
#SBATCH --export=ALL
#SBATCH --job-name="lab03"
#SBATCH --nodes=1
#SBATCH --output="lab03.%j.%N.out"
#SBATCH -t 00:45:00
##################### SLURM (do not change) ^  #####################

# Above are SLURM directives for job scheduling on a cluster,
export SLURM_CONF=/etc/slurm/slurm.conf

# Source Intel MKL environment
source /opt/intel/oneapi/setvars.sh --force
export MKL_NUM_THREADS=1
export MKL_DOMAIN_ALL=1

echo "----- Building -----"
# Do not change below, it is fixed folder everyone
SHAREDDIR=/home/coe4sp4/

cmake -S . -B $(pwd)/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=${SHAREDDIR}/libpfm4/ -DPROFILING_ENABLED=ON
cmake --build $(pwd)/build -- -j8
make -j8



echo "---- Running ----"

mkdir -p $(pwd)/logs
$(pwd)/build/lab03 --benchmark_out="$(pwd)/logs/lab03.json" --benchmark_out_format=json --benchmark_perf_counters="L1-dcache-loads"



echo "---- Running Tests ----"

$(pwd)/build/test/cholesky_test



echo "---- Plotting ----"
# create python virtual environment
python3 -m venv $(pwd)/venv
source $(pwd)/venv/bin/activate
pip install -r $(pwd)/script/requirements.txt
mkdir -p $(pwd)/plots

python3 $(pwd)/script/plot.py $(pwd)/logs/lab03.json
# TODO add more plotting commands

