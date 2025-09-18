#!/bin/bash

##################### SLURM (do not change) v  #####################
#SBATCH --export=ALL
#SBATCH --job-name="lab01"
#SBATCH --nodes=1
#SBATCH --output="lab01.%j.%N.out"
#SBATCH -t 00:15:00
##################### SLURM (do not change) ^  #####################

# Above are SLURM directives for job scheduling on a cluster,
export SLURM_CONF=/etc/slurm/slurm.conf


echo "----- Building -----"
# Do not change below, it is fixed folsr everyone
SHAREDDIR=/home/coe4sp4/

cmake -S . -B $(pwd)/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=${SHAREDDIR}/libpfm4/ -DPROFILING_ENABLED=ON
cmake --build $(pwd)/build -- -j8
make -j8



echo "---- Running ----"

mkdir -p $(pwd)/logs
$(pwd)/build/lab01 --benchmark_out="$(pwd)/logs/lab01.json" --benchmark_out_format=json --benchmark_perf_counters="L1-dcache-loads"


echo "---- Running Tests ----"

$(pwd)/build/test/sort_test


echo "---- Plotting ----"
# create python virtual environment
python3 -m venv $(pwd)/venv
source $(pwd)/venv/bin/activate
pip install -r $(pwd)/script/requirements.txt
mkdir -p $(pwd)/plots

python3 $(pwd)/script/plot.py $(pwd)/logs/lab01.json
