#!/bin/bash

##################### SLURM (do not change) v  #####################
#SBATCH --export=ALL
#SBATCH --job-name="lab02"
#SBATCH --nodes=1
#SBATCH --output="lab02.%j.%N.out"
#SBATCH -t 00:45:00
##################### SLURM (do not change) ^  #####################

# Above are SLURM directives for job scheduling on a cluster,
export SLURM_CONF=/etc/slurm/slurm.conf


echo "----- Building -----"
# Do not change below, it is fixed folder everyone
SHAREDDIR=/home/coe4sp4/

cmake -S . -B $(pwd)/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=${SHAREDDIR}/libpfm4/ -DPROFILING_ENABLED=ON
cmake --build $(pwd)/build -- -j8
make -j8



echo "---- Running ----"

mkdir -p $(pwd)/logs
$(pwd)/build/lab02-part1 --benchmark_out="$(pwd)/logs/lab02-part1.json" --benchmark_out_format=json
$(pwd)/build/lab02-part2 --benchmark_out="$(pwd)/logs/lab02-part2.json" --benchmark_out_format=json


echo "---- Running Tests ----"

$(pwd)/build/test/heat_test



echo "---- Plotting ----"
# create python virtual environment
python3 -m venv $(pwd)/venv
source $(pwd)/venv/bin/activate
pip install -r $(pwd)/script/requirements.txt
mkdir -p $(pwd)/plots

python3 $(pwd)/script/plot.py $(pwd)/logs/lab02-part1.json
# TODO add more plotting commands