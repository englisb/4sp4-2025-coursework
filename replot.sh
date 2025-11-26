#!/bin/bash
#SBATCH --job-name=replot
#SBATCH --output=replot.%j.out
#SBATCH --time=00:05:00
#SBATCH --partition=compute
#SBATCH --nodes=1

rm -rf plots
mkdir -p plots
python3 -m venv venv
source venv/bin/activate
pip install -q numpy matplotlib pandas
python3 script/plot.py

