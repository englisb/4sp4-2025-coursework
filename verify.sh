#!/bin/bash
#SBATCH --job-name=verify
#SBATCH --output=verify.%j.out
#SBATCH --time=00:05:00
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

module load cuda/12.0.140

nvcc -o verify_correctness verify_correctness.cu src/sparse_io.cpp \
     -I./include -I./build/_deps/fmt-src/include -lcusparse -std=c++17

if [ $? -eq 0 ]; then
    ./verify_correctness
else
    echo "Compilation failed"
    exit 1
fi
