[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/hlMxVAew)

# 🎓 Lab 02: Cache and Tiling

The goal of this lab is to learn the memory architecture of
a multi-core processor and how to use the memory hierarchy to
speedup an application At the
end of the lab:
* you will know memory hierarchy of the processor that you are working.
* you will be able to use profiler to analyze the performance of your code.
* you will apply tiling to benefit from the fast memory.

**Note:** You must review Tutorial 01 before starting this lab.
***



## 🛠️ Logging in to the ECE Cluster
The ECE computer server is a local server used for this course. The ECE server
has 6 nodes, each with a 20-core Intel CPU and an Ada2000 GPU. The ECE server
uses a SLURM scheduler. SLURM (Simple Linux Utility for Resource Management)
is an open-source workload manager that allocates exclusive resources (computer nodes)
to users for a specific duration to run their tasks. It manages a queue of jobs,
ensuring that resources are used efficiently and jobs are run fairly. You will
need a username to log in to the server.  Your username is the one that you use
to login to PCs in labs. With that, you will need to SSH to the server using the
following command:

 ```
 ssh <username>@srv-cad.ece.mcmaster.ca
 ```

***

## ➕ Cloning the Repository

This step explains how to clone the repository. You will use **`git`**, a version
control system, to clone the repository and push your code to it. To clone
the repository, you will need to use the following command:


```
git clone https://github.com/4sp4-2025/<repo name>.git
```

Where `repo name` is your repository. The repository is private, so you may get an
error when you enter your username and password. To solve this, you will need to
use a **Personal Access Token (PAT)** instead of your password.


***

## 🚀 Building and running the starter code

This section explains how you can build and run the code on the
ECE cluster. First, go to where the project is cloned:

```
cd <where/the/repo/is/cloned>
```
All necessary instructions to build and run the code are
provided in the `build_run.sh` file.
You can open the file using a text editor like `nano` or `vim`,
e.g., `vim build_run.sh`.  Use the following command to build
and run the code:

```
sbatch build_run.sh
```

This command submits the `build_run.sh` script to the SLURM scheduler.
The instructions in the `build_run.sh` file are
unix commands that will be executed on a compute node. The output of
the job will be saved in a `*.out`file.

* **Never use `bash`** to run your code on the login node.
  Use `sbatch` instead. `bash` runs the code directly on
  the login node, while `sbatch` submits the job to the SLURM
  scheduler, which runs it on an available compute node.
* You should **never** run code on the login node unless it
  takes less than 30 seconds.
* You can check the status of your job using `squeue -u <username>`.


***


## ✅ Tasks

You will first design an experiment to determine different levels of 
caches and cache line size. Then, you will implement a numerical solver 
for the heat equation and apply loop tiling to it to benefit from the 
cache hierarchy of the processor.

*Note*: the number of iterations in the google benchmark should not be larger than 3.

### Task 1 : Cache Hierarchy Analysis
In this task, you will analyze cache hierarchy of the processor's memory
and implement a copy benchmark to measure the size of different levels of caches. 
You will need to do time measurement for this part. 
You will also need to do profiling with proper counters to back up your cache size claims.
You may need to work with compiler optimization flags to get the best results.
To ensure the copy function works as expected, you should pick a flag that does not change the 
code behaviour. 

### Task 2 : Heat Equation Solver
In this task you will first implement a part of the numerical solver 
for the heat equation. Then, you will apply loop tiling to the solver 
to benefit from the cache hierarchy of the processor.


### Overview

The heat equation is a partial differential equation (PDE) that 
describes the distribution of heat (or temperature) in a given 
region over time. It is widely used in physics and engineering
to model heat conduction. The equation in two dimensions is given as:
```angular2html
∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²)
```
Where:
- `u(x, y, t)` is the temperature at position `(x, y)` and time `t`.
- `a` is the thermal diffusivity constant.
- `∂u/∂t` is the rate of change of temperature over time.
- `∂²u/∂x²` and `∂²u/∂y²` are the second spatial derivatives of temperature.

The equation is solved numerically using finite difference methods, 
where the spatial domain is discretized into a grid, and the time evolution 
is computed iteratively.

### Numerical Solver

#### Five-Point Stencil Formula

The numerical solver uses the five-point stencil formula to compute the 
temperature at each grid point:
```angular2html
u(i, j, t+Δt) = u(i, j, t) + αΔt( (u(i+1, j, t) - 2u(i, j, t) + u(i-1, j, t))/Δx² + (u(i, j+1, t) - 2u(i, j, t) + u(i, j-1, t))/Δy² )
```
Where:
- `u(i, j, t)` is the temperature at grid point `(i, j)` at time `t`.
- `Δt` is the time step.
- `Δx` and `Δy` are the spatial step sizes in the x and y directions, respectively.
- `α` is the thermal diffusivity constant.
- `u(i+1, j, t)`, `u(i-1, j, t)`, `u(i, j+1, t)`, and `u(i, j-1, t)` are the 
 temperatures at the neighboring grid points. The formula updates the temperature 
 at each grid point based on its current value and the values of its four immediate neighbors.
 This method is efficient and provides a good approximation of the heat distribution over time.

#### What is Stencil computation?
Stencil computations are a class of numerical data processing solution which update array
elements according to some fixed pattern, called a stencil.

#### Boundary Conditions
The solver implements Dirichlet boundary conditions, where the temperature at the
boundaries of the grid is fixed.

#### Initial Conditions
The initial temperature distribution is set using a Gaussian function centered in
the grid, simulating a heat source.

### Code Description

#### Key Components

1. **Data Structures**:
  - The `field` structure represents the temperature grid, including its dimensions, 
   grid spacing, and temperature values.

2. **Numerical Solver**:
  - The `evolve` function updates the temperature grid using the five-point stencil method.
  - The `evolve_tiled` function applies loop tiling to improve cache performance.

3**Output**:
- The code writes the temperature field to a PNG file at regular intervals and 
  computes the average temperature for analysis.


For task 2, you will implement the evolve first and then you will tile the evolve function. 
You will need to do proper experimentation and profiling to select the tile size(s).

*Hint* an improvement of around 5% is acceptable. More than 10% is excellent and 
will be rewarded with bonus points.

### The expected output
Your submission must include the following:

* Necessary benchmarks to make an experiment for the cache and cache line size.
* Use above data to determine the size of different levels of caches. (with plots)
* Implement the stencil kernel for the heat equation and provide enough tests to verify its correctness.
We do not grade your tests but it is important to verify your code works as expected.
* Apply loop tiling to the stencil kernel and show the performance improvement versus the baseline code.
* Profile the stencil kernel for different sizes of data to find the best tile size and 
 the loops that are selected for tiling (with plots)
* The necessary Python scripts (and packages) to generate all plots. You won't be allowed to push plots to the repo.
  We should be able to generate all plots using your scripts and the logs generated from running your
  code on the ECE server.


### Evaluation
Your lab submission will be graded based on these criteria:

* Your code compiles and runs successfully on the ECE cluster using the provided build_run.sh script (Pass/Fail).
* All grading tests are passed as expected.
* The quality of your plots and its analysis, which must:
  * Convey a correct and clear argument.
  * Be visually understandable with proper labels, units, and a concise 1-2 sentence executive summary.
* All plots must be generated using your provided Python scripts and the logs from your code runs on the ECE server.
  Plots pushed directly to the repository will not be accepted.
  **Negative** points will be assigned for any plots/logs that are pushed to the repo.


### Submission
Follow these guidelines for a successful submission:
* First, please implement all TODOs in the code and remove all comments that
  start with `TODO`.
* Push all your code to the main branch of your repository before the deadline.
* Ensure your code compiles and runs on the ECE cluster. Submissions that fail to
  compile or run will receive a grade of zero. The TA will only use the `build_run.sh` file for
  building and running your code. You may want to make a copy of the script for your final testing.
* The build_run.sh script's execution should only output results from Google Test and
  Google Benchmark. Do not include any extra output from cout or printf. This can negatively
  impact your grade, and regrade requests for this reason will not be accepted.
* All logs should be redirected to `logs/` directory, and all plots should be saved in the
  `plots/` directory. These directories are not tracked by git to avoid pushing log files
  or plots to the repository.




## Descriptive Answers (TODO)
Typically there is no single correct answer/plot for the following questions. Rely on your thought process!

### Plot(s) 1: Cache Hierarchy Experiment

![Figure 1: Cache Throughput Analysis](plots/plot1_cache_throughput.png)

Description: This plot would demonstrate memory bandwidth vs array size to identify cache levels through throughput measurements. Currently not implemented as Part 1 benchmarks (BM_COPY) are marked as "SKIPPED: Not implemented yet". The experiment would measure copy performance across different array sizes to identify L1 (48KB), L2 (3MB), and L3 (30MB) cache boundaries through bandwidth changes and latency spikes at cache transitions.

### Plot(s) 2: Heat Equation Performance Analysis

![Figure 2: Performance Comparison with Cache Hierarchy](plots/plot2_speedup.png)

Background on tiling: Loop tiling partitions the i×j iteration space into smaller rectangular tiles so that the data required to compute a tile fits into a given cache level and is reused before eviction. For a five-point 2D stencil, a Ti×Tj tile reads an (Ti+2)×(Tj+2) patch from the input field (one-cell boundary on each side) and writes a Ti×Tj patch to the output field. With doubles (8 bytes), a 32×32 tile has a working set of approximately [(34×34) + (32×32)]×8 ≈ 17 KB, which fits in L1; a 64×64 tile is ≈ 66 KB, which exceeds a 48 KB L1 but fits in L2.

Analysis: The tiled implementation runs consistently faster than the default row-major baseline, with the advantage shrinking as the problem becomes memory bound. For small grids (n = 64–128), tile working sets reside in L1 and reuse across both i and j is high; the speedup approaches 2×. Near n ≈ 512, the combined fields exceed a 3 MB L2 (approximate threshold n ≈ sqrt(3 MB / 16 B) ≈ 440 when accounting for two fields), so baseline capacity/conflict misses rise; tiling still amortizes reuse within tiles and yields about 1.5–1.6×. At n ≈ 1024 the problem sits comfortably in a 30 MB L3 (threshold n ≈ sqrt(30 MB / 16 B) ≈ 1400), and the observed gain is around 1.6×. Beyond n ≈ 2048, the working set exceeds L3 and execution is DRAM-bandwidth-bound for both versions; tiling reduces write-allocate traffic and preserves locality, but the benefit drops to roughly 1.2×–1.15×. These trends align with the shaded cache regions: the closer the effective working set is to L1/L2, the larger the gap; once memory bandwidth dominates, the gap narrows.

How the speedup is computed: For each grid size, speedup = time(BM_STENCIL) / time(BM_STENCIL_TILED). Times are the Google Benchmark real_time values for the same fixed number of steps, so the ratio isolates the impact of iteration reordering.

Relation to the L1 baseline: The bottom panel normalizes time by the n = 64 case, where a 32×32 tile fits in L1 and achieves near-ideal reuse. As n grows past L2 and then L3 capacities, normalized time increases steeply for both versions because bytes moved per step scale with n² while sustained bandwidth plateaus. Tiling retains an advantage at every size by shortening reuse distance, improving prefetch/TLB locality, and reducing needless write-allocate traffic, but the improvement shrinks once DRAM dominates.

### Plot(s) 3: Tile Size Optimization

![Figure 3: Tile Size Sweep Analysis](plots/plot3_tile_sweep.png)

Analysis: The sweep fixes the grid at 1024×1024 (residing in L3) and varies the tile size. The working set for a Ti×Tj tile is approximately [(Ti+2)(Tj+2) + TiTj] doubles, or eight bytes per double. Representative sizes are: 8×8 ≈ 1.3 KB (far below L1), 32×32 ≈ 17 KB (well within L1), 64×64 ≈ 66 KB (slightly above a 48 KB L1, within L2), 128×128 ≈ 260 KB (well within L2), and 256×256 ≈ 1.0 MB (still in L2). Very small tiles such as 8×8 incur high loop and boundary overhead and underutilize spatial locality, so speedup is near 1× or slightly below baseline. Medium tiles (32×32 and 64×64) exploit L1/L2 locality while keeping boundary cost modest, yielding roughly 1.5×–1.6×. The best runtime occurs near 128×128: the tile fits easily in L2, the boundary-to-interior ratio is low, and reuse across both dimensions is maximized, producing the peak speedup in the sweep. Increasing to 256×256 reduces the number of tiles but enlarges the per-tile footprint and reuse distance; although the footprint remains in L2, longer line lifetimes and write-allocate pressure reduce locality, so speedup slips slightly from the 128×128 optimum.

Connection to data representation and caches: Each grid cell is a double (8 bytes) and two fields are maintained (read and write) with a one-cell boundary during updates. Tiling is most effective when the read tile plus boundary and the destination sub-tile fit comfortably in a single cache level with associativity headroom. The observed U-shaped trend follows from this balance: very small tiles waste cycles on overhead, very large tiles dilute locality, and an intermediate size that maps to L1/L2 capacity yields the best performance for this grid size.


# Licence
Created by SwiftWare Lab on 9/25.
CE 4SP4 - High Performance Programming
Copyright (c) 2025 SwiftWare Lab.
Distribution of the code is not
allowed in any form without permission
from SwiftWare Lab.




