[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/-qJn85GV)
# 🎓 Lab 03: Vectorization

The goal of this lab is to learn how to use vector processors using AVX
  intrinsics. At the end of the lab:
* you will know how to work with AVX intrinsics in C++.
* you will learn linear solvers and n-body problems.
* You will analyze the performance of your code using Google Benchmark, Profilers, and
  plot the results.


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

In this lab, you will work with two different important 
computations: Cholesky decomposition and N-body simulation.

* Apply any necessary transformation to the Cholesky and n-body problems to make them vectorizable.
* Then implement the vectorized version of the Cholesky and n-body problems using avx intrinsics.


### Task 1 : Cholesky Decomposition

Cholesky decomposition is a numerical method used to solve systems of linear equations,
particularly when the coefficient matrix is symmetric and positive definite. It decomposes
a matrix into the product of a lower triangular matrix and its transpose. This method is
computationally efficient and is widely used in various applications, including optimization,
machine learning, and simulations.

In this task, you will first implement Cholesky decomposition and test it for correctness.
Then, you will make a new vectorized implementation using AVX intrinsics to improve its performance.

#### Overview
Cholesky decomposition is a method for decomposing a symmetric positive-definite matrix
`A` into the product of a lower triangular matrix `L` and its transpose `L^T`, such that:
```angular2html
A = LL^T
```
Where:
- `A` is a symmetric positive-definite matrix.
- `L` is a lower triangular matrix.
- `L^T` is the transpose of `L`.
- The decomposition is unique if `A` is positive definite.
- Cholesky decomposition is computationally efficient, requiring approximately `n^3/3`
  operations for an `n x n` matrix, which is about half the cost of other methods like
  LU decomposition.
- It is widely used in numerical simulations, optimization problems, and machine learning,
  particularly in algorithms like Gaussian processes and Kalman filters.
- Cholesky decomposition is numerically stable and less sensitive to round-off errors
  compared to other decomposition methods.
- It is particularly useful for solving large systems of linear equations
  where the matrix is symmetric and positive definite.
- The method is not applicable to non-symmetric or non-positive-definite matrices.
- once `L` is computed, it can be used to efficiently solve systems of equations
  of the form `Ax = b` by solving two triangular systems: `Ly = b` and `L^Tx = y`.

#### Algorithm
The Cholesky decomposition algorithm iteratively computes the elements of the
lower triangular matrix `L` using the following formulas:
```angular2html
L[j][j] = sqrt(A[j][j] - sum(L[j][k]^2 for k in range(j)))
L[i][j] = (A[i][j] - sum(L[j][k] * L[i][k] for k in range(j))) / L[j][j] for i > j
```
Where:
- `L[j][j]` is the diagonal element of `L`.
- `L[i][j]` is the off-diagonal element of `L`.
- `A[i][j]` is the element of the original matrix `A`.
- `sum` represents the summation over the specified range.
- The algorithm proceeds row by row, calculating the diagonal elements first,
  followed by the off-diagonal elements.
- The process continues until all elements of `L` are computed.

### Blocked Cholesky In-place Decomposition (Bonus / Optional)
Blocked Cholesky decomposition is an optimized version of the standard Cholesky
decomposition algorithm that improves performance by processing the matrix in 
blocks or submatrices. This approach takes advantage of modern computer architectures,
which are designed to efficiently handle data in blocks, leading to better cache utilization
and reduced memory access times.
The blocked Cholesky decomposition algorithm divides the matrix `A` into smaller
submatrices or blocks of size `b x b`. The algorithm then performs the decomposition
on these blocks, updating the blocks of the lower triangular matrix `L` iteratively. The key steps of the blocked Cholesky decomposition algorithm are as follows:
1. **Partitioning**: Divide the matrix `A` into blocks of size `b x b`.
2. **Block Decomposition**: For each block, perform the Cholesky decomposition
   on the diagonal block.
3. **Update Off-Diagonal Blocks**: Update the off-diagonal blocks using the results
   from the diagonal block decomposition.
4. **Repeat**: Continue the process for all blocks in the matrix until the entire
   matrix is decomposed.
The blocked Cholesky decomposition algorithm is particularly effective for large matrices,
as it reduces the number of memory accesses and improves cache performance.

**Important Note**: The Cholesky here should happen in-place. It means the ouptput matrix 
L should overwrite the input matrix A. This is important to be compatible with the 
Basic Linear Algebra Subprograms (BLAS) standard. 

**Note** You can enable MKL implementation using `-DUSE_MKL=ON` when calling `cmake`.


### Task 2 : N-body Simulation
N-body simulation is a computational method used to study the dynamics of a system
of particles under the influence of physical forces, such as gravity or electromagnetism.
It is widely used in astrophysics, molecular dynamics, and other fields to model
the interactions and movements of multiple bodies over time.
In this task, you will first learn a basic N-body simulation (the implementation is provided). 
Then, you will make a new vectorized implementation using AVX intrinsics to improve its performance.

#### Overview
N-body simulation involves calculating the trajectories of `N` particles
under the influence of mutual forces. The key components of an N-body simulation include:
- **Particles**: Each particle has properties such as position, velocity, mass,
  and charge (if applicable).
- **Forces**: The forces acting on each particle are computed based on the positions
  and properties of all other particles. Common forces include gravitational and electrostatic forces.
- **Time Integration**: The simulation progresses in discrete time steps, updating the positions
  and velocities of particles based on the computed forces.
- **Boundary Conditions**: Depending on the simulation, boundary conditions may be applied,
  such as periodic boundaries or reflective boundaries.
- **Visualization**: The results of the simulation can be visualized to analyze the behavior
  of the system over time.
- N-body simulations can be computationally intensive, especially for large `N`,
  as the number of interactions scales with `N^2`. Various optimization techniques,
  such as tree algorithms or fast multipole methods, can be employed to reduce computational complexity.
- N-body simulations are used in various applications, including:
  - Astrophysics: Modeling the dynamics of star clusters, galaxies, and planetary systems.
  - Molecular Dynamics: Studying the interactions of atoms and molecules in materials science and biochemistry.
  - Plasma Physics: Simulating the behavior of charged particles in a plasma.
  - Computer Graphics: Generating realistic animations of particle systems.
  - Game Development: Simulating physical interactions in virtual environments.
  - Robotics: Planning and simulating the motion of multiple robots in a shared space.
  - Fluid Dynamics: Modeling the behavior of particles in fluid flows.
  - and many more.

NOTE: Keep the size of `N` in n-body simulation to be smaller than 1000 to avoid long runtimes.  

#### Algorithm
The basic algorithm for an N-body simulation involves the following steps:
1. **Initialization**: Set up the initial conditions for the particles, including their positions,
   velocities, masses, and other relevant properties.
2. **Force Calculation**: For each particle, compute the net force acting on it due to all other particles. 
3. **Time Integration**: Update the positions and velocities of each particle based on the computed forces
   using numerical integration methods such as Euler's method or Verlet integration.
4. **Boundary Conditions**: Apply any necessary boundary conditions to the particles,
   such as reflecting off walls or wrapping around in periodic boundaries.
5. **Repeat**: Repeat the force calculation and time integration steps for a specified number of time steps
   or until a certain condition is met (e.g., particles reach a stable configuration).
6. **Output**: Record the positions and velocities of the particles at each time step for analysis and visualization.
7. **Optimization**: For large `N`, consider using optimization techniques to reduce computational complexity,
   such as spatial partitioning (e.g., octrees) or approximating distant interactions.
8. **Visualization**: Visualize the results of the simulation to analyze the behavior of the system over time.
9. **Analysis**: Analyze the simulation data to extract meaningful insights about the dynamics of the system.



### The expected output
Your submission must include the following:
* Cholesky baseline implementation and at least 3 test cases.
* Vectorized using AVX instructions and verified Cholesky implementation.
* Provide profiling analysis to justify the performance of 
 the vectorized Cholesky implementation for different dimensions (with plots). 
 Make sure to connect profiling results to your code optimization techniques.
* The bonus part: Optimizing the Cholesky code to outperform the BLAS library's 
  implementation of Cholesky decomposition (DPOTRF) for all matrix sizes. 
  You will need to show profiling analysis to justify your performance claims (with plots).
* Vectorized using AVX instructions and verified N-body implementation.
* Provide profiling analysis to justify the performance of 
 the vectorized N-body implementation (with plots). Make sure to connect 
 profiling results to your code optimization techniques.
* The necessary Python scripts (and packages) to generate all plots. 
 You won't be allowed to push plots to the repo. We should be able to 
 generate all plots using your scripts and the logs generated from running your
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

### Plot(s) 1: Cholesky decomposition performance analysis

TODO: make sure to reference the correct plot below

![Figure 1: Standard Sort Algorithm Complexity](plots/plot1.png)

Description: TODO: please provide details for your plot(s) here.


### Plot(s) 2: N-body simulation performance analysis
TODO: follow like above example


### Plot(s) 3: Bonus: blcoked Cholesky decomposition performance analysis
TODO: follow like above example



