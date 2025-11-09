[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/B4ZsHeQB)
# 🎓 Lab 04: Parallelism

The goal of this lab is to learn how to write parallel code for multicores and GPUs. 
At the end of the lab:
* you will know how to program multicores using OpenMP.
* you will learn sparse linear solvers and effect of sparsity in parallelism.
* you will be able to program GPUs using CUDA.


**Note:** You must review Tutorial 01 and 02 before starting this lab.
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

In this lab, you will work make sparse lower triangular solver parallel
using OpenMP and implement sorting on GPU using CUDA.
We have discuss sparse triangular solvers in the lectures. Please review
the sparsity lecture before starting this lab. Both tasks will use
the CSR format to store the sparse matrix. Four matrices from the 
suite sparse collection are provided in the `data/` folder. Also 
one small test matrix is provided for debugging purposes. 
All matrices are copied when you run the `build_run.sh` script.


* While the lab is about parallelism, you are allowed to apply all 
  optimization techniques you have learned so far.

### Task 1 : Multicore Programming
Implement the sequential sparse lower triangular solver with CSR format and then make it parallel
using OpenMP. Once implemented, you will need to discuss why the performance of 
the parallel implementation is better/worse than the sequential one for different sparse matrices.
You can use any performance counter or metric to justify your analysis.



**Note** You can enable MKL implementation using `-DUSE_MKL=ON` when calling `cmake`.


### Task 2 : GPU programming
Impelement SpTRSV on CSR on GPU using CUDA. Once implemented, you will need to discuss why the performance of
the GPU implementation is better/worse than the sequential one for different sparse matrices.
You can use any performance counter or metric to justify your analysis.

**Note** You should enable the GPU part by using `-DGPU_ENABLED=ON` when calling `cmake` (see `build_run.sh`).

**Note** Matrices are loaded in coordinate (COO) format. You will need to convert them to CSR format before using them in SpTRSV.

### The expected output
Your submission must include the following:
* Parallel sparse lower triangular solver using OpenMP and verified implementation.
* Provide profiling analysis to justify the performance of
  the parallel sparse lower triangular solver for different matrices (with plots). Make sure to connect profiling results to your code optimization techniques.
* GPU implementation of sorting and verified implementation.
* Provide profiling analysis to justify the performance of
  the GPU sorting implementation for different sizes of data (with plots). Make sure to connect profiling results to your code optimization techniques.
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

### Plot(s) 1: Parallel multicore SpTRSV performance analysis

TODO: make sure to reference the correct plot below

![Figure 1: ](plots/plot1.png)

Description: TODO: please provide details for your plot(s) here.


### Plot(s) 2: GPU SpTRSV performance analysis
TODO: follow like above example


### Plot(s) 3: Bonus: performance comparison with MKL and cusparse
TODO: follow like above example



