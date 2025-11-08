# 🎓 Tutorial 02: GPU Basics
This tutorial teaches you how to build and run your GPU code in CE 4SP4.

The main goals of this tutorial are:
* How to build and run GPU code on the ECE cluster.
* Get started with GPU programming and profiling.


***

## 🛠️ Logging in to the ECE Cluster
The ECE computer server is a local server used for this course. The ECE server has 6 nodes, each with a 20-core Intel
CPU and an Ada2000 GPU. The ECE server uses a SLURM scheduler. SLURM (Simple Linux Utility for Resource Management)
is an open-source workload manager that allocates exclusive resources (computer nodes) to users for a specific duration
to run their tasks. It manages a queue of jobs, ensuring that resources are used efficiently and jobs are run fairly.  
You will need a username to log in to the server.
Your username is the one that you use to login to PCs in labs. With that, you will need to SSH to the server using the following command:

 ```
 ssh <username>@srv-cad.ece.mcmaster.ca
 ```

***

## ➕ Cloning the Repository

This step explains how to clone the repository. You will use **`git`**, a version control system, to clone the
repository and push your code to it. To clone the repository, you will need to use the following command:


```
git clone https://github.com/4sp4-2025/<repo-name>.git
```

Where `repo-name` is specific to your repo.
The repository is private, so you may get an error when you enter your username and password. To solve this,
you will need to use a **Personal Access Token (PAT)** instead of your password.
See more information [here](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token).
We recommend using a **classic PAT**. After your PAT is set up, copy the token and save it somewhere secure.
You will use it instead of your password when prompted.
For a quick list of useful git commands, see the appendix [Useful Git commands](README.md#useful-git-commands).

***

## 🚀 Building and running the starter code

### The ECE cluster
This section explains how you can build and run the code on the ECE cluster. First, go to where the project is cloned:

```
cd <repo-name>
```
All necessary instructions to build and run the code are provided in the `build_run.sh` file.
You can open the file using a text editor like `nano` or `vim`, e.g., `vim build_run.sh`.
Use the following command to build and run the code:

```
sbatch build_run.sh
```

This command submits the `build_run.sh` script to the SLURM scheduler. The instructions in the `build_run.sh` file are
unix commands that will be executed on a compute node. The output of the job will be saved in a file named `tut02.<job_id>.<node_id>.out`,
where `<job_id>` is the ID of the job assigned by SLURM. You can check the content of this file using the `cat`, `nano`, or `vim` commands.

* **Never use `bash`** to run your code on the login node. Use `sbatch` instead. `bash` runs the code directly on
  the login node, while `sbatch` submits the job to the SLURM scheduler, which runs it on an available compute node.
* You should **never** run code on the login node unless it takes less than 30 seconds.
* You can check the status of your job using `squeue -u <username>`.
* See the appendix [Useful SLURM commands](README.md#useful-slurm-commands) for more information about `sbatch` and
  other useful commands.

### Local machine

While the ECE server is required for some parts of the course, it's often more convenient to work on your local machine
for tasks like debugging. We recommend setting up the code on your choice of editor. However, it is important to
have Nvidia GPU to be able to run your Cuda code locally.

***

## ✅ Tasks

You will need to start working on the TODO items described below:

### Task 1: Getting started with NVIDIA nvbench and GPU programming

You will need to add another implementation of the vector operation
and compare its performance with the existing two implementations of the same operation
using **NVIDIA nvbench (nvbench)**. nvbench is a framework for benchmarking GPU code that
we will use in this course. Your implementation (`mulAddKernel_v3`) should be designed such that each thread
performs 4 operations (instead of one operation per thread in the current two kernels).
Two nvbench benchmark functions are provided in the `main_bench.cu` file. Add another benchmark function for
your implementation. Keep the benchmark arguments the same.

* **NVIDIA nvbench** is a library to benchmark GPU code. See more info [here](https://github.com/NVIDIA/nvbench).

### Task 2: Getting started with GPU programming
Profile the three kernels and plot the most relevant metrics using NVIDIA Nsight.
The profiling report is stored as CSV/JSON files in the `logs` directory.
Plot `L1/TEX Hit Rate` and `Memory Throughput` for the three implementations across different sizes.


### Evaluation
You will be evaluated based on the following criteria:
* Quality and number of the test cases
* Performance comparison using Nvidia Benchmark
* No report is required for this tutorial.

### Submission
* Enter your McMaster ID in `mac_id.txt`.
* You will need to push your code to the repository's main branch.
* Make sure to commit and push your code before the deadline.
* Make sure your code compiles and runs on the ECE cluster. If your code does not compile or run, you will get a zero for this tutorial.
  The TA only runs the `build_run.sh` file to build and run your code.
* There should not be any extra output in the terminal when running the `build_run.sh` file.
  Only outputs from Google Test and Nvidia Benchmark should be printed. If you use `cout` or `printf` for debugging,
  make sure to remove them before submission.
  This may affect your grade negatively. Regrade requests due to extra printing will not be accepted.


## Descriptive Answers (TODO)
Typically there is no single correct answer/plot for the following questions. Rely on your thought process!

### Plot(s) 1: GPU profiligng results

TODO: make sure to reference the correct plot below

![Figure 1: ](plots/plot1.png)

Description: TODO: please provide details for your plot(s) here.


***

## 📚 Appendix

### Useful SLURM Commands

* `squeue -u <username>`: Check the status of your jobs.
* `scancel <job_id>`: Cancel a job.
* `sinfo`: Check the status of the cluster.
* `sacct -j <job_id>`: Check the status of a finished job.
* `scontrol show job <job_id>`: Check the details of a job.
* `sbatch <script_name>`: Submit a job.

### Useful Git Commands

* `git status`: Check the status of your repository.
* `git add <file_name>`: Add a file to the staging area.
* `git commit -m "commit message"`: Commit the changes in the staging area.
* `git push`: Push the changes to the remote repository.
* `git pull`: Pull the changes from the remote repository.
* `git log`: Check the commit history.
* `git checkout <branch_name>`: Switch to a different branch.
* `git branch`: Check the branches in your repository.
* `git merge <branch_name>`: Merge a branch into the current branch.
* `git remote -v`: Check the remote repository.
* `git clone <repository_url>`: Clone a repository.
* `git diff`: Check the differences between the working directory and the staging area.
* `git reset <file_name>`: Unstage a file.
* `git reset --hard`: Discard all changes in the working directory and the staging area.

### Additional Metrics for Nvidia Profiler
you can use  the below command to find the list of metrics:   
```
ncu --list-metrics
```

Once you selected the metric, you can add it to the profiler command line arguments:
```
ncu --metrics metric1,metric2,...
```
### Local setup 

*Profiler* you may need to follow the instructions [here](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)
