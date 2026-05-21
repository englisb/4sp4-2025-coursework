[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/NumM623a)
# 🎓 Tutorial 01: Basics
This tutorial teaches you how to build and run your labs in CE 4SP4.

The main goals of this tutorial are:
* How to login to the ECE cluster
* How to clone the repository
* How to build and run the code on the ECE cluster
* How to develop your code and test it and measure its performance

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
git clone https://github.com/cheshmi/basics.git
```

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
cd tutorial-01-basics
```
All necessary instructions to build and run the code are provided in the `build_run.sh` file. 
You can open the file using a text editor like `nano` or `vim`, e.g., `vim build_run.sh`. 
Use the following command to build and run the code:

```
sbacth build_run.sh
```

This command submits the `build_run.sh` script to the SLURM scheduler. The instructions in the `build_run.sh` file are 
unix commands that will be executed on a compute node. The output of the job will be saved in a file named `tut01.<job_id>.<node_id>.out`, 
where `<job_id>` is the ID of the job assigned by SLURM. You can check the content of this file using the `cat`, `nano`, or `vim` commands.

* **Never use `bash`** to run your code on the login node. Use `sbatch` instead. `bash` runs the code directly on 
the login node, while `sbatch` submits the job to the SLURM scheduler, which runs it on an available compute node. 
* You should **never** run code on the login node unless it takes less than 30 seconds.
* You can check the status of your job using `squeue -u <username>`.
* See the appendix [Useful SLURM commands](README.md#useful-slurm-commands) for more information about `sbatch` and 
other useful commands.

### Local machine

While the ECE server is required for some parts of the course, it's often more convenient to work on your local machine 
for tasks like debugging. We recommend setting up the code on your choice of editor. We recommend **CLion**, 
a commercial editor from JetBrains. We provide a short guide on how to set up this tutorial (and can be used in most other labs). 
However, please remember that at some point, you may not be able to use your local machine due to not having a GPU or 
the proper library packages. You will then need to rely on the server. This is especially true starting from Lab 2, 
where architecture-specific code optimization will be needed.

***

## ✅ Tasks

You will need to start working on the TODO items described below:

### Task 1: Getting started with Google Test

You should write three test cases for the given vector operation to ensure the correctness of the implementation. 
An example test is provided in the `test_vector_op.cpp` file. **Google Test** is a C++ testing framework developed 
by Google. See more info [here](https://google.github.io/googletest/).

### Task 2: Getting started with Google Benchmark

You will need to add another implementation of the vector operation and compare the performance of the two 
implementations using **Google Benchmark**. Google Benchmark is a framework for benchmarking C++ code that we will use 
in this course. One example of a Google Benchmark is provided in the `main_bench.cpp` file. 
Add another benchmark function for your implementation. The new benchmark should repeat 20 times and report 
the average time of 5 iterations. The benchmark should be run for vector sizes of 1000, 10000, and 100000.

* **Google Benchmark** is a C++ library to benchmark code. See more info [here](https://github.com/google/benchmark/blob/main/docs/user_guide.md).

### Evaluation
You will be evaluated based on the following criteria:
* Quality and number of the test cases 
* Performance comparison using Google Benchmark
* No report is required for this tutorial.

### Submission
* Enter your McMaster ID in `mac_id.txt`.
* You will need to push your code to the repository's main branch.
* Make sure to commit and push your code before the deadline.
* Make sure your code compiles and runs on the ECE cluster. If your code does not compile or run, you will get a zero for this tutorial. 
The TA only runs the `build_run.sh` file to build and run your code.
* There should not be any extra output in the terminal when running the `build_run.sh` file. 
Only outputs from Google Test and Google Benchmark should be printed. If you use `cout` or `printf` for debugging, 
make sure to remove them before submission. 
This may affect your grade negatively. Regrade requests due to extra printing will not be accepted.

  

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
