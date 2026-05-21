# 🎓 Tutorial 03: GPU Tensor Cores (not graded)
This tutorial teaches you how to use tensor cores in CE 4SP4.

The main goals of this tutorial are:
* implement a simple operation using tensor cores and half precision.


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
unix commands that will be executed on a compute node. The output of the job will be saved in a file named `tut03.<job_id>.<node_id>.out`,
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

There is no specific tasks. You can run the benchmark and see the performance 


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
