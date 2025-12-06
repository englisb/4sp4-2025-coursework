[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/TtqHArU_)
# 🎓 Project



By the end of the project, you will be able to:
* Optimize and design neural networks.
* Optimize and design sparse neural networks.
* Implement and optimize matrix multiplication.


**Note:** You must review Tutorial 01 before starting this lab. You will need to 
finish tutorial 02 for the GPU part.
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
 zuse a **Personal Access Token (PAT)** instead of your password.


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
Deep neural networks (DNN) consist of multiple layers that perform matrix multiplications 
followed by nonlinear operations.
In each layer, the input is multiplied by a weight matrix and then passed through a 
nonlinear activation function (for example, softmax).
The output of a layer becomes the input to the next layer.
Matrix multiplication is the most computationally expensive operation in neural networks.
Your main task is to implement a three-layer neural network.
Implement both dense and sparse variants of the network and compare their accuracy 
and performance against state-of-the-art implementations.



### DNN Description:
A DNN with two linear layers and two non-linear activation functions. 

#### Mathematical Representation:

Let's denote:

`X:` Input vector (n-dimensional)

`W1:` Weights matrix for the first layer (h x n)

`b1:` Biases vector for the first layer (h-dimensional)

`W2:` Weights matrix for the second layer (m x h)

`b2:` Biases vector for the second layer (m-dimensional)

`H:` Hidden layer activations (h-dimensional)

`Z:` Output vector before softmax (m-dimensional)

`Y:` Output vector (m-dimensional)

The forward pass of a DNN with two layers and softmax can be represented as follows:

`H = tanh(X * W1^T + b1)`

`Z = sigmoid(H * W2^T + b2)`

`Y = argmax(Z)`

Where:


tanh is an activation function, defined as:
`softmax(Z) = (exp(Z) - exp(-Z)) / (exp(Z) + exp(-Z))`.
sigmoid is another commonly used activation function in neural networks, defined as `1/(1 + exp(-z))`. argmax returns the index of the maximum element in the vector.

Explanation:

*Input Layer:* The input data is fed into the input layer.

*Hidden Layer:* The input is multiplied by the weights of the first layer, and the biases are added. The result is passed through the tanh activation function.

*Output Layer:* The activations from the hidden layer are multiplied by the weights of the second layer, and the biases are added. The result is passed through the sigmoid activation function to normalize the outputs into a probability distribution.

*Prediction:* Eventually argmax is used to predict the class label based on the output vector Z.


We have already trained the weight matrices and biases for this network on the MNIST dataset.
The weights and biases are included in the `data` folder; they are downloaded from the 
server when you run the provided `build_run.sh` script. See that script for details.
All model parameters are stored as CSV files in the `data/model/` folder.
The MNIST dataset is also provided in the `data` folder as a CSV file.
Functions for reading these CSV files are provided.
The MNIST CSV contains features and labels; labels are located in the first column after loading.

**Note:** The code assumes the data is in the `data` folder. If you set up the project in a 
different location than the server, update the paths accordingly.



### Task 1: Dense Neural Network

- Write a `Python` script (`script/dense_nn.py`) that implements the dense neural network described above. Measure and report the network 
accuracy on the `MNIST` dataset.
- Re-implement the same network in `C++` using your optimized matrix-matrix (`MM`) and matrix-vector (`MV`) operations. 
Measure execution time of the key calls in the dense NN (for example, `MM`, `MV`, and activation functions).
  test only the first 10 features of the `MNIST` dataset.
- Before the end-to-end C++ implementation, develop and optimize the `MM` and `MV` kernels. For `MV` benchmarking.
- Document the optimization strategies applied to each operation and provide an analysis of the resulting performance improvements.


### Task 2: Sparse Neural Network
- Implement a pruning algorithm to sparsify the weight matrices (in `script/sparsify_weight.py`). Magnitude\-based pruning 
(removing weights with the smallest absolute values) is acceptable.

- Write a `Python` script to prune `W1` and `W2` for sparsity levels from 50\% to 95\% in steps of 5\%. 
For each sparsity level, save the pruned weight matrices in dense CSV format using the naming convention: 
`<sparsity_level>_W1.csv` and `<sparsity_level>_W2.csv` (for example, `80_W1.csv` and `80_W2.csv`). 
When loading these files in `C++`, convert the dense matrices to a sparse format (e.g., CSR).

- Implement efficient sparse matrix\-matrix (SpMM) and sparse matrix\-vector (SpMV) kernels. Document the optimization 
strategies for each kernel and provide an analysis of the measured performance improvements.

- After validating SpMV and SpMM, implement the sparse neural network using these operations. Report accuracy and 
runtime for each sparsity level and compare the results with the dense neural network.

### Task 3: GPU Implementation 
Implement both dense and sparse neural networks on the GPU. Use CSR format for sparse matrices on the GPU.
First, implement optimized matrix-matrix (MM) and matrix-vector (MV) kernels on the GPU. Apply common GPU 
optimizations such as memory coalescing, tiling/shared memory, warp-level primitives, proper thread/block configuration, 
asynchronous copies, and tensor cores where applicable.
Then build the dense and sparse neural networks using these kernels and compare their performance on the GPU.

**Important Note** To enable GPU in the project, you will need to use `-DGPU_ENABLED=ON` when calling CMake. 


### Bonus Task: New Pruning Method and Outperforming vendor libraries
* Implement a new pruning method that outperforms magnitude-based pruning in terms of performance while maintaining 
similar accuracy. You may use any method.
* Compare the performance of your dense and sparse neural networks with vendor libraries such as Intel MKL (CPU) and 
cuBLAS/cuSPARSE (GPU). If you outperform these libraries, you will receive bonus points: target 1.2× faster than MKL 
(sparse NN vs dense MKL) and 1.1× faster than cuBLAS (sparse NN vs dense cuBLAS).

### The expected output
Your submission must include the following:

* Implementing dense NN using MM and MV operations (CPU and GPU).
* Implementing sparse NN using SpMM and SpMV operations for different sparsity levels from 50% to 95% with 
a step size of 5% (CPU and GPU).
* Bonus: build a new pruning method that outperforms magnitude-based pruning in terms of performance and provides a similar accuracy.
* Outperforming cuBLAS/cuSPARSE and Intel MKL for dense and sparse NN implementations (bonus).
* Provide stacked bar plots for different optimizations applied to MM/MV and SpMM/SPMV operations. 
  The x-axis should represent different matrix sizes (for MM) and different sparsity levels (for SpMM). 
  The y-axis should represent the execution time. Each bar should be divided into different colors representing 
  different optimizations applied (e.g., naive, cache blocking, vectorization, parallelization, etc.). 
  Provide separate plots for MM/MV and SpMM/SpMV.
* Repeat all above plots for GPU code as well.
* The necessary Python scripts (and packages) to generate all plots. You won't be allowed to push plots to the repo. 
  We should be able to generate all plots using your scripts and the logs generated from running your 
  code on the ECE server.
* four benchmarks are provided in the main directory:
  * `main_bench.cpp`: contains benchmarks for MM/MV/SpMM/SpMV on CPUs using Google Benchmark.
  * `main_bench.cu` : contains benchmarks for MM/MV/SpMM/SpMV on GPUs using NVBench.
  * `nn_cpu_bench.cpp`: contains benchmarks for dense and sparse neural networks on CPUs using Google Benchmark.
  * `nn_gpu_bench.cu`: contains benchmarks for dense and sparse neural networks on GPU.
  
   
### Evaluation
Your lab submission will be graded based on these criteria:

* Your code compiles and runs successfully on the ECE cluster using the provided `build_run.sh` script (Pass/Fail).
* All grading tests are passed as expected.
* The quality of your plots and its analysis, which must:
  * Convey a correct and clear argument.
  * Be visually understandable with proper labels, units, and a concise 1-2 sentence executive summary.
* All plots/model parameters must be generated using your provided Python scripts and the logs from your code runs on the ECE server. 
Plots pushed directly to the repository will not be accepted. 
  

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
* All logs should be redirected to `logs/` directory, all model parameters should be saved in 
`data/model`, and all plots should be saved in the `plots/` directory. These directories are not 
tracked by git to avoid pushing log files or plots to the repository. 

  
## Descriptive Answers
Typically, there is no single correct answer/plot for the following questions. Rely on your thought process!

**Important Notes:**
- All timing measurements use **real time (wall-clock time)** from Google Benchmark, not CPU time. CPU time sums across all cores and may not accurately reflect parallel performance.
- Input pixels are normalized to the 0-255 range before feeding into the neural network.
- Sparse NN accuracy targets vary by sparsification method and ratio; 80% is a reasonable baseline, though results may be higher or lower depending on pruning strategy.
- All matrix operations (GEMM, GEMV, SpMM, SpMV) are tested up to 4096x4096 matrices.
- Dense NN implementations include both GEMM-based (batch size 10 and full dataset) and GEMV-based (batch size 10, called 10 times) variants for performance comparison.

### Plot(s) 1: CPU MM/MV Performance Analysis (Task 1)

**GEMM Performance:**

![GEMM GFLOPs](plots/gemm_gflops.png)
*Description:* This plot shows GEMM performance in GFLOPs across different matrix sizes, demonstrating how computational throughput scales with problem size and optimization level.

![GEMM Median Runtime](plots/gemm_median.png)
*Description:* Median runtime comparison for GEMM across matrix sizes, showing wall-clock execution time improvements in naive and tiled implementations.

![GEMM Optimization Runtime](plots/gemm_optimization_stacked_runtime.png)
*Description:* Stacked bar plot showing runtime breakdown by optimization technique (naive, SIMD, parallelization, tiling) for different matrix sizes, revealing that tiled SIMD parallel implementations achieve 40-50x speedup over naive baseline.

![GEMM Optimization Throughput](plots/gemm_optimization_stacked_throughput.png)
*Description:* Stacked bar plot displaying throughput (operations per second) for each optimization level, demonstrating how cache blocking and vectorization combine to maximize computational efficiency.

**GEMV Performance:**

![GEMV GFLOPs](plots/gemv_gflops.png)
*Description:* GEMV GFLOPs performance across vector/matrix sizes, showing inconsistent absolute throughput due to vector size bottlenecks.

![GEMV Median Runtime](plots/gemv_median.png)
*Description:* Wall-clock median runtime for GEMV operations, displaying how naive baseline scales with problem size.

![GEMV Optimization Runtime](plots/gemv_optimization_stacked_runtime.png)
*Description:* Stacked runtime breakdown showing how vectorization and parallelization reduce GEMV execution time, with peak improvements around 10x for optimized versions.

![GEMV Optimization Throughput](plots/gemv_optimization_stacked_throughput.png)
*Description:* Throughput comparison across optimization levels for GEMV, demonstrating memory bandwidth utilization improvements through SIMD and parallel processing.

**Implementation Justification:**

Matrix multiplication operations—specifically general matrix-matrix multiplication (GEMM) and general matrix-vector multiplication (GEMV)—form the computational backbone of dense neural networks and dominate the execution time of forward pass operations. In the dense neural network architecture described for this task, each layer performs either a matrix-matrix or matrix-vector product followed by a bias addition and activation function. For the hidden layer computation `H = tanh(X * W1^T + b1)`, the multiplication `X * W1^T` constitutes a GEMM operation when processing batches of inputs or a GEMV operation for single-input inference. Similarly, the output layer `Z = sigmoid(H * W2^T + b2)` involves another GEMM/GEMV kernel. Given that these operations account for over 90% of the total floating-point operations in the network, optimizing MM and MV kernels is essential to achieving high-performance dense neural network implementations on both CPU and GPU architectures.

**Why Naive MM/MV Implementations Are Inefficient:**

Naive implementations of matrix multiplication suffer from fundamental performance bottlenecks rooted in poor memory hierarchy utilization. A straightforward three-nested-loop GEMM implementation exhibits extremely poor cache behavior: each element of the result matrix requires an entire row of matrix A and an entire column of matrix B to be accessed. For large matrices (e.g., 4096×4096), these memory accesses far exceed cache capacity, resulting in exceptional cache miss rates. Modern CPUs feature a memory hierarchy with L1 cache (~32KB, 4-cycle latency), L2 cache (~256KB, 12-cycle latency), L3 cache (~25MB shared, 40-cycle latency), and DRAM (~16GB, 200+ cycle latency). When data resides in DRAM rather than cache, the processor stalls waiting for memory, resulting in arithmetic intensity far below the theoretical peak of modern CPUs. The GEMM Median Runtime plot clearly demonstrates this inefficiency: naive implementations exhibit execution times orders of magnitude higher than optimized versions, confirming that memory latency, not computational throughput, limits performance.

**Loop Reordering and Memory Access Patterns:**

The first optimization applied to MM/MV kernels is loop reordering to improve spatial and temporal locality. In naive row-major GEMM implementations, accessing matrix B column-wise results in non-contiguous memory accesses with stride equal to the matrix width, causing cache line waste and excessive memory traffic. By reordering loops to traverse matrices in row-major order (i-k-j or k-i-j loop orderings), we ensure that consecutive memory locations are accessed sequentially, maximizing cache line utilization. Each 64-byte cache line loads 16 consecutive floats; with proper loop ordering, all 16 values are used before eviction, whereas strided access may use only 1 value per cache line, wasting 15/16 of memory bandwidth.

**SIMD Vectorization and Instruction-Level Parallelism:**

Modern CPUs provide SIMD (Single Instruction, Multiple Data) instruction sets such as AVX2 (256-bit vectors processing 8 floats simultaneously) and AVX-512 (512-bit vectors processing 16 floats simultaneously). Naive scalar code performs one floating-point operation per instruction; SIMD vectorization increases this to 8 or 16 operations per instruction, directly multiplying computational throughput. For GEMM operations, the innermost loop can be vectorized to compute 8 dot-product contributions simultaneously using AVX2 `_mm256_fmadd_ps` fused multiply-add instructions, which perform both multiplication and addition in a single cycle. The GEMM GFLOPs plot demonstrates the impact of SIMD: vectorized implementations achieve 4-8× higher GFLOPs than scalar code, approaching theoretical peak performance for the CPU's floating-point units. For GEMV operations, SIMD benefits are more modest due to memory-bound characteristics (discussed below), but the GEMV GFLOPs plot still shows 2-4× improvements from vectorization, confirming that SIMD is essential for maximizing floating-point throughput even in memory-bound kernels.

**Cache Blocking (Tiling):**
Cache blocking (tiling) partitions large matrix multiplications into smaller sub-problems (tiles) that fit within cache, but the choice of tile size is critical and non-obvious. While tiling fundamentally improves data reuse, selecting suboptimal tile sizes can result in minimal or even negative performance gains. For example, an 8x8 or 16×16 tile underutilizes the L1 cache's 32KB capacity, failing to exploit full cache bandwidth. Conversely, a large (relative to problem size) tile like 64×64 could exceed L1 capacity and force evictions to L2, increasing memory latency. The GEMM Optimization Stacked Runtime plot demonstrates this trade-off: naive tiling with poorly chosen sizes may provide only slight speedup or even worse, whereas careful tuning squeezes out the last capital. Additionally, tile size interacts complexly with multithreading: if each thread processes an independent tile, too-small tiles create excessive synchronization overhead across threads, while too-large tiles cause uneven load distribution if matrix dimensions are not perfectly divisible by tile size. Thus, cache blocking is powerful but requires empirical tuning; blindly applying tiling without architecture-specific optimization can result in disappointing speedups or even slowdowns compared to simple vectorized implementations, highlighting that performance optimization demands careful attention to hardware characteristics and experimental validation.

**Multithreading and Thread-Level Parallelism:**

The ECE cluster nodes feature 20-core Intel CPUs, providing substantial thread-level parallelism that must be exploited to achieve peak performance. OpenMP parallel directives partition matrix tiles across threads, allowing independent cores to compute different blocks simultaneously. For GEMM operations on large matrices, this parallelization scales nearly linearly up to 20 threads, as each thread operates on independent data with minimal cache coherence overhead. The GEMM Median Runtime plot shows that multithreaded implementations reduce execution time proportionally to thread count, confirming efficient scaling. For GEMV operations, parallelization benefits are more limited due to lower arithmetic intensity, but the GEMV Median Runtime plot still demonstrates healthy speedups from multithreading (with SIMD), indicating that even memory-bound operations benefit from distributing memory bandwidth across multiple cores.

**GEMV Performance Characteristics and Memory-Bound Behavior:**

GEMV operations exhibit fundamentally different performance characteristics than GEMM due to lower arithmetic intensity. For a GEMV operation computing `y = A * x` with A being N×N, each element of A is accessed once per result vector, creating an inherent memory-bound bottleneck. The GEMV GFLOPs plot reflects this: while absolute throughput numbers appear high due to measurement granularity at such fast timescales, the practical performance is memory-bandwidth-limited. However, the GEMV Median Runtime plot shows that optimization still provides substantial benefits by reducing memory access latency through vectorized loads and efficient prefetching. The GEMV Optimization Stacked Runtime plot demonstrates measurable improvements from SIMD vectorization and parallelization, confirming that even memory-bound operations benefit from proper optimization to maximize bandwidth utilization. The distinction between GEMM and GEMV performance highlights why GEMM is preferred for batch processing in the neural network: GEMM's higher arithmetic intensity enables compute-bound performance, delivering substantially faster execution than GEMV-based approaches.

**Justification for Dense Neural Network Implementation:**

The dense neural network forward pass consists of two primary GEMM/GEMV operations (hidden layer and output layer), two bias additions, and two activation functions. By implementing GEMM/GEMV kernels that achieve efficient memory utilization, the overall neural network achieves high-speed forward inference on batch inputs. The benchmark evidence demonstrates that cache tiling, SIMD vectorization, and multithreading combine to transform matrix operations from memory-bandwidth-limited to compute-efficient, enabling rapid neural network evaluation. The C++ implementation maintains 91% accuracy matching the Python baseline, confirming numerical correctness throughout the optimization process. For batch size 10 processing (as specified in Task 1), GEMM-based implementations leverage data reuse to sustain near-peak throughput, justifying the choice of GEMM over GEMV as the primary computational primitive.

### Plot(s) 2: CPU Sparse Operations and NN Analysis (Task 2)

**SPMM Performance:**

![SPMM GFLOPs](plots/spmm_optimization_gflops.png)
*Description:* SpMM throughput (GFLOPs) across different sparsity levels (50-95% in 5% steps), showing consistent performance as sparsity increases .

![SPMM Optimization Runtime](plots/spmm_optimization_runtime.png)
*Description:* Stacked bar plot showing SpMM runtime across sparsity levels, revealing how optimized implementations perform as non-zero elements decrease.

**SPMV Performance:**

![SPMV GFLOPs](plots/spmv_optimization_gflops.png)
*Description:* SpMV throughput across sparsity levels, showing unchanging computational efficiency gains as matrix sparsity increases.

![SPMV Optimization Runtime](plots/spmv_optimization_runtime.png)
*Description:* Optimization strategy comparison for SpMV across sparsity ranges, demonstrating no runtime improvements with sparse formats on the CPU.

**Sparse Neural Network Analysis:**

![NN Sparsity Runtime](plots/nn_sparsity_runtime.png)
*Description:* End-to-end sparse NN runtime versus sparsity level (50-95%), showing ~2x speedup at 95% sparsity, validating the performance benefits of magnitude-based pruning.

![NN Sparsity Accuracy](plots/nn_sparsity_accuracy.png)
*Description:* Accuracy versus sparsity plot revealing the accuracy-performance trade-off: accuracy decreases as sparsity increases, with different sparsity levels showing the balance between model compression and prediction quality.

**Implementation Justification:**

**A. Pruning & Sparsification Rationale:**
Magnitude-based pruning is employed as the primary sparsification strategy due to its simplicity, effectiveness, and minimal accuracy degradation at moderate sparsity levels. This approach removes weights with the smallest absolute values, preserving the most salient connections in the network. The choice to explore sparsity levels from 50% to 95% in 5% increments enables systematic evaluation of the accuracy-performance trade-off curve. Lower sparsity levels (50-70%) preserve most network capacity and maintain near-baseline accuracy, while higher levels (80-95%) degrade accuracy. Pruned matrices are saved as dense CSV files during the preprocessing phase for compatibility with existing infrastructure and ease of visualization, but converted to CSR (Compressed Sparse Row) format during C++ execution to exploit structural sparsity and eliminate zero multiplications. Performance benefits only emerge after converting to sparse formats: dense storage and computation provide no advantage over the optimized dense kernels from Task 1, but CSR representation enables dramatic acceleration by skipping zero elements entirely.

**B. CSR Format Justification:**

The Compressed Sparse Row (CSR) format is chosen for SpMV and SpMM operations because it provides three critical advantages for neural network inference. First, non-zero values are stored contiguously in memory, eliminating the memory waste and cache line pollution of coordinate formats. Row indices are also contiguous and accessed sequentially during computation, maximizing CPU prefetching effectiveness and cache line utilization—each 64-byte cache line loads 16 consecutive float values that are all used in nearby computations. Second, row-wise traversal in CSR perfectly matches the access pattern required by matrix multiplication, where each thread processes independent rows without synchronization, enabling efficient parallelization. Third, CSR's compact representation reduces memory traffic: a dense 256×256 matrix with 50% sparsity requires 256×256×4 = 262KB, whereas CSR requires only row/column indices plus non-zero values (~131KB), reducing memory bandwidth demands by approximately 50%. The benchmark data confirms this advantage, evident in the higher GFLOP throughput by orders of magnitude oveer dense formats.

**C. Sparse Kernel Optimization Justification:**

SpMV and SpMM kernels leverage multiple optimization strategies to accelerate sparse operations beyond what dense kernels provide. The fundamental advantage is skipping zero multiplications: in the standard dense operation, every element is processed regardless of value; in sparse CSR kernels, only non-zero elements participate in computation. Vectorization of non-zero value blocks improves arithmetic throughput by processing consecutive non-zero elements with AVX2 SIMD instructions, allowing multiple multiply-add operations to execute in parallel. However, vectorization is more limited in sparse contexts due to irregular memory access patterns in the column indices, creating data dependencies that prevent full SIMD utilization. Multithreading handles irregular row lengths by assigning independent rows to different threads; since each thread accesses disjoint memory regions (different rows), synchronization overhead is minimal and scaling remains nearly linear. Sparse tiling differs from dense blocking: rather than fitting tiles into cache, sparse tiling groups rows to improve memory access locality and reduce TLB misses on the column index array. Branchless inner loops are critical for performance: rather than checking "if value is non-zero," the kernel iterates over only stored values, avoiding pipeline stalls from mispredicted branches.

**D. Sparse Neural Network Performance and Accuracy Trade-off:**

The NN Sparsity Runtime plot shows end-to-end sparse neural network forward-pass acceleration across sparsity levels: sparse inference becomes dramatically faster as pruning increases. This acceleration comes from two sources: (1) reduced non-zero elements in weight matrices, eliminating multiplications via CSR representation, and (2) reduced memory traffic from smaller working sets, improving cache efficiency. The NN Sparsity Accuracy plot reveals the accuracy-performance trade-off: the network maintains near-baseline accuracy at 50% sparsity, and exhibits steeper accuracy loss at 80%+ sparsity, indicating that too-aggressive pruning removes critical weight pathways. The curve validates magnitude-based pruning: it removes the least-important weights first, preserving salient connections and enabling gradual accuracy degradation rather than sharp collapse. Dense neural networks remain slower at low sparsity and improve linearly with matrix sparsity.

### Plot(s) 3: GPU Performance Analysis (Task 3)

**GPU GEMM Performance:**

![GPU GEMM GFLOPs](plots/gemm_gflops_gpu.png)
*Description:* GPU GEMM throughput in GFLOPs across matrix sizes up to 4096x4096, demonstrating significant throughput improvements over CPU implementations through GPU parallelism.

![GPU GEMM Median Runtime](plots/gemm_median_gpu.png)
*Description:* Wall-clock median runtime for GPU GEMM showing execution times across different matrix sizes.

**GPU GEMV Performance:**

![GPU GEMV GFLOPs](plots/gemv_gflops_gpu.png)
*Description:* GPU GEMV throughput demonstrating performance characteristics for vector-matrix operations on GPU.

![GPU GEMV Median Runtime](plots/gemv_median_gpu.png)
*Description:* GPU GEMV wall-clock runtime comparison across different problem sizes.

**GPU Sparse Operations:**

![GPU SPMM GFLOPs](plots/spmm_optimization_gflops_gpu.png)
*Description:* GPU SpMM performance across sparsity levels in CSR format, showing throughput improvements as sparsity increases.

![GPU SPMM Runtime](plots/spmm_optimization_runtime_gpu.png)
*Description:* GPU SpMM runtime versus sparsity levels, demonstrating how execution time varies with matrix sparsity.

![GPU SPMV GFLOPs](plots/spmv_optimization_gflops_gpu.png)
*Description:* GPU SpMV throughput across different sparsity levels.

![GPU SPMV Runtime](plots/spmv_optimization_runtime_gpu.png)
*Description:* GPU SpMV runtime showing performance scaling with sparsity.

**GPU Neural Networks:**

![GPU NN Sparsity Runtime](plots/nn_sparsity_runtime_nn_gpu.png)
*Description:* End-to-end GPU sparse NN runtime across sparsity levels showing how performance scales with different pruning ratios.

**Implementation Justification:**

**A. Why GPU Implementation is Essential:**

The progression from CPU-optimized dense operations (Task 1) to sparse operations (Task 2) to GPU implementations (Task 3) represents the final stage of performance optimization in the neural network acceleration pipeline. While the 20-core CPU achieves impressive absolute performance through SIMD vectorization, parallelism, and cache optimization, modern deep learning demands orders-of-magnitude higher throughput to serve real time inference at scale. The NVIDIA Ada GPU architecture provides 21 Streaming Multiprocessors (SMs) with 128 CUDA cores each (2,688 total), each executing independent threads in Single-Instruction Multiple-Thread (SIMT) fashion. This represents 2,688/20 ≈ 135× more threads than the CPU can execute simultaneously, enabling massive data parallelism that far exceeds CPU capabilities. Beyond thread count, the Ada GPU offers 960 GB/s global memory bandwidth compared to the CPU's estimated 100–200 GB/s, substantially accelerating memory-intensive operations like GEMM and SpMM. The GPU implementation preserves the mathematical equivalence of the dense and sparse neural networks from Tasks 1 and 2, replacing only the computational backend to leverage GPU parallelism. Dense operations (GEMM/GEMV) benefit from the GPU's high arithmetic throughput and memory bandwidth, achieving substantially higher GFLOPs than CPU implementations. Sparse operations (SpMM/SpMV) additionally benefit from parallel processing of multiple rows simultaneously and efficient memory access patterns through CSR format, where non-zero elements are processed in bulk across threads.

**B. Dense GPU MM/MV Kernel Design and Memory Optimization:**

The dense GEMM kernel on GPU is structured around the principle of maximizing arithmetic intensity and memory coalescing. The kernel employs a 2D thread block organization (e.g., 32×32 threads per block) where each thread computes a small tile of the output matrix C, typically 4×4 or 8×8 elements. For a 32×32 thread block computing C, the grid is configured as `grid(ceil(M/256), ceil(N/256))`, where each block computes a 256×256 output tile. Within the block, threads are indexed as `threadIdx.y` (row) and `threadIdx.x` (column), with each thread (i, j) computing C[blockIdx.y×256 + threadIdx.y×8, blockIdx.x×256 + threadIdx.x×8] through a series of dot products. Memory coalescing is achieved through careful thread indexing: when a warp of 32 threads (threads 0–31 in the same half-warp) access global memory, hardware merges accesses to nearby addresses into minimal transactions. By ensuring that threadIdx.x is the innermost loop dimension and memory addresses are contiguous along this axis, consecutive threads in a warp load consecutive elements from matrices A and B. For example, loading a row of matrix A into registers requires threads (i, 0), (i, 1), ..., (i, 31) to load addresses A[...][0], A[...][1], ..., A[...][31]—perfect coalescing where all 32 addresses fall in the same 128-byte cache line (4 floats × 32 threads × 4 bytes). In contrast, uncoalesced access patterns (e.g., column-wise loads) would require 32 separate transactions, wasting 31/32 of memory bandwidth.

Shared memory tiling dramatically improves arithmetic intensity by reducing global memory traffic. The kernel loads tiles of A and B into shared memory before computing partial results. For example, the innermost loop over K iterations loads a 32×32 block of A and a 32×32 block of B (2,048 bytes each) into shared memory, allowing 2,048 elements of each matrix to be accessed from fast shared memory (96 KB per block, ~100 GB/s bandwidth) rather than slow global memory (960 GB/s peak but with higher latency). Each loaded element is reused 32 times (dot product of 32 elements), achieving arithmetic intensity of 32 multiplications ÷ 2 loads = 16 FLOPs per element loaded. In scalar global memory access, each element is loaded once and used once (intensity = 1 FLOP/element), 16× worse. The tiling strategy parallels Task 1's cache blocking: instead of the CPU's L1/L2/L3 caches, the GPU uses software-managed shared memory, providing explicit control over data locality. Thread cooperation within the block is essential: all 1,024 threads (32×32) synchronize via `__syncthreads()` after loading tiles to ensure all data is resident before computation begins. Warp-level behavior is structured to minimize divergence: all warps in a block follow the same loop structure and control flow; divergence occurs only in sparse kernels (Section C) where different threads may iterate different numbers of times. For dense GEMM, uniform thread behavior ensures all 32 threads per warp execute the same instructions simultaneously, achieving full utilization.

Block configuration balances occupancy, shared memory, and register pressure. A 32×32 block uses 1,024 threads, which exceeds the typical 768-thread target for maximum occupancy on Ada (achieving 2 blocks per SM = 2,048 threads). Shared memory requirements are 32×32×4 bytes × 2 (A and B tiles) = 8,192 bytes, well within the 96 KB per-block limit. Register pressure is managed by limiting each thread to compute at most 8×8 output elements (64 registers for C, plus A/B tile indices, plus loop counters ≈ 100 registers per thread), leaving headroom to avoid register spilling to slow local memory. Alternative configurations (e.g., 16×16 blocks with 4×4 output per thread) offer lower occupancy but higher instruction-level parallelism; experiments with the GPU NN benchmarks confirm that 32×32 provides good balance.

Dense GEMV (matrix-vector product) presents a different challenge: the operation inherently has low arithmetic intensity (one matrix row, one vector—arithmetic intensity ≈ M ÷ (M + N) ≈ 1). The GEMV kernel assigns rows to warps or blocks; each warp computes a dot product of one row with the vector via parallel reduction. For example, a warp of 32 threads loads 32 elements of the vector and 32 elements of one matrix row, computes 32 products, then reduces to a single result via `warp_reduce_sum()` using warp shuffles. Warp shuffles (`__shfl_sync`) are faster than shared memory reductions for this scale (32 elements), executing in log₂(32) = 5 cycles instead of multiple rounds of shared memory writes and synchronization. Multiple warps process different rows in parallel, with thread blocks handling batch operations or multiple matrix-vector products. While GEMV achieves lower absolute GFLOPs than GEMM (memory-bound vs. compute-bound), GPU parallelism still provides substantial speedup: processing 128 rows simultaneously via 4 warps per block × (SMs/occupancy) blocks achieves far higher throughput than sequential CPU execution, even at lower per-thread utilization.

Tensor cores (Nvidia's specialized FP32/TF32 units) accelerate GEMM on Ada: each SM has 8 tensor cores per warp, enabling 8×8×16 matrix multiplies in a single instruction. Tensor cores can multiply two 16×16 FP16 matrices and accumulate into a 16×16 FP32 result, delivering 2×16×16 = 512 FLOPs per instruction (vs. 32 for scalar FMA). For full GEMM 1024³, replacing FP32 scalar operations with TF32 tensor cores (executing FP32 inputs with reduced precision accumulation) can accelerate computation by ~4–8× with minimal accuracy loss. However, tensor cores require specific matrix layouts and block sizes; they are most effective for batch GEMM or large matrices where communication overhead is amortized. In this project, Task 3 focuses on optimized scalar CUDA kernels with coalescing and shared memory; tensor cores represent a secondary optimization path beyond the scope of the core requirements.

**C. Sparse GPU Implementation (CSR SpMV/SpMM):**

Sparse operations on GPU face a fundamental challenge: irregular memory access patterns and load imbalance due to varying row densities. CSR format stores non-zero values contiguously, row pointers indicating where each row's values begin, and column indices mapping values back to matrix columns. For SpMV `y = A × x`, the kernel iterates through each row's non-zero values, accumulates products, and writes the result. The key difference from dense SpMV is that only non-zero elements are processed: a thread processing a sparse row with 100 non-zeros (80% sparsity in a 500-element row) computes 100 multiplications instead of 500, achieving 80% computation reduction.

SpMV kernel design must handle load imbalance: row nnz varies widely (some rows dense, others sparse), so assigning one row per thread causes many threads to idle while others work. Common strategies include: (1) **row-wise parallelism**: multiple warps per row for rows with many non-zeros, single thread per row for sparse rows; (2) **merge-path partitioning**: compute a binary tree of non-zero elements and assign path segments to threads, ensuring roughly equal work per thread; (3) **warp-cooperative**: 32 threads cooperate on a single row, dividing non-zeros among threads. This project employs warp-level parallelism: each warp (32 threads) processes multiple rows cooperatively. For each row, threads load non-zero values and column indices, compute products with corresponding vector elements, and reduce results via parallel reduction.

Memory coalescing in sparse kernels is challenging: column indices are irregular (may jump across the matrix arbitrarily), so loading `x[col[i]]` often causes poor coalescing. A mitigation strategy is to transform the algorithm: instead of SpMV as `y[i] = sum(A[i][j] * x[j])`, reformulate as column-wise accumulation: `for each non-zero A[i][j]: y[i] += A[i][j] * x[j]`. Since non-zero values are stored row-wise and threads process rows cooperatively, column accesses are scattered. However, atomic additions `atomicAdd(y[i], ...)` serialize these accesses. A better approach uses shared memory: threads accumulate partial sums for rows in a small batch (e.g., 32 rows), then atomically add to global memory, reducing contention. SpMM (sparse matrix × dense matrix) offers better parallelism: the output matrix is dense, so multiple threads can compute different output columns independently. Kernel design assigns threads to (row, col) pairs of output; each thread processes a row of the sparse matrix A and a column of the dense matrix B. This provides regular memory coalescing on the dense matrix B (multiple threads load the same columns) while handling sparse row accesses uniformly.

Load balancing in sparse kernels is addressed through careful thread assignment. Rather than assigning threads statically (thread i processes row i), dynamic scheduling based on non-zero counts can improve utilization. For example, a task queue tracks which rows have been processed; idle threads fetch the next unprocessed row from the queue. This ensures threads processing dense rows don't block sparse rows. Alternatively, binary search on row pointers determines which threads process which rows, ensuring each thread processes approximately the same number of non-zeros.

**D. GPU-Specific Optimizations:**

Shared memory in dense GEMM (as described in Section B) is critical: loading A and B tiles reduces global memory pressure and improves cache behavior. In sparse kernels, shared memory stores partial sums and temporary vectors. For SpMV, if multiple warps in a block process the same rows, shared memory accumulates contributions before atomic writes to global memory, reducing contention.

Warp-level primitives accelerate reduction operations. `__shfl_sync` enables fast communication within a warp without shared memory: `result = __shfl_down_sync(mask, value, offset)` shifts values across threads, allowing fast parallel reductions. For SpMV row reductions (32 threads summing dot products), warp shuffle reduces cost to 5 instructions instead of shared memory + synchronization.

Asynchronous copies (Nvidia's `__global__ void copy_async` or `cuda::memcpy_async`) overlap global memory loads with computation. For large GEMM, after computing one tile's result, the next tile is prefetched while the current tile computation proceeds, hiding memory latency. This technique is especially effective for bandwidth-limited kernels.

Occupancy measures active warps per SM: Ada has 128 warps per SM (4,096 threads). A 32×32 block (1,024 threads) uses 32 warps; with 2 blocks per SM, occupancy is 64/128 = 50%. Increasing occupancy to 75% (6 blocks of 256 threads each) hides more memory latency but requires lower register/shared memory per block. The GEMM kernel balances occupancy (via 32×32 blocks) with shared memory efficiency, achieving practical performance without excessive register spilling.

**E. Sparse GPU Optimization Effectiveness:**

The GPU sparse kernels show consistent speedup with increasing sparsity across all implementations. The SpMV runtime plot demonstrates that baseline implementation scales linearly with sparsity reduction (~50 µs at 50% sparsity down to ~10 µs at 90% sparsity), a 5× speedup. The coalesced and combined optimizations maintain comparable performance, indicating that the primary benefit comes from reduced non-zero element processing rather than memory access pattern improvements. The SpMM runtime plot shows similar trends: baseline drops from ~30,000 µs at 50% sparsity to ~4,000 µs at 90% sparsity (7.5× speedup), with shared memory optimization (orange) providing the largest improvement, reaching >80 GFLOPs at 90% sparsity compared to baseline's ~70 GFLOPs. This demonstrates that shared memory usage effectively reduces global memory pressure, allowing sparse kernels to sustain high throughput even with irregular access patterns. The consistent improvement across all sparsity levels shows that sparse matrix structure is efficiently exploited on GPU through warp-level parallelism and shared memory tiling.

### Plot(s) 4: Bonus - Advanced Pruning and Vendor Comparison

**Alternative Pruning Method:**

![Alternative Pruning Comparison](plots/bonus_pruning_comparison.png)
*Description:* Runtime and accuracy comparison between magnitude-based and structured/block-wise pruning showing 30-40% better performance at 80-90% sparsity while maintaining similar 80-85% accuracy through improved memory access patterns.

**Vendor Library Comparisons:**

![Vendor Library Comparison CPU](plots/bonus_vendor_comparison_cpu.png)
*Description:* CPU sparse NN performance versus Intel MKL dense operations across matrix sizes, demonstrating that optimized sparse implementations reach 75-85% of MKL performance with potential to achieve 1.2x speedup target through kernel fusion.

**Sparsity Bonus Implementation Analysis**
![Vendor Library Comparison GPU](plots/sparsity_comparison.png)
We compared our custom SparseGPT pruning method against the standard magnitude-based pruning at two sparsity levels (90% and 60%). The results from the plot are below:

| Method                      | Accuracy (%) | Runtime (ms) |
|-----------------------------|--------------|--------------|
| Magnitude-based (90%)       | 67.34        | 1338.37      |
| Magnitude-based (60%)       | 91.21        | 2620.50      |
| SparseGPT (90%)             | 87.86        | 1350.83      |

#### Key Observations

- **Accuracy**: SparseGPT at 90% sparsity achieves 87.86% accuracy, which is 20.52% higher than magnitude-based pruning at the same sparsity (67.34%). While magnitude-based pruning at 60% sparsity reaches slightly higher accuracy (91.21%), it does so with much lower sparsity (i.e., more weights retained).
- **Runtime**: SparseGPT (90%) runs in 1350.83 ms, nearly identical to magnitude-based (90%) and 48% faster than magnitude-based (60%), which takes 2620.50 ms.
- **Efficiency**: SparseGPT maintains high accuracy even at extreme sparsity (90%), whereas magnitude-based pruning suffers a dramatic accuracy drop at this level. To match SparseGPT’s accuracy, magnitude-based pruning must retain far more weights, resulting in much slower runtime.

#### Why SparseGPT Is Better

SparseGPT leverages second-order information (the Hessian) to identify and prune weights with minimal impact on model output, and compensates for pruned weights using an Optimal Brain Surgeon (OBS) update. This approach preserves the most salient weights and adjusts the remaining ones to maintain performance, allowing the network to remain accurate even when aggressively pruned.

Magnitude-based pruning, in contrast, simply removes weights with the smallest absolute values, ignoring their actual contribution to the output. At high sparsity, this leads to loss of critical connections and a sharp drop in accuracy. To maintain accuracy, magnitude-based pruning must keep more weights, which increases runtime and memory usage.

reference link: https://arxiv.org/abs/2301.00774

