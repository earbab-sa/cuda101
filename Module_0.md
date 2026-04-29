## Module 0
Get familiar with the basics.

## Topics
1.  Why CUDA if I already use PyTorch / JAX / CuPy?    
2.  CPU vs GPU: throughput vs latency, SIMD/SIMT, data parallel workloads.    
3.  The CUDA ecosystem: CUDA Toolkit, drivers, libraries (cuBLAS, cuDNN, Thrust), profilers (Nsight), and Python bindings (PyCUDA, Numba, CuPy).    
4.  Course tooling: compiler (nvcc), VS Code or CLI, basic project structure.


## 1. Why CUDA? 
It gives you fine‑grained control, custom kernels, and a deeper understanding of performance that high‑level libraries can’t fully expose.     

### 1.1 Custom operations beyond the library  
High‑level frameworks cover common ops, but research often needs fused or non‑standard kernels that aren’t implemented yet. CUDA lets you 
implement exactly the operation you want, with the memory layout and parallelism you choose, instead of forcing your idea into the existing op set.

### 1.2 Performance and efficiency headroom     
PyTorch/JAX/CuPy already call CUDA under the hood, but they optimize for generality and usability, not your specific workload. Writing or tuning 
kernels in CUDA lets you control memory access patterns, fusion, tiling, and launch parameters to squeeze out performance that generic kernels can’t reach.

### 1.3 Understanding what your tools are doing
Knowing CUDA gives you a mental model for how GPU kernels, streams, and memory transfers behave, which improves how you structure high‑level code. It also 
helps you interpret profiler output and debug performance issues when your PyTorch/JAX code is “mysteriously slow” or under‑utilizing the GPU.

### 1.4 Interop and extending Python GPU stacks
Libraries like CuPy and Numba expose CUDA concepts (streams, raw kernels) and are often used as an intermediate “mid‑level” between pure Python 
and C++ extensions. When you know CUDA, you can move tensors between PyTorch, CuPy, and custom kernels (e.g., via DLPack) and implement high‑performance 
pieces without waiting on library updates.

### 1.5 Portability of expertise across systems
CUDA is the de facto standard for NVIDIA GPUs, so its execution and memory model underpins many higher‑level tools (PyTorch, JAX XLA backends, 
CuPy, Numba). The same knowledge transfers to other ecosystems (C++ HPC code, custom simulators, non‑Python stacks), which matters if you ever need to 
step outside Python or deploy in constrained environments.

## 2. CPU vs GPU
CPUs are optimized for low‑latency, flexible execution of a few threads, while GPUs are optimized for high‑throughput execution of massive numbers of 
similar operations using SIMD/SIMT on data‑parallel workloads.

### 2.1 Throughput vs latency
A CPU is good at “finish this complex task as fast as possible”; a GPU is good at “process a huge batch of similar tasks as fast as possible.”
- CPUs (latency‑oriented): Few powerful cores, high clock speeds, large caches, branch prediction, and sophisticated control logic to minimize the time for a single or small number of tasks.     
- GPUs (throughput‑oriented): Many simpler cores, lower clock speeds per core, smaller caches, and very wide parallel units designed to maximize total work per unit time when there are thousands of independent operations.
  
### 2.2 SIMD vs SIMT
- SIMD (Single Instruction, Multiple Data): Classic CPU vector units (SSE/AVX etc.) execute one instruction over multiple data lanes in a single core, with 
one program counter and explicit masking/packing of data.
- SIMT (Single Instruction, Multiple Threads) on GPUs: Hardware groups many lightweight threads into warps; each warp executes one instruction across threads
like SIMD, but each thread has its own logical state and program counter, and the GPU hardware manages divergence and reconvergence.       

Conceptually, SIMT is “threaded SIMD”: it scales to tens of thousands of threads and hides memory latency by quickly switching between warps while others wait on memory.     

### 2.3 Data‑parallel workloads
GPUs excel when:    
- There are many independent elements (large arrays, batches, grids).    
- Control flow is relatively uniform (few divergent branches).   
- Each element does enough work to amortize memory and launch overheads.     

CPUs, even with SIMD, tend to do better for
- Small problem sizes.   
- Heavily branched logic.
- Per‑task latency and complex control flow matter more than bulk throughput.

## 3. CUDA Ecosystem
The CUDA ecosystem is a stack: drivers at the bottom, the CUDA Toolkit on top, then high‑level libraries, tools, and Python bindings that you use directly. 
### 3.1 CUDA Drivers
NVIDIA’s GPU driver lets the OS and applications talk to the GPU and is required to run any CUDA program. It manages low‑level tasks like context creation, 
memory management, and kernel scheduling on the hardware.

### 3.2 CUDA Toolkit
The CUDA Toolkit is the developer package: compiler (nvcc), runtime libraries, headers, GPU‑accelerated math and algorithm libraries, and tools. It provides both low‑level (driver/runtime APIs) and higher‑level abstractions so you can write, build, debug, and optimize GPU code.

### 3.3 Core CUDA Libraries
- cuBLAS: GPU‑accelerated BLAS (dense linear algebra) for vector/matrix ops like GEMM; heavily used under the hood in ML and scientific codes.    
- cuDNN: Deep neural network primitives (convolutions, pooling, RNNs, etc.), powering many DL frameworks’ GPU backends.    
- Thrust: A C++ template library offering STL‑like algorithms (sort, reduce, scan, etc.) on GPUs, so you can write high‑level C++ instead of raw kernels for common patterns.
- Profilers and tools (Nsight family): Nsight Systems / Nsight Compute are NVIDIA’s profiling tools for timeline analysis, kernel performance, memory behavior, and bottleneck identification in CUDA applications. They integrate with the Toolkit to help you inspect kernel launches, GPU utilization, and interactions between CPU and GPU.     

### 3.4 Python Bindings and GPU‑Python Stacks
- PyCUDA exposes CUDA driver/runtime APIs in Python, letting you compile and launch custom kernels and manage device memory directly from Python.     
- Numba (CUDA target) uses a JIT compiler to turn decorated Python functions into CUDA kernels, giving a Pythonic path to custom GPU code without writing C++.    
- CuPy is a NumPy/SciPy‑like array library backed by CUDA; it wraps libraries such as cuBLAS and cuDNN and provides high‑level GPU arrays plus options for
  raw kernels when needed.

## 4. Tooling
### nvcc: the CUDA Compiler
```nvcc``` is NVIDIA’s compiler driver that splits host (CPU) and device (GPU) code in a ```.cu``` file and invokes the right host and device compilers. You compile CUDA 
sources with commands like ```nvcc -o myprog main.cu```, producing a normal executable that links against CUDA runtime libraries.

### VS Code or CLI
- CLI workflow: Use a terminal with ```nvcc``` on your ```PATH```, compile via ```nvcc``` commands or a Makefile/CMake build, then run your binary from the shell.    
- VS Code: Install C/C++ extensions, configure include paths and compiler path to point at the CUDA Toolkit, then use build tasks/launch configs so VS Code
calls ```nvcc``` and can provide IntelliSense and debugging.

### Basic CUDA Project Structure
Every CUDA program is “heterogeneous”: host code sets up data and launches kernels, and device code (in ```.cu```/```.cuh```) runs on the GPU using the triple‑chevron syntax and APIs like ```cudaDeviceSynchronize``` to coordinate with the CPU. A minimal CUDA project usually looks like this:    
- Source files:
  * ```src/main.cu``` – host ```main``` plus kernels, or
  * ```src/kernels.cu``` (device code) and ```src/main.cpp``` (host code) if you separate concerns.

- Headers:
  * ```include/kernels.cuh``` for kernel declarations and shared types.

- Build files:
  * ```CMakeLists.txt``` or ```Makefile``` that compiles ```.cu``` with ```nvcc``` and links CUDA libraries.

## Wrap Up: 
Keep notes of all the new terminology you encountered in this page and make sure you are comfortable describing each in a sentence or two after you completed the modules. 
