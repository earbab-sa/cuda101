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

## 3. CUDA Ecosystem

## 4. Tooling


Wrap Up: 
Keep note of all the new terminology you encountered in this page and make sure you are comfortable describing each in a sentence or two after you completed the modules. 
