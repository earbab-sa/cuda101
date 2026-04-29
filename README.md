## CUDA 101
Introduction to CUDA programming for Python programmers.

### Audience
Strong Python programmers (NumPy, PyTorch/JAX/TF OK), basic C/C++ familiarity helpful but not required.

### Goal
By the end, they can write, profile, and optimize simple CUDA kernels, understand GPU performance basics, and reason about GPU execution even when working “from Python”.



Here’s a focused “CUDA 101 for Python Experts” syllabus aimed at CS PhD students who know Python, NumPy, and ML/DS frameworks but are new to CUDA/C++.

***

## Course framing

Audience:  
- Strong Python programmers (NumPy, PyTorch/JAX/TF OK), basic C/C++ familiarity helpful but not required.  
Goal:  
- By the end, they can write, profile, and optimize simple CUDA kernels, understand GPU performance basics, and reason about GPU execution even when working “from Python”.

Assume: 6–8 modules, each 2–3 hours (mix of lecture + hands-on).

***

## Module 0 – Logistics and mental model reset

**Objectives**

- Align expectations: CUDA as a mental model for what PyTorch/JAX/etc. already do under the hood.
- Clarify prerequisites and environment.

**Topics**

- Why CUDA if I already use PyTorch / JAX / CuPy?
- CPU vs GPU: throughput vs latency, SIMD/SIMT, data parallel workloads.
- The CUDA ecosystem: CUDA Toolkit, drivers, libraries (cuBLAS, cuDNN, Thrust), profilers (Nsight), and Python bindings (PyCUDA, Numba, CuPy).
- Course tooling: compiler (`nvcc`), VS Code or CLI, basic project structure.

**Assignments**

- Environment check: verify NVIDIA driver, CUDA Toolkit, and a simple “device query” program or Python GPU check (`torch.cuda.is_available()` or similar).
- Short reflection: “What GPU code do you indirectly run today via Python libraries?”

***

## Module 1 – CUDA execution model for Python minds

**Objectives**

- Build an intuition for threads, blocks, grids, and warps using analogies to NumPy broadcasting and vectorization.
- Understand what a kernel is and how it’s launched.

**Topics**

- Host (CPU) vs device (GPU) roles.
- Kernels as “vectorized functions” vs Python loops vs NumPy ufuncs.
- Thread hierarchy: threads, blocks, grids; warps as execution units.
- Mapping data indices to threads: 1D examples (arrays), link to `np.arange` / `np.vectorize`.

**Hands-on**

- Start from a high-level Python/NumPy vector-add.
- Show equivalent pseudo-CUDA kernel and launch configuration on slides.
- Small exercise: given a mathematical operation (e.g., saxpy: `y = a*x + y`), design how you’d map indices to threads (paper/pencil, no coding yet).

**Assignments**

- Worksheet: for given data shapes, propose grid/block sizes and index computations.

***

## Module 2 – Minimal CUDA C++: your first kernel

**Objectives**

- Get over the “C++ barrier” with a tiny, idiomatic but minimal CUDA C++ example.
- Compile and run a basic kernel end-to-end.

**Topics**

- Anatomy of a `.cu` file: includes, `__global__` kernel, `main`.
- Kernel launch syntax: `kernel<<<gridDim, blockDim>>>(args...)`.
- Thread indexing: `threadIdx`, `blockIdx`, `blockDim`, `blockIdx.x * blockDim.x + threadIdx.x`.
- Synchronization at the host level: `cudaDeviceSynchronize`.

**Hands-on**

- Live code: 1D vector addition with unified memory (`cudaMallocManaged`) to avoid early memory-copy complexity.
- Show side-by-side: Python/NumPy version vs CUDA kernel.
- Build and run with `nvcc`, print correctness checks.

**Assignments**

- Modify the vector-add kernel to implement:
  - Elementwise scalar multiply.
  - Elementwise ReLU and sigmoid (with attention to numeric stability in C++).  
- Simple written question: explain what happens if `N` is not divisible by `blockDim.x` and how to guard against out-of-bounds.

***

## Module 3 – Memory model and data movement (with Python analogies)

**Objectives**

- Understand host vs device memory and why data motion dominates performance.
- Learn basic CUDA memory APIs.

**Topics**

- Host memory vs device memory; PCIe as a “slow bus”.
- `cudaMalloc`, `cudaFree`, `cudaMemcpy`: directions (`HostToDevice`, `DeviceToHost`, `DeviceToDevice`).
- Unified memory vs explicit copies: trade-offs in simplicity vs control.
- Analogy to `.cpu()` / `.cuda()` in PyTorch, `device` placement in JAX.

**Hands-on**

- Convert the unified-memory example to explicit device allocations + copies.
- Time a simple loop:
  - CPU-only NumPy.
  - GPU with heavy copy overhead.
  - GPU with reused device buffers and minimized transfers.
- Plot timing from Python (e.g., `subprocess` calling the CUDA binary) to keep students in their comfort zone.

**Assignments**

- Implement a small pipeline:
  - Generate data on host.
  - Copy to device, run kernel, copy back, verify.
- Short questions:
  - When is unified memory acceptable?
  - How would you structure a training loop to minimize host–device transfers?

***

## Module 4 – 2D data and kernels (images, matrices)

**Objectives**

- Move from 1D to 2D indexing and build intuition for mapping matrices/tensors to thread grids.
- Prepare for matrix operations and basic ML kernels.

**Topics**

- 2D and 3D grids/blocks: `threadIdx.x/y`, `blockIdx.x/y`, `blockDim.x/y`.
- Mapping `(row, col)` indices to global thread IDs.
- Typical patterns for images and matrices.

**Hands-on**

- Implement a 2D kernel for:
  - Elementwise operations on a 2D array (e.g., brightness scaling of an image).
- Show equivalence to pure NumPy code operating on `H x W` arrays.
- Visual sanity check (e.g., load small image in Python, apply CUDA transform, display).

**Assignments**

- Implement:
  - Matrix transpose kernel (naive).
  - Elementwise binary op on two matrices (e.g., `C = A * B + bias`).
- Optional: discuss edge cases where matrix dimensions are not multiples of block sizes.

***

## Module 5 – Introduction to performance and memory access

**Objectives**

- Give a first taste of performance tuning without going too deep.
- Teach students to think about memory access patterns and occupancy.

**Topics**

- Coalesced vs non-coalesced global memory access.
- Register usage, shared memory basics (conceptual).
- Occupancy, block size selection heuristics.
- Very high-level look at Nsight Systems / Nsight Compute without expecting mastery.

**Hands-on**

- Compare naive vs slightly optimized kernels:
  - Example: vector add with different block sizes.
  - Example: 2D transpose naive vs tiled (introducing shared memory conceptually).
- Simple profiling from command line: measure kernel timing, use basic metrics.

**Assignments**

- Micro-benchmarks:
  - Vary block size and measure runtime, plot speed vs block size (using Python).
- Written prompt:
  - Given a set of access patterns, identify which ones are likely coalesced.

***

## Module 6 – From CUDA C++ to Python: Numba, CuPy, and custom ops

**Objectives**

- Connect low-level CUDA knowledge to practical Python workflows.
- Show how to write and call custom kernels directly from Python.

**Topics**

- Numba `@cuda.jit` and its similarities/differences to CUDA C++.
- CuPy raw kernels and elementwise kernels.
- High-level frameworks (e.g., PyTorch custom CUDA extensions) at a conceptual level.

**Hands-on**

- Re-implement the earlier vector-add and 2D ops using:
  - Numba CUDA kernels.
  - (Optional) CuPy raw kernel.
- Compare code volume and ergonomics vs C++ while emphasizing that the same concepts apply.

**Assignments**

- Implement a custom Python-level GPU function for something slightly nontrivial:
  - Example: row-wise softmax or L2 normalization for a 2D tensor.
- Ask students to profile and compare:
  - Python-loop version.
  - NumPy version (CPU).
  - Numba/CuPy CUDA version.

***

## Module 7 – Classic parallel patterns and reductions

**Objectives**

- Introduce common GPU parallel patterns they see in ML/DS workloads.
- Give them building blocks they can recognize in libraries.

**Topics**

- Reductions: sum, max, argmax.
- Prefix sums (scan) at a conceptual level.
- Basic atomics and where they show up.

**Hands-on**

- Implement a simple block-level reduction (sum) with shared memory.
- Compare to a Python/NumPy reduction, then call the CUDA kernel from a small driver.
- Discuss how this generalizes to log-sum-exp, norms, etc.

**Assignments**

- Implement:
  - Norm computation (L2) using a reduction.
- Short conceptual questions:
  - Why are reductions “harder” than elementwise ops?
  - Where do you expect to see reductions in ML (loss computation, metrics, etc.)?

***

## Module 8 – Capstone: design and implement a small GPU kernel project

**Objectives**

- Consolidate all concepts in a small end-to-end project.
- Practice reading/writing both CUDA C++ and Python integration.

**Project options** (students pick one):

- Implement and benchmark a fused GPU kernel for a small neural network building block (e.g., `y = relu(Ax + b)`).
- Write a small image processing pipeline (blur, edge detection, or simple filter) fully on the GPU.
- Implement a simplified k-means step (assignment + partial reduction) on the GPU.

**Deliverables**

- CUDA C++ or Numba/CuPy code.
- Python harness for:
  - Data generation.
  - Correctness checks vs NumPy.
  - Simple performance plots.
- Short write-up (2–3 pages) covering:
  - Problem description.
  - Design decisions (grid/block configuration, memory layout).
  - Performance observations and future optimization ideas.

***

If you’d like, I can next flesh out one module (for example, Module 2) with a concrete lecture outline plus exact code skeletons and exercises you can drop into a repo or slides.
