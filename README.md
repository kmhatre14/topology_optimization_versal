# Topology Optimization Acceleration using AI Engine on AMD Versal

## Project Overview

This repository implements and accelerates a topology optimization workflow using Finite Element Analysis (FEA) and a convolutional neural network (CRONet) mapped onto the AI Engine (AIE) architecture on AMD's Versal platform (VEK280). It includes a full C++ conversion of FEA routines, integration into AIE kernels, and vectorized and non-vectorized kernel variants for benchmarking and deployment

# Repository Structure
```
aie/dt/
├── aie_bf16_bf16_64x96x64_api8x8x4/
├── bf16_experiment/
├── cronet/
├── cronet_v2/
│   ├── data_types.hpp
│   ├── main.cpp             ← Entrypoint for CRONet graph construction
│   ├── mat.cpp / mat.hpp    ← Matrix utilities
│   ├── nn.hpp               ← CRONet inference
│   ├── test.cpp             ← Sample test runner
├── fea/
│   ├── fea_linear_solver/   ← Vectorized FEA kernel (by us) --> Run for Reviewing Performance of Vectorized Code
│   ├── fea_updated_vedant/  ← Scalar (non-vectorized) baseline FEA kernel (by us) --> Run for Reviewing Performance of Non Vectorized Code
│   │   ├── kernels/
│   │   │   └── kernels.cc   ← Custom FEA kernel logic
│   │   ├── aie_top_all.cc   ← AIE graph and stream integration
│   │   ├── data/
│   │   │   ├── x.txt, K.txt, U.txt, KE.txt  ← Sample I/O for test cases
├── Makefile_VEK280_dt       ← Top-level Makefile for VEK280 AIE compile
├── env_source.sh            ← Environment setup script
```

**Key Contributions**

**✅ What we wrote or modified:**

* `kernels.cc`: Custom C++ implementation of the Finite Element solver (FE) with stiffness assembly, boundary conditions, and custom Gaussian elimination (`spsolve()`)
* `fea_linear_solver/`: Vectorized kernel version of FEA for the AIE using AIE API calls for data streaming and optimization
* `aie_top_all.cc`: AIE graph for sequential dataflow between FEA and CRONet kernels
* `mat.hpp`, `mat.cpp`: C++ matrix manipulation helpers (ported from Python numpy functionality)
* `main.cpp`, `test.cpp`: Simulation and debug runners
* `Makefile_VEK280_dt`: Modified for hardware target compilation and simulation
* Data files (`x.txt`, `K.txt`, etc.): Provide working test vectors for simulation and verification

**What we used from external sources:**

* AIE architectural concepts and templates inspired from AMD Vitis AIE tutorials
* CRONet model and architecture ported from a Python implementation based on standard CNN structures
* CRONet Paper

**FEA Implementation: Variables and Setup**

The topology optimization problem, as described in the referenced paper, assumes a discretized design domain using a finite element mesh. The key variables for the FEA implementation are:

* `nelx = 60` and `nely = 20`: These define the number of elements in the horizontal and vertical directions, respectively, resulting in a total of $N = nelx \times nely = 1200$ elements for the full problem. For initial project testing, a smaller mesh of $6 \times 2 = 12$ elements was used
* $x_e \in [0, 1]$: This represents the design variable for each element, indicating material density. Values close to 1 denote solid material, while values near 0 represent voids
* $\mathbf{U}$: The global displacement vector
* $\mathbf{F}$: The global force vector
* $\mathbf{K}$: The global stiffness matrix
* $u_e$: The element displacement vector
* $k_e$: The element stiffness matrix
* $\mathbf{x}$: A vector holding the design variables for all elements' material densities, with a lower bound $x_{min}$ to prevent singularities in the stiffness matrix
* $p$: A penalization factor that biases the solution towards solid or void regions
* $V_0$: The total volume of the design domain
* $V(\mathbf{x})$: The current material volume, constrained by the volume fraction $f$, which limits the total amount of material that can be used

The FEA process begins with an initial uniform distribution of material across the design domain. FEA is then used to generate displacement data $[\mathbf{x}^t, \mathbf{F}, \mathbf{U}^t]$ for training the CRONet model. In each optimization iteration, CRONet predicts displacements, and FEA correction is applied only when significant design changes occur, continuing until convergence is reached and an optimal structure is obtained. The objective of the optimization is to maximize structural stiffness. Table I (not shown) outlines the inputs and outputs of the FEA algorithm.

**Constraints**

The original Python program for topology optimization is structured around the function `top(nelx, nely, volfrac, penal, rmin)`. The parameters used are:

* `nelx = 60`, `nely = 20`: Defining the 2D grid dimensions
* `volfrac = 0.5`: The volume fraction, limiting the total material usage to 50% of the design domain
* `penal = 3.0`: The penalization factor, encouraging the solution to converge to either fully solid or void elements
* `rmin = 1.5`: The filter size, controlling the spatial smoothing of the material distribution to prevent overly fragmented designs

**Modifications to Input: Explaining the Reduced Linear System**

In FEA, the discretization of the domain leads to a large system of linear equations, fundamentally represented as $\mathbf{K} \cdot \mathbf{U} = \mathbf{F}$. To solve for the unknown field variables (displacements $\mathbf{U}$), this system is often reduced by considering only the "free" degrees of freedom (DOFs), where displacements are unknown. In this project's context, each element has 4 nodes, and assuming 2 DOFs per node in a 2D problem, a fully unconstrained system for the initial $6 \times 2$ mesh would have $12 \times 4 \times 2 = 96$ DOFs. However, boundary conditions (fixed displacements) reduce the number of unknowns. For the full $60 \times 20$ mesh, the initial number of DOFs is $(60 \times 20) \times 4 \times 2 = 9600$. The example mentions a reduction from 2562 DOFs to 2540 free DOFs, implying 22 fixed DOFs. This reduction yields a smaller, more manageable linear system: $\mathbf{K}_{free} \cdot \mathbf{U}_{free} = \mathbf{F}_{free}$, where $\mathbf{K}_{free} \in \mathbb{R}^{2540 \times 2540}$ is the reduced stiffness matrix, $\mathbf{U}_{free} \in \mathbb{R}^{2540 \times 1}$ is the vector of unknown displacements, and $\mathbf{F}_{free} \in \mathbb{R}^{2540 \times 1}$ is the corresponding reduced force vector. The reduction process involves partitioning the original system based on free and fixed DOFs and extracting the submatrices corresponding to the free DOFs to form this smaller system, which is then solved to find $\mathbf{U}_{free}$

**Vectorization in AIE**

To enable self-contained computation within the AIE kernel, all necessary data, including the material density field, boundary conditions, and the $8 \times 8$ elemental stiffness matrix, were either hardcoded or passed via AIE buffers. For example, the $\mathbf{KE}$ matrix, typically computed by `lk()`, was embedded as a constant 2D array within the kernel. Similarly, parameters like `nelx`, `nely`, and the force vectors were defined directly in the kernel code to avoid external file I/O or access to the host's main memory during kernel execution

The porting of the raw C++ FEA solver to the Vitis AI Engine (AIE) platform necessitated several architectural and implementation-level modifications to align with the AIE toolchain and effectively utilize its vector processing capabilities. The core `FE()` function was refactored and rewritten as `FE_kernel()` (as shown in Figs. 12 and 16) into a flattened, monolithic structure suitable for execution on the AI Engine. The specific changes implemented are detailed below:

**Summary of Changes**
* The AI Engine environment has limited support for the C++ Standard Template Library (STL) and does not support dynamic memory allocation or object-based structures
  * All uses of `std::vector` and `std::array`, as well as any dynamically allocated structures, were removed
  * These were replaced with statically allocated `float` and `int` arrays of fixed sizes
  * Utility features such as sorting, which might have been provided by STL, were implemented manually
* Inlining and Flattening of Subroutines
  * `lk()`: For computing the local element stiffness matrix
  * `spsolve()`: For solving the reduced linear system
  * Various matrix utility functions
* To optimize for the AIE, these subroutines were completely flattened and their functionality was rewritten directly inside the `FE_kernel()`:
  * The element stiffness matrix ($\mathbf{KE}$) is now hardcoded within the kernel
  * Matrix assembly, application of boundary conditions, and the linear system solving process are performed in-place within a single function
* Hardcoding of Problem Data
* File I/O and runtime memory access are not efficient within AI Engine kernels:
  * Input parameters like `nelx`, `nely`, and the element stiffness matrix $\mathbf{KE}$ were hardcoded into the `FE_kernel()` function
  * The material density array `x` is read from an `input_buffer`, while all other simulation constants are embedded as `constexpr`
* Use of AIE Vector API
  * Matrix row operations were performed using functions like `aie::load_v`, `aie::store_v`, `aie::add`, `aie::mul`, and `aie::reduce_add_v`
  * The Gaussian elimination and back substitution stages of the sparse linear solver were partially vectorized

**A. Mapping CNN to AIE: Tile Mapping and GEMM Conversion**

The initial implementation for CRONet on the AIE in C++ involves the Trunk Network and utilizes two primary classes: `CustomMatrix` for matrix manipulations and `NeuralNetwork` encapsulating different layers and activation functions (convolution, element-wise matrix multiplication, and sigmoid). Convolution operations on NCHW formatted images are implemented by converting them into General Matrix Multiplication (GEMM) operations using subroutines: `im2col()`, `conv2mat`, and `gemm2conv()`.

**Steps**

* The `im2col()` routine:
    * Rearranges input tensor data (N images, C channels, H height, W width) and a kernel (K kernels, R height, S width, C channels) into columns
    * Facilitates conversion to GEMM operations
    * The output is a matrix of size $(R \cdot S \cdot C) \times (N \cdot P \cdot Q)$
    * Output dimensions P and Q are calculated as:
        * $P = \lfloor \frac{H - R + 2 \cdot pad}{stride + 1} \rfloor$
        * $Q = \lfloor \frac{W - S + 2 \cdot pad}{stride + 1} \rfloor$
* The `conv2mat()` function:
    * Flattens the convolution kernel
    * The output matrix has dimensions $K \times (R \cdot S \cdot C)$
* The `gemm` function:
    * Performs matrix multiplication
    * Multiplies the output of `im2col()` with the flattened kernel from `conv2mat()`
* The `gemm2conv()` function:
    * Reshapes the GEMM output
    * Converts the 2D GEMM output (size $K \times (P \cdot Q)$) back to the NCHW convolutional output format ($K \times C \times P \times Q$)
* GEMM on AIE:
    * Utilizes tiling to process matrices in sizes compatible with AIE (96x64 and 64x96)
    * Employs the custom `nn::mattiled` function to store matrices in a 4D tensor for order preservation
    * Merges results after individual tiled matrix multiplications
    * Applies a sigmoid activation function to the final convolution output

**How to Build and Run**

```bash
1. source /<dir_path>/Vitis/2024.1/settings64.sh (outside path)
2. export PLATFORM_REPO_PATHS=/<dir_path/Vitis/2024.1/base_platforms/ (outside path)
3. cd topology_optimization_versal
4. make -f Makefile_VEK280_dt aiesim AIE_SRC=aie/dt/a<dir_path> TARGET=hw BUFF_OPT=0 &> compile.log
```

**AIE and Versal Highlights**

**Target Device:** AMD Versal VEK280 (AIE-ML)

**Core Architecture:**

* SIMD/VLIW processors per tile
* Tile-local memory banks (128 KB accessible)
* MM2S and S2MM DMA for efficient memory-stream transfers

**AIE Programming Model:**

* C++/C with AIE APIs
* Emphasis on vectorization and tiling for performance

**Methodology & Motivation**

**Python-to-C++ translation of FEA:**

* Developed custom `lk()` for element stiffness.
* Manual `spsolve()` using forward elimination + back-substitution.

**FEA pipeline in AIE:**

1. Load design variables `x`
2. Assemble global stiffness matrix `K`
3. Apply BCs and solve $\mathbf{K} \cdot \mathbf{U} = \mathbf{F}$

**AIE Graph Implementation:**

* Used AIE graph to sequentially run:
    * FEA Kernel $\rightarrow$ CRONet Kernel $\rightarrow$ Output
* Vectorization benchmarked using two versions of the kernel.

