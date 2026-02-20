# Matrix-Free FEM Solver for the Advection-Diffusion-Reaction Equation

A finite element solver for the advection-diffusion-reaction (ADR) equation in 2D/3D using [deal.II](https://www.dealii.org/). Compares a **matrix-free** approach (sum factorization + SIMD vectorization + geometric multigrid) against a traditional **matrix-based** approach (sparse matrix assembly) in terms of performance and memory usage.

## Table of Contents

- [Mathematical Formulation](#mathematical-formulation)
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Build & Run](#build--run)
- [Project Structure](#project-structure)
- [Parameter Files](#parameter-files)
- [Output](#output)
- [Performance Results](#performance-results)

## Mathematical Formulation

### Strong form

$$\begin{cases} -\nabla \cdot (\mu \, \nabla u) + \boldsymbol{\beta} \cdot \nabla u + \gamma \, u = f & \text{in } \Omega \subset \mathbb{R}^d, \quad d \in \{2,3\} \\[4pt] u = g & \text{on } \Gamma_D \subset \partial\Omega \\[4pt] \nabla u \cdot \vec{n} = h & \text{on } \Gamma_N = \partial\Omega \setminus \Gamma_D \end{cases}$$

where $\mu$ is the diffusion coefficient, $\boldsymbol{\beta}$ is the advection field, $\gamma$ is the reaction coefficient, and $f$ is the forcing term.

### Weak form

Find $u \in V$: $\quad a(u, v) = f(v) + a(R_g, v) \quad \forall \, v \in V$

where:

$$a(u, v) = \underbrace{\int_\Omega \mu \, \nabla u \cdot \nabla v}_{\text{diffusion}} + \underbrace{\int_\Omega (\boldsymbol{\beta} \cdot \nabla u) \, v}_{\text{advection}} + \underbrace{\int_\Omega \gamma \, u \, v}_{\text{reaction}}$$

$$f(v) = \int_\Omega f \, v + \int_{\Gamma_N} h \, v$$

## Prerequisites

| Dependency | Version |
|------------|---------|
| Ubuntu | 24.04 LTS |
| deal.II | 9.5.1 |
| MPI (OpenMPI) | any recent |
| Boost | >= 1.72.0 |
| fmt | any recent |

## Environment Setup


```bash
sudo apt update
sudo apt install -y build-essential cmake gdb
sudo apt install -y libopenmpi-dev openmpi-bin
sudo apt install -y libboost-all-dev
sudo apt install -y libdeal.ii-dev
sudo apt install -y libfmt-dev
```

## Build & Run

```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)

# Matrix-free solver:
mpirun -np 4 ./bin/matrix_free_test

# Matrix-based solver:
mpirun -np 4 ./bin/matrix_based_test
```

Parameter file paths are set in `tests/matrix_free_adr.cpp` and `tests/matrix_based_adr.cpp`.

## Project Structure

```
nmpde-matrix-free-solver/
├── include/
│   ├── general_definitions.hpp   # Constants (fe_degree=2, dim=3), debug macros
│   ├── prm_handler.hpp           # Parameter file handler (parses .prm files)
│   ├── adr_operator.hpp          # Matrix-free ADR operator (cell kernel, smoother)
│   ├── adr_problem.hpp           # Matrix-free problem driver
│   └── adr_mb_problem.hpp        # Matrix-based problem driver
├── src/
│   ├── adr_prm_handler.cpp       # Parameter parsing implementation
│   ├── adr_problem.cpp           # Matrix-free problem implementation
│   └── adr_mb_problem.cpp        # Matrix-based problem implementation
├── tests/
│   ├── matrix_free_adr.cpp       # Entry point: matrix-free executable
│   └── matrix_based_adr.cpp      # Entry point: matrix-based executable
├── input/
│   ├── params/                   # Default parameter files
│   │   ├── pb_3d_mf.prm          #   3D test (matrix-free)
│   │   ├── pb_3d_mb.prm          #   3D test (matrix-based)
│   │   ├── test_mf.prm           #   Reference test (matrix-free)
│   │   ├── test_mb.prm           #   Reference test (matrix-based)
│   │   └── default_mf.prm        #   Default parameters
│   ├── 3D/
│   │   ├── homo.prm              #   Homogeneous Dirichlet BCs
│   │   └── nonhomo.prm           #   Mixed Dirichlet + Neumann BCs
│   └── 2D/
│       ├── homo_small.prm        #   Mild advection
│       └── homo_biggest.prm      #   Strong advection
├── csv/                          # Benchmark results (memory + timing)
├── results/                      # Solution output (VTU/PVTU files)
├── CMakeLists.txt
└── cmake-common.cmake
```

## Parameter Files

Problems are configured via `.prm` files using deal.II's `ParameterHandler` format. Coefficient expressions support standard math functions (`sin`, `cos`, `exp`, `pi`, etc.). See `input/` for examples.

| Parameter | Description |
|-----------|-------------|
| `Refinements` | Global refinement levels (comma-separated) |
| `Diffusion` | Diffusion coefficient $\mu(\mathbf{x})$ |
| `Advection x/y/z` | Components of advection field $\boldsymbol{\beta}$ |
| `Reaction` | Reaction coefficient $\gamma(\mathbf{x})$ |
| `Force` | Forcing term $f(\mathbf{x})$ |
| `Dirichlet BC` | BC values per boundary ID |
| `Dirichlet Tags` | Boundary IDs for Dirichlet conditions |
| `Neumann BC` | Neumann flux values per boundary ID |
| `Neumann Tags` | Boundary IDs for Neumann conditions |
| `Solver type` | `CG` (symmetric) or `GMRES` (non-symmetric) |
| `Max iters` | Maximum solver iterations |
| `Tolerance` | Solver convergence tolerance |
| `Output file` | Prefix for output VTU files |

## Output

**Solution files:** Written to `results/` as `.vtu` and `.pvtu` files. Open with [ParaView](https://www.paraview.org/) for visualization. Output is automatically skipped when the mesh exceeds 1M elements.

## Authors

Samuele Allegranza, Vale Turco, Bianca Michielan, Valeriia Potrebiina
