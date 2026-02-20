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
- [Authors](#authors)

## Mathematical Formulation

### Strong form

```math
\begin{cases} -\nabla \cdot (\mu \nabla u) + \mathbf{\beta} \cdot \nabla u + \gamma u = f & \text{in } \Omega \subset \mathbb{R}^d \quad d \in \{1,2,3\} \\ u = g & \text{on } \Gamma_D \subset \partial\Omega \\ \nabla u \cdot \vec{n} = h & \text{on } \Gamma_N = \partial\Omega \setminus \Gamma_D \end{cases}
\\
```

where $\mu$ is the diffusion coefficient, $\mathbf{\beta}$ is the advection coefficient, $\gamma$ is the reaction coefficient, and $f$ is the forcing term.

### Weak form

Find $u \in V$: $\quad a(u, v) = f(v) + a(R_g, v) \quad \forall v \in V$

where:

$$a(u, v) = \underbrace{\int_\Omega \mu \nabla u \cdot \nabla v}_{\text{diffusion}} + \underbrace{\int_\Omega (\mathbf{\beta} \cdot \nabla u) v}_{\text{advection}} + \underbrace{\int_\Omega \gamma u v}_{\text{reaction}}$$

$$f(v) = \int_\Omega f v + \int_{\Gamma_N} h v$$

## Prerequisites

| Dependency | Version |
|------------|---------|
| Ubuntu | 24.04 LTS |
| deal.II | 9.5.1 |
| MPI (OpenMPI) | any recent |
| Boost | >= 1.72.0 |
| fmt | any recent |

## Environment Setup

On ubuntu 24.04 you only need to install these libraries:
```bash
sudo apt update
sudo apt install -y build-essential cmake gdb
sudo apt install -y libopenmpi-dev openmpi-bin
sudo apt install -y libboost-all-dev
sudo apt install -y libdeal.ii-dev
sudo apt install -y libfmt-dev
sudo apt install -y doxygen graphviz
```

## Build & Run
To build:
```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

We can run all matrix free tests with MPI as follows:
```
cd results
mpirun -np 4 ../bin/matrix_free_test
```

We can run all matrix based tests as follows:
```
cd results
../bin/matrix_based_test
```

Parameter file paths are set in `tests/matrix_free_adr.cpp` and `tests/matrix_based_adr.cpp`.

To generate documentation with doxygen, run:
```bash
cd build
make docs
```

## Project Structure

```
nmpde-matrix-free-solver/
├── bin/                        # Where the binaries are compiled
├── build/                      # CMake build directory
├── docs/                       # Contains doxygen documentation
├── images/                     # A place to put paraview output files 
├── include/                    # The core of the library
├── src/                        # The implementaton of .hpp files 
├── lib/                        # Static libraries are compiled here
├── tests/                      # Set of test for the library
├── input/                      # A set of input parameters for the tests
├── csv/                        # Directory to store csv results of tests
├── results/                    # Paraview files from test
├── CMakeLists.txt
└── cmake-common.cmake
```

## Parameter Files

Problems are configured via `.prm` files using deal.II's `ParameterHandler` format. Coefficient expressions support standard math functions (`sin`, `cos`, `exp`, `pi`, etc.). See `input/` for examples.

| Parameter | Description                                           |
|-----------|-------------------------------------------------------|
| `Refinements` | Global refinement levels (comma-separated)            |
| `Diffusion` | Diffusion coefficient $\mu(\mathbf{x})$               |
| `Advection x/y/z` | Components of advection field $\mathbf{\beta}$        |
| `Reaction` | Reaction coefficient $\gamma(\mathbf{x})$             |
| `Force` | Forcing term $f(\mathbf{x})$                          |
| `Dirichlet BC` | BC values per boundary ID                             |
| `Dirichlet Tags` | Boundary IDs for Dirichlet conditions                 |
| `Neumann BC` | Neumann flux values per boundary ID                   |
| `Neumann Tags` | Boundary IDs for Neumann conditions                   |
| `Solver type` | `CG` (symmetric) or `GMRES` (non-symmetric)           |
| `Max iters` | Maximum solver iterations                             |
| `Tolerance` | Solver convergence tolerance                          |
| `Output file` | Name for the output files, extension will be appended |

## Output

**Solution files:** Written to `results/` as `.vtu` and `.pvtu` files. Open with [ParaView](https://www.paraview.org/) for visualization. Output is automatically skipped when the mesh exceeds 1M elements.

## Authors
Samuele Allegranza, Vale Turco, Bianca Michielan, Valeriia Potrebiina
