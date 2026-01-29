# Project 7: Matrix-Free Solvers

Implementation of matrix-free solvers for the advection-diffusion-reaction equation using deal.II library.

## Problem Description

This project solves the advection-diffusion-reaction problem:

```
-∇·(μ∇u) + ∇·(βu) + γu = f    in Ω
u = g                          on Γ_D ⊂ ∂Ω
∇u·n = h                       on Γ_N = ∂Ω\Γ_D
```

where:
- μ(x) is the diffusion coefficient
- β(x) is the advection field
- γ(x) is the reaction coefficient
- f(x) is the source term
- g(x) and h(x) are boundary values

## Objectives

1. Implement a **matrix-free solver** using deal.II's MatrixFree framework
2. Implement a **matrix-based solver** for comparison
3. Compare performance in terms of:
   - Computational efficiency (wall time)
   - Memory consumption
   - Parallel scalability
   - Implementation complexity

## Project Structure

```
├── CMakeLists.txt              # Main CMake configuration
├── cmake-common.cmake          # Common CMake settings
├── README.md                   # This file
├── include/
│   ├── advection_diffusion_problem.h # Problem class header
│   ├── matrix_based_operator.h       # Matrix-based operator class
│   └── matrix_free_operator.h        # Matrix-free operator class
├── src/
│   ├── main.cpp                      # Main driver
│   └── advection_diffusion_problem.cpp # Problem class implementation
├── tests/                      # Benchmarks and tests (see Testing section)
│   ├── CMakeLists.txt
│   ├── matrix_based_adr.cpp
│   ├── matrix_based_adr_seq.cpp
│   ├── matrix_free_adr.cpp
│   └── matrix_free_poisson.cpp
├── meshes/                     # Mesh files
└── results/                    # Output directory
```

## Building the Project

```bash
# Create build directory
mkdir build
cd build

# Configure with CMake
cmake ..

# Build
make

# Run
./matrix_free_solver
```

## Development tests

The `tests/` directory contains several benchmark and validation tests. These are built automatically when you compile the project.

### Available Tests

- **`matrix_free_poisson`**: Matrix-free Poisson solver. A simpler validation case ensuring the matrix-free infrastructure works correctly for the Poisson equation.
- **`matrix_based_adr`**: Matrix-Based Advection-Diffusion-Reaction Benchmark (MPI Parallel). Useful for comparing performance against the matrix-free implementation in a distributed setting.
- **`matrix_based_adr_seq`**: Matrix-Based Advection-Diffusion-Reaction Benchmark (Serial). Useful to quickly compute the solution of a cefined problem.
- **`matrix_free_adr`**: Matrix-free Advection-Diffusion-Reaction Solver. This is the primary test for the matrix-free implementation, including memory benchmarking.

### Running Tests

The test executables are located in the `build/tests/` directory after building.

```bash
cd build/tests

# Run matrix-free ADR solver
mpirun -np 8 ./matrix_free_adr

# Run matrix-based parallel ADR solver
mpirun -np 8 ./matrix_based_adr

# Run matrix-based serial ADR solver
./matrix_based_adr_serial
```

## Implementation Guide

### TODO List

#### 1. Coefficient Functions (src/advection_diffusion_problem.cc)
- [ ] Implement `DiffusionCoefficient::value()` - define μ(x)
- [ ] Implement `AdvectionCoefficient::value()` - define β(x)
- [ ] Implement `ReactionCoefficient::value()` - define γ(x)
- [ ] Implement `RightHandSide::value()` - define f(x)
- [ ] Implement boundary value functions g(x) and h(x)

#### 2. Mesh Generation (make_grid)
- [ ] Create initial mesh using GridGenerator
- [ ] Set appropriate boundary IDs for Dirichlet/Neumann conditions
- [ ] Refine mesh globally or adaptively

#### 3. System Setup (setup_system)
- [ ] Distribute DoFs
- [ ] Setup hanging node constraints
- [ ] Apply Dirichlet boundary constraints
- [ ] Initialize MatrixFree data structure
- [ ] Initialize matrix-free operator with coefficients
- [ ] Compute diagonal for preconditioning
- [ ] Setup sparsity pattern for matrix-based approach
- [ ] Assemble system matrix

#### 4. Matrix-Free Operator (include/matrix_free_operator.h)
- [ ] Store coefficient functions in vectorized format
- [ ] Implement `local_apply()` cell loop
  - [ ] Evaluate solution values and gradients
  - [ ] Apply weak form: ∫[μ∇u·∇v + β·∇u v + γuv] dx
  - [ ] Submit values and gradients
- [ ] Implement `vmult()` using cell_loop
- [ ] Implement diagonal computation for Jacobi preconditioner

#### 5. Matrix-Based Operator (include/matrix_based_operator.h)
- [ ] Create dynamic sparsity pattern
- [ ] Implement standard assembly loop
  - [ ] Loop over cells and quadrature points
  - [ ] Evaluate coefficients at quadrature points
  - [ ] Assemble local matrix contributions
  - [ ] Distribute to global matrix

#### 6. RHS Assembly (assemble_rhs)
- [ ] Implement standard RHS assembly with FEValues
- [ ] Add Neumann boundary contributions (face integrals)
- [ ] Apply constraints

#### 7. Solvers
- [ ] **Matrix-free solver**:
  - [ ] Setup Jacobi preconditioner using diagonal
  - [ ] Configure CG solver
  - [ ] Track iteration count and solve time
- [ ] **Matrix-based solver**:
  - [ ] Setup SSOR or ILU preconditioner
  - [ ] Configure CG solver
  - [ ] Track iteration count and solve time

#### 8. Performance Analysis (print_performance_metrics)
- [ ] Collect timing data for both methods
- [ ] Compute memory usage estimates
- [ ] Calculate speedup factors
- [ ] Generate comparison tables
- [ ] Plot scaling curves (optional)

#### 9. Verification
- [ ] Implement manufactured solution
- [ ] Compute L2 and H1 errors
- [ ] Verify convergence rates
- [ ] Compare matrix-free vs matrix-based solutions

#### 10. Output (output_results)
- [ ] Write solution to VTU files
- [ ] Visualize with ParaView
- [ ] Save performance data to files

## Expected Results

### Performance Characteristics

**Matrix-Free Advantages:**
- Lower memory consumption: O(N) vs O(N log N) for sparse matrix
- Better cache efficiency and vectorization
- 5-10× speedup for high-order elements (p ≥ 3)
- Excellent parallel scalability

**Matrix-Based Advantages:**
- Simpler implementation
- More flexible preconditioners available
- Competitive for low-order elements (p = 1, 2)


## References

- deal.II Step-37: [Matrix-free methods](https://www.dealii.org/current/doxygen/deal.II/step_37.html)
- deal.II Documentation: [MatrixFree class](https://www.dealii.org/current/doxygen/deal.II/classMatrixFree.html)
- Kronbichler & Kormann (2012): "Fast matrix-free evaluation of discontinuous Galerkin finite element operators"
