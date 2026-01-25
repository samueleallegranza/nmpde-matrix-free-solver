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
project7/
├── CMakeLists.txt           # Build configuration
├── README.md                # This file
├── parameters/
│   └── parameters.prm       # Runtime parameters (optional)
├── include/
│   ├── matrix_free_operator.h    # Matrix-free operator class
│   └── matrix_based_operator.h   # Matrix-based operator class
├── src/
│   ├── main.cc                         # Main driver
│   ├── advection_diffusion_problem.h   # Problem class header
│   └── advection_diffusion_problem.cc  # Problem class implementation
└── results/                 # Output directory (created at runtime)
```

## Dependencies

- deal.II (version ≥ 9.5.0) compiled with:
  - C++17 support
  - LAPACK/BLAS
  - Threading support (TBB or std::thread)
  - Optional: MPI for parallel runs

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

### Sample Output

```
========================================
Cycle 0
========================================
Number of active cells: 1024
Number of degrees of freedom: 4225

Solving with matrix-free method...
Matrix-free solver converged in 87 iterations.

Solving with matrix-based method...
Matrix-based solver converged in 87 iterations.

Performance Comparison:
DoFs  | Setup MF | Setup MB | Solve MF | Solve MB | Speedup
------|----------|----------|----------|----------|--------
4225  | 0.05s    | 0.12s    | 0.08s    | 0.45s    | 5.6×
```

## Testing Strategy

1. **Correctness**: Use manufactured solutions to verify convergence rates
2. **Consistency**: Ensure both solvers produce identical results (within tolerance)
3. **Scaling**: Test with increasing mesh refinement and polynomial degree
4. **Parallel**: Run with different thread counts to measure scalability

## References

- deal.II Step-37: [Matrix-free methods](https://www.dealii.org/current/doxygen/deal.II/step_37.html)
- deal.II Documentation: [MatrixFree class](https://www.dealii.org/current/doxygen/deal.II/classMatrixFree.html)
- Kronbichler & Kormann (2012): "Fast matrix-free evaluation of discontinuous Galerkin finite element operators"

## Tips for Implementation

1. Start with the matrix-based solver to verify correctness
2. Implement matrix-free for simple case first (constant coefficients)
3. Add variable coefficients incrementally
4. Use manufactured solutions for testing
5. Compare iteration counts between methods (should be identical)
6. Profile code to identify bottlenecks
7. Experiment with different polynomial degrees to see performance crossover

## Notes

- Solutions should match to machine precision (< 1e-10 difference)
- Iteration counts should be identical for same preconditioner
- Matrix-free advantages become clear for p ≥ 2
- Consider using `-march=native` compiler flag for vectorization