#include "advection_diffusion_problem.h"

// ====================== Coefficient Functions ====================== 

template <int dim>
double DiffusionCoefficient<dim>::value(const Point<dim> &p,
                                        const unsigned int /*component*/) const
{
  // TODO: Define diffusion coefficient μ(x)
  // Example: constant or spatially varying
  return 1.0;
}

template <int dim>
Tensor<1, dim> AdvectionCoefficient<dim>::value(const Point<dim> &p) const
{
  // TODO: Define advection field β(x)
  // Example: constant or rotating field
  Tensor<1, dim> advection;
  // advection[0] = 1.0;
  return advection;
}

template <int dim>
double ReactionCoefficient<dim>::value(const Point<dim> &p,
                                       const unsigned int /*component*/) const
{
  // TODO: Define reaction coefficient γ(x)
  return 0.0;
}

template <int dim>
double RightHandSide<dim>::value(const Point<dim> &p,
                                 const unsigned int /*component*/) const
{
  // TODO: Define right-hand side f(x)
  // For manufactured solution: compute f = -∇·(μ∇u) + ∇·(βu) + γu
  return 1.0;
}

template <int dim>
double DirichletBoundaryValues<dim>::value(const Point<dim> &p,
                                           const unsigned int /*component*/) const
{
  // TODO: Define Dirichlet boundary values g(x)
  return 0.0;
}

template <int dim>
double NeumannBoundaryValues<dim>::value(const Point<dim> &p,
                                         const unsigned int /*component*/) const
{
  // TODO: Define Neumann boundary values h(x)
  return 0.0;
}

// ====================== Main Problem Class ====================== 

template <int dim, int fe_degree>
AdvectionDiffusionProblem<dim, fe_degree>::AdvectionDiffusionProblem()
  : fe(fe_degree)
  , dof_handler(triangulation)
  , computing_timer(std::cout, TimerOutput::never, TimerOutput::wall_times)
{
  // TODO: Initialize coefficient functions
  mu_function = std::make_shared<DiffusionCoefficient<dim>>();
  beta_function = std::make_shared<AdvectionCoefficient<dim>>();
  gamma_function = std::make_shared<ReactionCoefficient<dim>>();
  rhs_function = std::make_shared<RightHandSide<dim>>();
}

template <int dim, int fe_degree>
void AdvectionDiffusionProblem<dim, fe_degree>::make_grid()
{
  TimerOutput::Scope t(computing_timer, "mesh generation");

  // TODO: Create mesh (e.g., hyper_cube, hyper_ball, etc.)
  // GridGenerator::hyper_cube(triangulation, 0.0, 1.0);
  // triangulation.refine_global(3);

  std::cout << "Number of active cells: " << triangulation.n_active_cells()
            << std::endl;
}

template <int dim, int fe_degree>
void AdvectionDiffusionProblem<dim, fe_degree>::setup_system()
{
  TimerOutput::Scope t(computing_timer, "setup system");

  // TODO: Distribute DoFs
  // dof_handler.distribute_dofs(fe);

  std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;

  // TODO: Setup constraints for Dirichlet boundary conditions
  // constraints.clear();
  // DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  // VectorTools::interpolate_boundary_values(dof_handler, 
  //                                          0, // boundary_id
  //                                          DirichletBoundaryValues<dim>(),
  //                                          constraints);
  // constraints.close();

  // TODO: Initialize matrix-free data structure
  // typename MatrixFree<dim, double>::AdditionalData additional_data;
  // additional_data.tasks_parallel_scheme = 
  //   MatrixFree<dim, double>::AdditionalData::partition_partition;
  // additional_data.mapping_update_flags = 
  //   (update_gradients | update_JxW_values | update_quadrature_points);
  
  // matrix_free_data = std::make_shared<MatrixFree<dim, double>>();
  // matrix_free_data->reinit(MappingQ1<dim>(),
  //                          dof_handler,
  //                          constraints,
  //                          QGauss<1>(fe_degree + 1),
  //                          additional_data);

  // TODO: Initialize matrix-free operator
  // matrix_free_operator.initialize(matrix_free_data,
  //                                 mu_function,
  //                                 beta_function,
  //                                 gamma_function);
  // matrix_free_operator.compute_diagonal();

  // TODO: Initialize vectors
  // matrix_free_data->initialize_dof_vector(solution_matrix_free);
  // matrix_free_data->initialize_dof_vector(system_rhs);

  // TODO: Setup matrix-based operator
  // matrix_based_operator.initialize_sparsity_pattern(dof_handler, constraints);
  // matrix_based_operator.assemble_matrix(dof_handler,
  //                                       QGauss<dim>(fe_degree + 1),
  //                                       constraints,
  //                                       mu_function,
  //                                       beta_function,
  //                                       gamma_function);

  // TODO: Initialize standard vectors
  // solution_matrix_based.reinit(dof_handler.n_dofs());
  // system_rhs_standard.reinit(dof_handler.n_dofs());
}

template <int dim, int fe_degree>
void AdvectionDiffusionProblem<dim, fe_degree>::assemble_rhs()
{
  TimerOutput::Scope t(computing_timer, "assemble RHS");

  // TODO: Assemble right-hand side vector
  // FEValues<dim> fe_values(fe,
  //                         QGauss<dim>(fe_degree + 1),
  //                         update_values | update_quadrature_points |
  //                         update_JxW_values);
  
  // const unsigned int dofs_per_cell = fe.dofs_per_cell;
  // const unsigned int n_q_points = fe_values.n_quadrature_points;
  
  // Vector<double> cell_rhs(dofs_per_cell);
  // std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  
  // for (const auto &cell : dof_handler.active_cell_iterators())
  // {
  //   cell_rhs = 0;
  //   fe_values.reinit(cell);
  //   
  //   for (unsigned int q = 0; q < n_q_points; ++q)
  //   {
  //     const double rhs_value = rhs_function->value(fe_values.quadrature_point(q));
  //     
  //     for (unsigned int i = 0; i < dofs_per_cell; ++i)
  //     {
  //       cell_rhs(i) += (fe_values.shape_value(i, q) * 
  //                       rhs_value *
  //                       fe_values.JxW(q));
  //     }
  //   }
  //   
  //   // TODO: Add Neumann boundary contributions if needed
  //   
  //   cell->get_dof_indices(local_dof_indices);
  //   constraints.distribute_local_to_global(cell_rhs,
  //                                          local_dof_indices,
  //                                          system_rhs_standard);
  // }

  // TODO: Copy to distributed vector for matrix-free
  // system_rhs = system_rhs_standard;
}

template <int dim, int fe_degree>
void AdvectionDiffusionProblem<dim, fe_degree>::solve_matrix_free()
{
  TimerOutput::Scope t(computing_timer, "solve (matrix-free)");

  // TODO: Setup preconditioner
  // PreconditionJacobi<MatrixFreeAdvectionDiffusion<dim, fe_degree, double>> 
  //   preconditioner;
  // preconditioner.initialize(matrix_free_operator);

  // TODO: Solve with CG
  // SolverControl solver_control(1000, 1e-12 * system_rhs.l2_norm());
  // SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);
  
  // solution_matrix_free = 0;
  // cg.solve(matrix_free_operator,
  //          solution_matrix_free,
  //          system_rhs,
  //          preconditioner);
  
  // constraints.distribute(solution_matrix_free);

  std::cout << "Matrix-free solver converged in "
            << "TODO" << " iterations." << std::endl;
}

template <int dim, int fe_degree>
void AdvectionDiffusionProblem<dim, fe_degree>::solve_matrix_based()
{
  TimerOutput::Scope t(computing_timer, "solve (matrix-based)");

  // TODO: Setup preconditioner
  // PreconditionSSOR<SparseMatrix<double>> preconditioner;
  // preconditioner.initialize(matrix_based_operator.get_system_matrix(), 1.2);

  // TODO: Solve with CG
  // SolverControl solver_control(1000, 1e-12 * system_rhs_standard.l2_norm());
  // SolverCG<Vector<double>> cg(solver_control);
  
  // solution_matrix_based = 0;
  // cg.solve(matrix_based_operator.get_system_matrix(),
  //          solution_matrix_based,
  //          system_rhs_standard,
  //          preconditioner);
  
  // constraints.distribute(solution_matrix_based);

  std::cout << "Matrix-based solver converged in "
            << "TODO" << " iterations." << std::endl;
}

template <int dim, int fe_degree>
void AdvectionDiffusionProblem<dim, fe_degree>::compute_error() const
{
  // TODO: Compute L2 and H1 errors if exact solution is known
  // Vector<float> difference_per_cell(triangulation.n_active_cells());
  
  // VectorTools::integrate_difference(dof_handler,
  //                                   solution_matrix_free,
  //                                   ExactSolution<dim>(),
  //                                   difference_per_cell,
  //                                   QGauss<dim>(fe_degree + 2),
  //                                   VectorTools::L2_norm);
  
  // const double L2_error = VectorTools::compute_global_error(
  //   triangulation, difference_per_cell, VectorTools::L2_norm);
  
  // std::cout << "L2 error: " << L2_error << std::endl;
}

template <int dim, int fe_degree>
void AdvectionDiffusionProblem<dim, fe_degree>::output_results(
  const unsigned int cycle)
{
  TimerOutput::Scope t(computing_timer, "output results");

  // TODO: Output solution to VTU file
  // DataOut<dim> data_out;
  // data_out.attach_dof_handler(dof_handler);
  
  // Vector<double> solution_output(dof_handler.n_dofs());
  // solution_output = solution_matrix_free; // or solution_matrix_based
  
  // data_out.add_data_vector(solution_output, "solution");
  // data_out.build_patches();
  
  // std::ofstream output("solution-" + std::to_string(cycle) + ".vtu");
  // data_out.write_vtu(output);
}

template <int dim, int fe_degree>
void AdvectionDiffusionProblem<dim, fe_degree>::print_performance_metrics()
{
  // TODO: Print comparison table
  std::cout << "\n========================================" << std::endl;
  std::cout << "Performance Comparison" << std::endl;
  std::cout << "========================================" << std::endl;
  
  // TODO: Print table with columns:
  // - DoFs
  // - Setup time (MF vs MB)
  // - Solve time (MF vs MB)
  // - Speedup factor
  // - Iterations (MF vs MB)
  // - Memory usage (MF vs MB)
  
  computing_timer.print_summary();
}

template <int dim, int fe_degree>
void AdvectionDiffusionProblem<dim, fe_degree>::run()
{
  const unsigned int n_cycles = 5;
  
  for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
  {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Cycle " << cycle << std::endl;
    std::cout << "========================================" << std::endl;

    if (cycle == 0)
    {
      make_grid();
    }
    else
    {
      // TODO: Refine mesh
      // triangulation.refine_global(1);
    }

    setup_system();
    assemble_rhs();
    
    std::cout << "\nSolving with matrix-free method..." << std::endl;
    solve_matrix_free();
    
    std::cout << "\nSolving with matrix-based method..." << std::endl;
    solve_matrix_based();
    
    // TODO: Compare solutions
    // LinearAlgebra::distributed::Vector<double> difference = solution_matrix_free;
    // difference -= solution_matrix_based;
    // std::cout << "Difference between solutions: " 
    //           << difference.l2_norm() << std::endl;
    
    output_results(cycle);
  }

  print_performance_metrics();
}

// Explicit instantiations
template class AdvectionDiffusionProblem<2, 1>;
template class AdvectionDiffusionProblem<2, 2>;
template class AdvectionDiffusionProblem<2, 3>;
template class AdvectionDiffusionProblem<3, 1>;
template class AdvectionDiffusionProblem<3, 2>;
template class AdvectionDiffusionProblem<3, 3>;