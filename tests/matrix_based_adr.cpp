/* ---------------------------------------------------------------------
 * Matrix-Based ADR Benchmark (MPI Parallel) + Memory Logging
 * --------------------------------------------------------------------- */

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/memory_consumption.h> // Include for memory stats

// Distributed Grid
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

// Trilinos Wrappers
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/affine_constraints.h>

#include <fstream>
#include <iostream>
#include <sys/resource.h> // POSIX standard for getrusage

using namespace dealii;

// --------------------------------------------------------------------------
// 1. Coefficients
// --------------------------------------------------------------------------
template <int dim>
class Coefficient
{
public:
  static double get_mu(const Point<dim> &p)
  {
     return 1. / (0.05 + 2. * p.square());
  }

  static Tensor<1, dim> get_beta(const Point<dim> &)
  {
    Tensor<1, dim> beta;
    beta[0] = 1.0;
    beta[1] = 0.5;
    if (dim > 2) beta[2] = 0.0;
    return beta;
  }

  static double get_gamma(const Point<dim> &)
  {
    return 1.0;
  }
};

// --------------------------------------------------------------------------
// 2. Benchmarking Solver Class (MPI)
// --------------------------------------------------------------------------
template <int dim>
class MPIADRBenchmark
{
public:
  MPIADRBenchmark();
  void run();

private:
  void setup_system();
  void assemble_system();
  void solve();
  void output_results(const unsigned int cycle);
  void print_memory_usage(const std::string &stage); // Helper function

  MPI_Comm                                  mpi_communicator;
  parallel::distributed::Triangulation<dim> triangulation;
  FE_Q<dim>                                 fe;
  DoFHandler<dim>                           dof_handler;
  MappingQ1<dim>                            mapping;

  AffineConstraints<double>                 constraints;
  
  TrilinosWrappers::SparseMatrix            system_matrix;
  TrilinosWrappers::MPI::Vector             solution;
  TrilinosWrappers::MPI::Vector             system_rhs;
  TrilinosWrappers::MPI::Vector             locally_relevant_solution;

  IndexSet                                  locally_owned_dofs;
  IndexSet                                  locally_relevant_dofs;

  ConditionalOStream                        pcout;
  Timer                                     timer;
};

template <int dim>
MPIADRBenchmark<dim>::MPIADRBenchmark()
  : mpi_communicator(MPI_COMM_WORLD)
  , triangulation(mpi_communicator,
                  Triangulation<dim>::limit_level_difference_at_vertices,
                  parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy)
  , fe(2) 
  , dof_handler(triangulation)
  , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
{}

// Helper to print memory usage
template <int dim>
void MPIADRBenchmark<dim>::print_memory_usage(const std::string &stage)
{
    // Get Resident Set Size (RSS) in Kilobytes
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    double local_memory_mb = usage.ru_maxrss / 1024.0; // Convert KB to MB (Linux)
    
    // On macOS, ru_maxrss is bytes, so divide by 1024*1024. On Linux it is KB.
    #ifdef __APPLE__
      local_memory_mb /= 1024.0; 
    #endif

    // Find the maximum memory usage across all MPI processes
    double max_memory_mb = Utilities::MPI::max(local_memory_mb, mpi_communicator);
    double min_memory_mb = Utilities::MPI::min(local_memory_mb, mpi_communicator);

    pcout << "  Memory (" << stage << "):        " 
          << "Max: " << max_memory_mb << " MB / Min: " << min_memory_mb << " MB" 
          << std::endl;
}

template <int dim>
void MPIADRBenchmark<dim>::setup_system()
{
  timer.restart();

  dof_handler.distribute_dofs(fe);

  locally_owned_dofs = dof_handler.locally_owned_dofs();
  locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

  solution.reinit(locally_owned_dofs, mpi_communicator);
  system_rhs.reinit(locally_owned_dofs, mpi_communicator);
  
  locally_relevant_solution.reinit(locally_owned_dofs,
                                   locally_relevant_dofs,
                                   mpi_communicator);

  constraints.clear();
  constraints.reinit(locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  VectorTools::interpolate_boundary_values(mapping,
                                           dof_handler,
                                           0, // Boundary ID
                                           Functions::ZeroFunction<dim>(),
                                           constraints);
  constraints.close();

  // Setup Matrix Sparsity (Crucial for Parallel Trilinos)
  TrilinosWrappers::SparsityPattern sp(locally_owned_dofs, mpi_communicator);
  DoFTools::make_sparsity_pattern(dof_handler, sp, constraints, false);
  sp.compress();

  system_matrix.reinit(sp);

  pcout << "   Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

  timer.stop();
  pcout << "   Setup time:                (CPU/wall) " 
        << timer.cpu_time() << "s/" << timer.wall_time() << "s" << std::endl;
        
  print_memory_usage("After Setup");
}

template <int dim>
void MPIADRBenchmark<dim>::assemble_system()
{
  timer.restart();

  system_matrix = 0;
  system_rhs = 0;

  QGauss<dim> quadrature_formula(fe.degree + 1);

  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                          update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned())
      {
          fe_values.reinit(cell);
          cell_matrix = 0;
          cell_rhs    = 0;

          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              const Point<dim> p = fe_values.quadrature_point(q);
              const double dx    = fe_values.JxW(q);

              const double mu         = Coefficient<dim>::get_mu(p);
              const Tensor<1,dim> beta = Coefficient<dim>::get_beta(p);
              const double gamma      = Coefficient<dim>::get_gamma(p);

              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  const double         phi_i      = fe_values.shape_value(i, q);
                  const Tensor<1, dim> grad_phi_i = fe_values.shape_grad(i, q);

                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                      const double         phi_j      = fe_values.shape_value(j, q);
                      const Tensor<1, dim> grad_phi_j = fe_values.shape_grad(j, q);

                      cell_matrix(i, j) +=
                        (mu * grad_phi_j * grad_phi_i +       
                         (beta * grad_phi_j) * phi_i +        
                         gamma * phi_j * phi_i                
                        ) * dx;
                    }
                  cell_rhs(i) += 1.0 * phi_i * dx;
                }
            }

          cell->get_dof_indices(local_dof_indices);
          constraints.distribute_local_to_global(cell_matrix,
                                                 cell_rhs,
                                                 local_dof_indices,
                                                 system_matrix,
                                                 system_rhs);
      }
    }

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
  
  timer.stop();
  pcout << "  Assembly time:             (CPU/wall) " 
        << timer.cpu_time() << "s/" << timer.wall_time() << "s" << std::endl;
        
  print_memory_usage("After Assembly");
}

template <int dim>
void MPIADRBenchmark<dim>::solve()
{
  timer.restart();

  // new
  TrilinosWrappers::PreconditionSSOR preconditioner;
  preconditioner.initialize(system_matrix, 
                            TrilinosWrappers::PreconditionSSOR::AdditionalData(1.0));

  // ReductionControl is more robust than SolverControl 
  // It stops when the residual is reduced by a certain factor
  ReductionControl solver_control(10000, 1.0e-16, 1.0e-6);

  // STICK WITH GMRES for ADR (Advection-Diffusion-Reaction)
  // Using CG here will likely lead to incorrect results/divergence
  TrilinosWrappers::SolverGMRES solver(solver_control);

  pcout << "  Solving the linear system" << std::endl;
  //new

  solver.solve(system_matrix, solution, system_rhs, preconditioner);

  constraints.distribute(solution);

  timer.stop();
  pcout << "  Solve time (" << solver_control.last_step() << " it):      (CPU/wall) " 
        << timer.cpu_time() << "s/" << timer.wall_time() << "s" << std::endl;
        
  print_memory_usage("After Solve");
}

template <int dim>
void MPIADRBenchmark<dim>::output_results(const unsigned int cycle)
{
  locally_relevant_solution = solution;

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(locally_relevant_solution, "solution");
  data_out.build_patches(mapping);

  data_out.write_vtu_with_pvtu_record(
    "./", "adr_mb_mpi_solution", cycle, mpi_communicator, 3);
}

template <int dim>
void MPIADRBenchmark<dim>::run()
{
  pcout << "Running Matrix-Based Benchmark (MPI, Dimension " << dim << ")" << std::endl;

  for (unsigned int cycle = 0; cycle < 4; ++cycle)
  {
      pcout << "Cycle " << cycle << std::endl;

      if (cycle == 0)
      {
          GridGenerator::hyper_cube(triangulation, 0, 1);
          triangulation.refine_global(2);
      }
      triangulation.refine_global(1);

      setup_system();
      assemble_system();
      solve();
      output_results(cycle);
      pcout << std::endl;
  }
}

int main(int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      MPIADRBenchmark<3> problem;
      problem.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << "Exception: " << exc.what() << std::endl;
      return 1;
    }
  return 0;
}