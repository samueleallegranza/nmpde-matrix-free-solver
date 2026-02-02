/* ---------------------------------------------------------------------
 *
 * Matrix-free Advection-Diffusion-Reaction Solver + Memory Benchmarking
 *
 * Solves: - div(mu * grad(u)) + div(beta * u) + gamma * u = f
 *
 * ---------------------------------------------------------------------
 */

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/memory_consumption.h> // Included for memory stats

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/diagonal_matrix.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <iostream>
#include <fstream>
#include <sys/resource.h> // POSIX standard for getrusage
#include  <default_coefficient.hpp>

namespace Step37_ADR
{
  using namespace dealii;
  // --------------------------------------------------------------------------
  // Problem Solver Class
  // --------------------------------------------------------------------------




int main(int argc, char *argv[])
{
  try
    {
      using namespace Step37_ADR;

      Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

      ADRProblem<dimension> adr_problem;
      adr_problem.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}