#include "advection_diffusion_problem.h"
#include <deal.II/base/utilities.h>
#include <deal.II/base/mpi.h>

int main(int argc, char *argv[])
{
  try
  {
    using namespace dealii;

    // TODO: Initialize MPI if needed for parallel runs
    // Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    // Choose dimension and polynomial degree
    const int dim = 2;
    const int fe_degree = 2;

    std::cout << "========================================" << std::endl;
    std::cout << "Project 7: Matrix-Free Solvers" << std::endl;
    std::cout << "Advection-Diffusion-Reaction Problem" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Dimension: " << dim << std::endl;
    std::cout << "FE degree: " << fe_degree << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Create and run problem
    AdvectionDiffusionProblem<dim, fe_degree> problem;
    problem.run();

    std::cout << "\n========================================" << std::endl;
    std::cout << "Computation finished successfully!" << std::endl;
    std::cout << "========================================" << std::endl;
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
