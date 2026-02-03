/* ---------------------------------------------------------------------
 *
 * Matrix-free Advection-Diffusion-Reaction Solver + Memory Benchmarking
 *
 * Solves: - div(mu * grad(u)) + div(beta * u) + gamma * u = f
 *
 * ---------------------------------------------------------------------
 */

#include <default_coefficient.hpp>
#include <adr_operator.hpp>
#include <adr_problem.hpp>
#include <general_definitions.hpp>


namespace ADR_Problem_Test {
    using namespace dealii;

    int main(int argc, char *argv[])
    {
        try {
            using namespace ADR_Problem_Test;

            Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

            ADRProblem<dimension> adr_problem;

            for (int refinement_cycle = 0; refinement_cycle < 3; refinement_cycle++) {
                adr_problem.run(refinement_cycle, "solution");
            }

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
}