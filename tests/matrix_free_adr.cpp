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

using namespace dealii;

int main(int argc, char *argv[]) {
    try {

        Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv,1);

        ADRProblem<2> adr_problem("../input/params/parameters_g.xml");;

        for (unsigned int refinement_cycle = 0; refinement_cycle < 3; refinement_cycle++) {
            adr_problem.run(refinement_cycle, "solution-matrix-free");
        }

    }
    catch (std::exception &exc) {
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
    catch (...) {
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