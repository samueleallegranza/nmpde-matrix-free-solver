#include <general_definitions.hpp>
#include <adr_mb_problem.hpp>

using namespace dealii;

int main(int argc, char *argv[]) {
    try {

        Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv,1);

        ADRMBProblem<2> adr_problem;

        for (unsigned int refinement_cycle = 0; refinement_cycle < 3; refinement_cycle++) {
            adr_problem.run(refinement_cycle,"../input/params/test_mf.prm");
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