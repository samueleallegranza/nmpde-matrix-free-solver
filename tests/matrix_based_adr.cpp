#include <general_definitions.hpp>
#include <adr_mb_problem.hpp>

int main(int argc, char *argv[]) {
    using namespace dealii;
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv,1);

    MatrixBasedADR::ADRMBProblem<2> adr_problem;

    for (unsigned int refinement_cycle = 0; refinement_cycle < 3; refinement_cycle++) {
        adr_problem.run(refinement_cycle,"../input/params/test_mb.prm");
    }
}