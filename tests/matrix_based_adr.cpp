#include <general_definitions.hpp>
#include <adr_mb_problem.hpp>

int main(int argc, char *argv[]) {
    using namespace dealii;
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv,1);

    MatrixBasedADR::ADRMBProblem<2> adr_problem;
    adr_problem.run("../input/params/test_mb.prm");
}