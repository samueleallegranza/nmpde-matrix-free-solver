#include <adr_problem.hpp>
#include <general_definitions.hpp>

int main(int argc, char *argv[]) {
    using namespace dealii;
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv,1);

    MatrixFreeADR::ADRProblem<2> adr_problem;
    adr_problem.run("../input/params/test_mf.prm");

}