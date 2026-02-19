#include <adr_problem.hpp>
#include <general_definitions.hpp>

int main(int argc, char *argv[]) {
    using namespace dealii;
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv,1);

    MatrixFreeADR::ADRProblem<3> adr_problem;
    adr_problem.run("../input/params/pb_3d_mf.prm");
}