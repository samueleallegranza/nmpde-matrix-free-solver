#include <adr_problem.hpp>
#include <general_definitions.hpp>

int main(int argc, char *argv[]) {

    using namespace dealii;
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv,1);

    MatrixFreeADR::ADRProblem<3> adr_problem;
    adr_problem.declare_parameters();
    adr_problem.print_parameters("default.json");
    // adr_problem.run("../input/3D/nonhomo.prm");

}