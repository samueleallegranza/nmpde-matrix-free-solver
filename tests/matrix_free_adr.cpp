#include <adr_operator.hpp>
#include <adr_problem.hpp>
#include <general_definitions.hpp>

double compute_l2_error(const LinearAlgebra::distributed::Vector<double> &u1,
                        const LinearAlgebra::distributed::Vector<double> &u2)
{
    if (u1.size() != u2.size() || u1.locally_owned_size() != u2.locally_owned_size()) {
        std::cerr << "Error: Vectors have different sizes. Cannot compare." << std::endl;
        return -1.0;
    }

    double local_sum_sq = 0.0;

    for (unsigned int i = 0; i < u1.locally_owned_size(); ++i) {
        double diff = u1.local_element(i) - u2.local_element(i);
        local_sum_sq += diff * diff;
    }

    double global_sum_sq = Utilities::MPI::sum(local_sum_sq, MPI_COMM_WORLD);

    return std::sqrt(global_sum_sq);
}

int main(int argc, char *argv[]) {
    try {

        Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv,1);

        ADRProblem<2> adr_problem;

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