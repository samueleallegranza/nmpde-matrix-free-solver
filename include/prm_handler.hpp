#ifndef NNPDE_STUDY_PDEPARAMSHANDLER_HPP
#define NNPDE_STUDY_PDEPARAMSHANDLER_HPP

#include <general_definitions.hpp>

namespace ADR {
    using namespace dealii;

    template <int dim>
    class ADRParamHandler {

    public:
        using String = std::string;
        using ConstantMap = std::map<String, double>;
        using BoundaryIds  = types::boundary_id;

        ADRParamHandler() :
        diffusion_c(1),
        advection_c(dim),
        reaction_c(1),
        force_term(1),
        mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)),
        mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)),
        pcout(std::cout, mpi_rank == 0) {

        }
        ~ADRParamHandler() = default;

        void declare_parameters();
        void init(const String &filename);
        void print_parameters(const String &filename);
        void print_editable_parameters(const String &filename);

        String output_filename;
        ParameterHandler prm;
        FunctionParser<dim> diffusion_c,
                            advection_c,
                            reaction_c,
                            force_term;
        unsigned int max_iters;
        String preconditioner;
        std::vector<BoundaryIds> dirichlet_bc_tags, neumann_bc_tags;
        std::vector<int> refinements;
        std::vector<std::unique_ptr<FunctionParser<dim>>> dirichlet_bc, neumann_bc;
        double epsilon;
        bool symmetric_solver;
        bool param_initialized = false;
        bool initialized = false;

        private:
        void print_parameters_as(const String &filename,ParameterHandler::OutputStyle style);

        String variables = FunctionParser<dim>::default_variable_names();
        ConstantMap constants;

        const unsigned int mpi_size;
        const unsigned int mpi_rank;
        ConditionalOStream pcout;

    };
}
template class ADR::ADRParamHandler<1>;
template class ADR::ADRParamHandler<2>;
template class ADR::ADRParamHandler<3>;

#endif //NNPDE_STUDY_PDEPARAMSHANDLER_HPP