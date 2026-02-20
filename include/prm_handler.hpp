/**
 * Library that handles the interaction with .prm files
 * and initialization of function parsers
 * @file prm_handler.hpp
 * @sa adr_problem.hpp adr_mb_problem.hpp
 */

#ifndef NNPDE_STUDY_PDEPARAMSHANDLER_HPP
#define NNPDE_STUDY_PDEPARAMSHANDLER_HPP

#include <general_definitions.hpp>

/**
 * Common namespace for both Matrix-Free and Matrix-Based solvers
 * @namespace ADR
 */
namespace ADR {
    using namespace dealii;

    /**
     * @brief Handler of .prm files for both Matrix-Free and Matrix-Based solvers.
     * @tparam dim the dimension of the
     */
    template <int dim>
    class ADRParamHandler {

    public:
        using String = std::string;
        using ConstantMap = std::map<String, double>;
        using BoundaryIds  = types::boundary_id;

        /// @brief The constructor initializes FunctionParsers and MPI
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

        /**
         * @brief Declares the prm structure
         *
         * The structure needs to be declared before
         * writing the default parameters file and before reading
         */
        void declare_parameters();

        /**
         * Initializes all the coefficients, function and parameters needed by ADR solvers
         * @param filename The file containing the right prm or xml structure
         */
        void init(const String &filename);

        /**
         * Prints the parameters with default values to a .prm file.
         * You need to run this only once to get an editable file with the correct structure.
         * @brief Outputs the default parameter file
         * @param filename The output file name, extension should be .prm
         */
        void print_parameters(const String &filename);
        /**
         * Prints the parameters with default values to a .xml file.
         * You need to run this only once to get an editable file with the correct structure.
         * Deal.ii has a convenient editor for .xml parameter files.
         * @brief Outputs the default parameter file
         * @param filename The output file name, extension should be .prm
         * @sa https://github.com/dealii/parameter_gui
         */
        void print_editable_parameters(const String &filename);

        /**
         * Filename to be used for all outputs, this is assumed to have no extension.
         * The solvers add extensions and modify the file name to add distinction between files.
         */
        String output_filename;

        /// @brief The actual prm handler class
        ParameterHandler prm;

        /// @brief FunctionParser for the diffusion coefficient
        FunctionParser<dim> diffusion_c;
        /// @brief FunctionParser for the advection coefficient
        FunctionParser<dim> advection_c;
        /// @brief FunctionParser for the reaction coefficient
        FunctionParser<dim> reaction_c;
        /// @brief FunctionParser for the force term
        FunctionParser<dim> force_term;

        /// @brief Vector containing the refinements to test the solver on
        std::vector<int> refinements;

        /// @brief The maximum number of iterations for the solvers
        unsigned int max_iters;
        /// @brief Decides whether CG or GMRES is used
        bool symmetric_solver;

        /// Vector containing the boundary tags (IDs) associated to the Dirichlet boundary condition with same index
        /// @sa dirichlet_bc
        std::vector<BoundaryIds> dirichlet_bc_tags;
        /// Vector containing the boundary tags (IDs) associated to the Neumann boundary condition with same index
        /// @sa neumann_bc
        std::vector<BoundaryIds> neumann_bc_tags;

        /// Vector containing the Dirichlet boundary condition associated to the boundary tags (IDs) with same index
        /// @sa dirichlet_bc_tags
        std::vector<std::unique_ptr<FunctionParser<dim>>> dirichlet_bc;
        /// Vector containing the Neumann boundary condition associated to the boundary tags (IDs) with same index
        /// @sa neumann_bc_tags
        std::vector<std::unique_ptr<FunctionParser<dim>>> neumann_bc;

        /// @brief The tolerance to apply when computing the solution
        double epsilon;

        /// @brief Control value to prevent re-declaration of the parameters
        bool param_initialized = false;
        /// @brief Control value to prevent re-initialization of the parameters
        bool initialized = false;

        private:
        /// @brief Utility to print a parameter file with any output style (only .prm and .xml used)
        void print_parameters_as(const String &filename,ParameterHandler::OutputStyle style);

        /// @brief The string "x,y,z" cut to the last variable that matters for that dimension
        String variables = FunctionParser<dim>::default_variable_names();
        /// @brief A map to mathematical constant, for example "pi" is associated to π
        ConstantMap constants;

        /// @brief The number of MPI processes. Needed to log in parallel
        const unsigned int mpi_size;
        /// @brief The MPI rank of this process. Needed to log in parallel
        const unsigned int mpi_rank;
        /// @brief An MPI conditional stream for logging
        ConditionalOStream pcout;

    };
}

template class ADR::ADRParamHandler<1>;
template class ADR::ADRParamHandler<2>;
template class ADR::ADRParamHandler<3>;

#endif //NNPDE_STUDY_PDEPARAMSHANDLER_HPP