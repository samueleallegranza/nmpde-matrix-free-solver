/**
 * Library that solves ADR problems using a Matrix-Free operator
 * @file adr_problem.hpp
 * @sa adr_operator.hpp
 */

#ifndef PROJECT7_MATRIXFREE_ADR_PROBLEM_HPP
#define PROJECT7_MATRIXFREE_ADR_PROBLEM_HPP

#include <general_definitions.hpp>
#include <adr_operator.hpp>
#include <prm_handler.hpp>

/**
 * Namespace for Matrix free ADR solvers
 * @namespace MatrixFreeADR
 */
namespace MatrixFreeADR {
    using namespace dealii;

    /**
     * @brief An ADR problem solver that uses Matrix-Free operators
     * @tparam dim the dimension of the problem
     */
    template <int dim>
    class ADRProblem : public ADR::ADRParamHandler<dim> {
    public:
        ADRProblem();

        /**
         * Utility to run the solver on different mesh refinements in sequence
         * @param param_filename The name of the file containing the ADR parameters
         */
        void run(std::string param_filename);

        /**
         * @brief Getter for the solution vector
         * @return The solution vector
         */
        const LinearAlgebra::distributed::Vector<double>& get_solution();

    private:
        /**
         * @brief Setups the MG and FEs
         *
         * This function sets up finite elements spaces and quadrature for all levels of multigrid
         * @param param_filename The file name to be used to initialize all the ADR parameters
         */
        void setup_system(std::string param_filename);

        /**
         * @brief Assembler of the rhs
         *
         * Assemble the RHS of the system, this accounts for the force term Neumann boundary conditions
         * and subtract the \f$Au_0\f$ where \f$u_0\f$ is a vector
         * that only has dirichlet values on the Dirichlet boundaries
         */
        void assemble_rhs();

        /**
         * @brief solves the algebraic system
         *
         * Solves the linear problem using Multigrid with Jacobi smoother.
         * Either CG or GMRES iterations are used with Jacobi preconditioner.
         * We do assume the matrix to have all diagonal elements and be somewhat "symmetric"
         */
        void solve();

        /**
         * @brief prints the results of the specified cycle to a .pvtu file
         * @param cycle The cycle number, essentially the refinement level
         */
        void output_results(const int cycle = 0) const;

        /**
         * @brief Logs the memory usage
         */
        void print_memory_usage() const;

        /**
         * @brief prints the memory usage to a .csv file
         * @param refinement The refinement level
         */
        void print_memory_usage_to_file(const int refinement = 0) const;

        #ifdef DEAL_II_WITH_P4EST
            /// @brief The mesh
            parallel::distributed::Triangulation<dim, dim> triangulation;
        #else
            /// @brief The mesh
            Triangulation<dim> triangulation;
        #endif

        /// @brief The finite element for square grids
        FE_Q<dim> fe;
        /// @brief The class handling the DoFs
        DoFHandler<dim> dof_handler;

        /// @brief A mapping used by Multigrid to unsure continuity
        MappingQ1<dim> mapping;

        /// @brief Constraint to be passed at each level of Multigrid
        AffineConstraints<double> constraints;
        using SystemMatrixType = ADROperator<dim, degree_finite_element, double>;
        /// @brief The Matrix-Free operator, essentially the sparse matrix of the system
        SystemMatrixType system_matrix;

        /// @brief A structure to pass constraints to each level of Multigrid
        MGConstrainedDoFs mg_constrained_dofs;
        using LevelMatrixType = ADROperator<dim, degree_finite_element, float>;
        /// @brief A collection of Matrix-Free operator for the coarser grids
        MGLevelObject<LevelMatrixType> mg_matrices;

        /// @brief A vector containing the solution
        LinearAlgebra::distributed::Vector<double> solution;
        /// @brief The RHS vector
        LinearAlgebra::distributed::Vector<double> system_rhs;

        /// @brief An MPI conditional stream for logging
        ConditionalOStream pcout;
    };

}

template class MatrixFreeADR::ADRProblem<1>;
template class MatrixFreeADR::ADRProblem<2>;
template class MatrixFreeADR::ADRProblem<3>;

#endif //PROJECT7_MATRIXFREE_ADR_PROBLEM_HPP