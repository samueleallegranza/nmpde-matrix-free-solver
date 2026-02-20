/**
 * Library that solves ADR problems using a Matrix-Based operator
 * @file adr_mb_problem.hpp
 */

#ifndef PROJECT7_MATRIXFREE_ADR_PROBLEM_HPP
#define PROJECT7_MATRIXFREE_ADR_PROBLEM_HPP

#include <general_definitions.hpp>
#include <prm_handler.hpp>

/**
 * Namespace for Matrix based ADR solvers
 * @namespace MatrixBasedADR
 */
namespace MatrixBasedADR {
    using namespace dealii;

    /**
     * @brief Stores temporary data on each cell
     * @tparam dim the dimension of the problem
     */
    template <int dim>
    struct ScratchData {
        /**
         * @param mapping The FE mapping
         * @param fe the finite element object
         * @param quadrature_degree the quadrature object
         * @param update_flags the flags to pass to the finite element object
         */
        ScratchData(
            const Mapping<dim> &mapping,
            const FiniteElement<dim> &fe,
            const unsigned int quadrature_degree,
            const UpdateFlags update_flags
        ) :
            fe_values(mapping, fe, QGauss<dim>(quadrature_degree), update_flags),
            fe_f_values(fe, QGauss<dim-1>(quadrature_degree+1), update_flags)
        {}

        /**
         * Copy constructor
         * @param scratch_data the other object
         */
        ScratchData(const ScratchData<dim> &scratch_data) :
            fe_values(
                scratch_data.fe_values.get_mapping(),
                scratch_data.fe_values.get_fe(),
                scratch_data.fe_values.get_quadrature(),
                scratch_data.fe_values.get_update_flags()
            ),
            fe_f_values(
                scratch_data.fe_f_values.get_fe(),
                scratch_data.fe_f_values.get_quadrature(),
                scratch_data.fe_f_values.get_update_flags()
            )
        {}

        /// @brief Evaluator of finite elements
        FEValues<dim> fe_values;
        /// @brief Evaluator of finite elements on the faces
        FEFaceValues<dim> fe_f_values;
    };

    /**
     * @brief Contains the output of each cell assembly
     */
    struct CopyData {
        /// @brief the Multigrid level
        unsigned int level;
        /// @brief the matrix of the cell we are working on
        FullMatrix<double> cell_matrix;
        /// @brief the rhs ofthe cell we are working on
        Vector<double> cell_rhs;
        /// @brief A map from global to local DoFs (and vice versa)
        std::vector<types::global_dof_index> local_dof_indices;

        /**
         * @brief Initializes objects on the current cell
         * @tparam Iterator The cell iterator
         * @param cell The cell we are working on
         * @param dofs_per_cell The DoFs of the cell we are working on
         */
        template <class Iterator>
        void reinit(const Iterator &cell, unsigned int dofs_per_cell) {
            cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
            cell_rhs.reinit(dofs_per_cell);

            local_dof_indices.resize(dofs_per_cell);
            cell->get_active_or_mg_dof_indices(local_dof_indices);
            level = cell->level();
        }
    };

    /**
     * @brief An ADR problem solver
     * @tparam dim the dimension of the problem
     */
    template <int dim>
    class ADRMBProblem : public ADR::ADRParamHandler<dim> {
    public:
        ADRMBProblem();

        /**
         * Utility to run the solver on different mesh refinements in sequence
         * @param param_filename the name of the file containing the ADR parameters
         */
        void run(std::string param_filename);

    private:
        /**
         * @brief Performs work on a single cell
         * @tparam Iterator The cell iterator
         * @param cell The cell to work on
         * @param scratch_data A ScratchData structure with the necessary information
         * @param copy_data A CopyData structure with the necessary information
         */
        template <class Iterator>
        void cell_worker(
            const Iterator &cell,
            ScratchData<dim> &scratch_data,
            CopyData &copy_data
        );

        /**
         * @brief Setup MG and FE
         *
         * This function setups the Sparse Matrices, Vectors, Finite Elements and Quadratures
         * for all levels of Multigrid
         * @param param_filename The file name to be used to initialize all the ADR parameters
         */
        void setup_system(std::string param_filename);

        /**
         * @brief Assembles the first level of the Multigrid matrix
         */
        void assemble_system();

        /**
         * @brief Assembles all the levels of the Multigrid matrices
         */
        void assemble_multigrid();

        /**
         * @brief refines the grid adaptively
         * @note Not used for fair comparison
         */
        void refine_grid();

        /**
         * @brief solves the algebraic system
         *
         * Solves the linear problem using Multigrid with Jacobi smoother.
         * Either CG or GMRES iterations are used with Jacobi preconditioner.
         * We do assume the matrix to have all diagonal elements and be somewhat "symmetric".
         * We use HouseHolder solver on the coarsest level
         */
        void solve();

        /**
         * @brief prints the results of the specified cycle to a .pvtu file
         * @param cycle The cycle number, essentially the refinement level
         */
        void output_results(const unsigned int cycle = 0) const;

        /**
         * @brief Logs the memory usage
         */
        void print_memory_usage() const;

        /**
         * @brief prints the memory usage to a .csv file
         * @param refinement The refinement level
         */
        void print_memory_usage_to_file(const unsigned int refinement = 0) const;

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

        /// @brief The class that creates the sparsity pattern for the matrix
        SparsityPattern sparsity_pattern;
        /// @brief The class that handles the sparse matrix
        SparseMatrix<double> system_matrix;

        /// @brief Constraint to be passed at each level of Multigrid
        AffineConstraints<double> constraints;

        /// @brief A mapping used by Multigrid to unsure continuity
        MappingQ1<dim> mapping;

        /// @brief A vector containing the solution
        Vector<double> solution;
        /// @brief The RHS vector
        Vector<double> system_rhs;

        /// @brief A vector containing sparsity pattern information for each Multigrid level
        MGLevelObject<SparsityPattern> mg_sparsity_patterns;
        /// @brief A vector containing sparsity pattern information for interfaces on each Multigrid level
        MGLevelObject<SparsityPattern> mg_interface_sparsity_patterns;

        /// @brief A vector containing sparse matrices for each Multigrid level
        MGLevelObject<SparseMatrix<double>> mg_matrices;
        /// @brief A vector containing sparse matrices that only work on the interfaces for each Multigrid level
        MGLevelObject<SparseMatrix<double>> mg_interface_matrices;
        /// @brief A structure to pass constraints to each level of Multigrid
        MGConstrainedDoFs mg_constrained_dofs;

        /// @brief An MPI conditional stream for logging
        ConditionalOStream pcout;
    };

}

template class MatrixBasedADR::ADRMBProblem<1>;
template class MatrixBasedADR::ADRMBProblem<2>;
template class MatrixBasedADR::ADRMBProblem<3>;

#endif //PROJECT7_MATRIXFREE_ADR_PROBLEM_HPP