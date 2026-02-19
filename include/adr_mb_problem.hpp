#ifndef PROJECT7_MATRIXFREE_ADR_PROBLEM_HPP
#define PROJECT7_MATRIXFREE_ADR_PROBLEM_HPP

#include <general_definitions.hpp>
#include <prm_handler.hpp>

namespace MatrixBasedADR {
    using namespace dealii;

    template <int dim>
    struct ScratchData {
        ScratchData(
            const Mapping<dim> &mapping,
            const FiniteElement<dim> &fe,
            const unsigned int quadrature_degree,
            const UpdateFlags update_flags
        ) :
            fe_values(mapping, fe, QGauss<dim>(quadrature_degree), update_flags),
            fe_f_values(fe, QGauss<dim-1>(quadrature_degree+1), update_flags)
        {}

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

        FEValues<dim> fe_values;
        FEFaceValues<dim> fe_f_values;
    };

    struct CopyData {
        unsigned int level;
        FullMatrix<double> cell_matrix;
        Vector<double> cell_rhs;
        std::vector<types::global_dof_index> local_dof_indices;

        template <class Iterator>
        void reinit(const Iterator &cell, unsigned int dofs_per_cell) {
            cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
            cell_rhs.reinit(dofs_per_cell);

            local_dof_indices.resize(dofs_per_cell);
            cell->get_active_or_mg_dof_indices(local_dof_indices);
            level = cell->level();
        }
    };

    template <int dim>
    class ADRMBProblem : public ADR::ADRParamHandler<dim> {
    public:
        ADRMBProblem();
        void run(std::string param_filename);

    private:
        template <class Iterator>
        void cell_worker(
            const Iterator &cell,
            ScratchData<dim> &scratch_data,
            CopyData &copy_data
        );
        void setup_system(std::string param_filename);
        void assemble_system();
        void assemble_multigrid();
        void refine_grid();
        void solve();
        void output_results(const unsigned int cycle = 0) const;
        void print_memory_usage() const; // Added
        void print_memory_usage_to_file(const unsigned int refinement = 0) const; // Added

        #ifdef DEAL_II_WITH_P4EST
            parallel::distributed::Triangulation<dim, dim> triangulation;
        #else
            Triangulation<dim> triangulation;
        #endif

        FE_Q<dim> fe;
        DoFHandler<dim> dof_handler;

        SparsityPattern sparsity_pattern;
        SparseMatrix<double> system_matrix;

        AffineConstraints<double> constraints;

        MappingQ1<dim> mapping;

        Vector<double> solution;
        Vector<double> system_rhs;

        MGLevelObject<SparsityPattern> mg_sparsity_patterns;
        MGLevelObject<SparsityPattern> mg_interface_sparsity_patterns;

        MGLevelObject<SparseMatrix<double>> mg_matrices;
        MGLevelObject<SparseMatrix<double>> mg_interface_matrices;
        MGConstrainedDoFs mg_constrained_dofs;

        double             setup_time;
        ConditionalOStream pcout;
        ConditionalOStream time_details;
    };

}

template class MatrixBasedADR::ADRMBProblem<1>;
template class MatrixBasedADR::ADRMBProblem<2>;
template class MatrixBasedADR::ADRMBProblem<3>;

#endif //PROJECT7_MATRIXFREE_ADR_PROBLEM_HPP