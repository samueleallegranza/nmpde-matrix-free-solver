#ifndef PROJECT7_MATRIXFREE_ADR_PROBLEM_HPP
#define PROJECT7_MATRIXFREE_ADR_PROBLEM_HPP

#include <general_definitions.hpp>
#include <adr_operator.hpp>
#include <prm_handler.hpp>

namespace MatrixFreeADR {
    using namespace dealii;

    template <int dim>
    class ADRProblem : public ADR::ADRParamHandler<dim> {
    public:
        ADRProblem();
        void run(std::string param_filename);
        const LinearAlgebra::distributed::Vector<double>& get_solution();

    private:
        void setup_system(std::string param_filename);
        void assemble_rhs();
        void solve();
        void output_results(const unsigned int cycle = 0) const;
        void print_memory_usage() const; // Added
        void print_memory_usage_to_file(const unsigned int refinement = 0) const; // Added

        #ifdef DEAL_II_WITH_P4EST
            parallel::distributed::Triangulation<dim, dim> triangulation;
        #else
            Triangulation<dim> triangulation;
        #endif

        FE_Q<dim>       fe;
        DoFHandler<dim> dof_handler;

        MappingQ1<dim> mapping;

        AffineConstraints<double> constraints;
        using SystemMatrixType = ADROperator<dim, degree_finite_element, double>;
        SystemMatrixType system_matrix;

        MGConstrainedDoFs mg_constrained_dofs;
        using LevelMatrixType = ADROperator<dim, degree_finite_element, float>;
        MGLevelObject<LevelMatrixType> mg_matrices;

        LinearAlgebra::distributed::Vector<double> solution;
        LinearAlgebra::distributed::Vector<double> system_rhs;

        double setup_time;
        ConditionalOStream pcout;
        ConditionalOStream time_details;
    };

}

template class MatrixFreeADR::ADRProblem<1>;
template class MatrixFreeADR::ADRProblem<2>;
template class MatrixFreeADR::ADRProblem<3>;

#endif //PROJECT7_MATRIXFREE_ADR_PROBLEM_HPP