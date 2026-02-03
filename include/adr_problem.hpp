#ifndef PROJECT7_MATRIXFREE_ADR_PROBLEM_HPP
#define PROJECT7_MATRIXFREE_ADR_PROBLEM_HPP

#include <general_definitions.hpp>
#include <default_coefficient.hpp>
#include <adr_operator.hpp>

using namespace dealii;

template <int dim>
class ADRProblem {
public:
    ADRProblem();
    void run(const unsigned int refinement, std::string filename);

private:
    void setup_system();
    void assemble_rhs();
    void solve();
    void output_results(std::string filename, const unsigned int cycle = 0) const;
    void print_memory_usage(const std::string &stage) const; // Added

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

    double             setup_time;
    ConditionalOStream pcout;
    ConditionalOStream time_details;
};

#endif //PROJECT7_MATRIXFREE_ADR_PROBLEM_HPP