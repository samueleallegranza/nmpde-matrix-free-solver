#ifndef PROJECT7_MATRIXBASED_ADR_PROBLEM_HPP
#define PROJECT7_MATRIXBASED_ADR_PROBLEM_HPP

#include <general_definitions.hpp>
#include <default_coefficient.hpp>

// Additional includes for matrix-based approach
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_tools.h>

template <int dim>
class ADRProblemMB {
public:
    ADRProblemMB();
    void run(unsigned int refinement, std::string filename);

private:
    void setup_system();
    void assemble_system();
    void solve();
    void output_results(std::string filename, const unsigned int cycle = 0) const;
    void print_memory_usage(const std::string &stage) const;

    MPI_Comm mpi_communicator;

    #ifdef DEAL_II_WITH_P4EST
        parallel::distributed::Triangulation<dim, dim> triangulation;
    #else
        Triangulation<dim> triangulation;
    #endif

    FE_Q<dim>       fe;
    DoFHandler<dim> dof_handler;
    MappingQ1<dim>  mapping;

    AffineConstraints<double> constraints;

    TrilinosWrappers::SparseMatrix  system_matrix;
    TrilinosWrappers::MPI::Vector   solution;
    TrilinosWrappers::MPI::Vector   system_rhs;

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    double             setup_time;
    ConditionalOStream pcout;
    ConditionalOStream time_details;
};


template <int dim>
ADRProblemMB<dim>::ADRProblemMB()
    : mpi_communicator(MPI_COMM_WORLD)
#ifdef DEAL_II_WITH_P4EST
    , triangulation(
        mpi_communicator,
        Triangulation<dim>::limit_level_difference_at_vertices)
#else
    , triangulation(Triangulation<dim>::limit_level_difference_at_vertices)
#endif
    , fe(degree_finite_element)
    , dof_handler(triangulation)
    , setup_time(0.0)
    , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    , time_details(
        std::cout,
        false &&
        Utilities::MPI::this_mpi_process(mpi_communicator) == 0
        ) {
}


template <int dim>
void ADRProblemMB<dim>::print_memory_usage(const std::string &stage) const {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    double local_memory_mb = usage.ru_maxrss / 1024.0;

    #ifdef __APPLE__
      local_memory_mb /= 1024.0;
    #endif

    double max_memory_mb = Utilities::MPI::max(local_memory_mb, mpi_communicator);
    double min_memory_mb = Utilities::MPI::min(local_memory_mb, mpi_communicator);

    pcout << "  Memory (" << stage << "):        "
          << "Max: " << max_memory_mb << " MB / Min: " << min_memory_mb << " MB"
          << std::endl;
}


template <int dim>
void ADRProblemMB<dim>::setup_system() {
    Timer time;
    setup_time = 0;

    dof_handler.distribute_dofs(fe);

    pcout << "   Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

    locally_owned_dofs    = dof_handler.locally_owned_dofs();
    locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

    solution.reinit(locally_owned_dofs, mpi_communicator);
    system_rhs.reinit(locally_owned_dofs, mpi_communicator);

    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    VectorTools::interpolate_boundary_values(
        mapping,
        dof_handler,
        0,
        DirichletBoundaryCondition<dim>(),
        constraints
    );
    constraints.close();

    // Build sparsity pattern
    TrilinosWrappers::SparsityPattern sp(locally_owned_dofs, mpi_communicator);
    DoFTools::make_sparsity_pattern(dof_handler, sp, constraints, false);
    sp.compress();

    system_matrix.reinit(sp);

    setup_time += time.wall_time();
    time_details << "Distribute DoFs & B.C.     (CPU/wall) "
                 << time.cpu_time() << "s/" << time.wall_time() << 's' << std::endl;

    print_memory_usage("After Setup");
}


template <int dim>
void ADRProblemMB<dim>::assemble_system() {
    Timer time;

    system_matrix = 0;
    system_rhs    = 0;

    QGauss<dim> quadrature_formula(fe.degree + 1);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points    = quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // Same coefficient objects as matrix-free version
    DiffusionCoefficient<dim> mu_func;
    AdvectionCoefficient<dim> beta_func;
    ReactionCoefficient<dim>  gamma_func;
    ForceTerm<dim>            f_func;

    for (const auto &cell : dof_handler.active_cell_iterators()) {
        if (cell->is_locally_owned()) {
            fe_values.reinit(cell);
            cell_matrix = 0;
            cell_rhs    = 0;

            for (unsigned int q = 0; q < n_q_points; ++q) {
                const Point<dim> &p = fe_values.quadrature_point(q);
                const double dx     = fe_values.JxW(q);

                // Evaluate coefficients from default_coefficient.hpp
                const double mu    = mu_func.value(p);
                const double gamma = gamma_func.value(p);
                const double f_val = f_func.value(p);

                // Get advection vector
                Tensor<1, dim> beta = beta_func.template vector_value<double>(p);

                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                    const double         phi_i      = fe_values.shape_value(i, q);
                    const Tensor<1, dim> grad_phi_i = fe_values.shape_grad(i, q);

                    for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                        const double         phi_j      = fe_values.shape_value(j, q);
                        const Tensor<1, dim> grad_phi_j = fe_values.shape_grad(j, q);

                        // Diffusion: mu * grad(phi_j) . grad(phi_i)
                        // Advection: (beta . grad(phi_j)) * phi_i
                        // Reaction:  gamma * phi_j * phi_i
                        cell_matrix(i, j) +=
                            (mu * grad_phi_j * grad_phi_i +
                             (beta * grad_phi_j) * phi_i +
                             gamma * phi_j * phi_i
                            ) * dx;
                    }
                    // RHS: f * phi_i
                    cell_rhs(i) += f_val * phi_i * dx;
                }
            }

            cell->get_dof_indices(local_dof_indices);
            constraints.distribute_local_to_global(
                cell_matrix, cell_rhs, local_dof_indices,
                system_matrix, system_rhs);
        }
    }

    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);

    setup_time += time.wall_time();
    time_details << "Assemble system            (CPU/wall) "
                 << time.cpu_time() << "s/" << time.wall_time() << 's' << std::endl;

    print_memory_usage("After Assembly");
}


template <int dim>
void ADRProblemMB<dim>::solve() {
    Timer time;

    TrilinosWrappers::PreconditionSSOR preconditioner;
    preconditioner.initialize(system_matrix,
                              TrilinosWrappers::PreconditionSSOR::AdditionalData(1.0));

    const int    max_iterations = 500;
    const double tolerance      = 1.0e-12;
    SolverControl solver_control(max_iterations, tolerance * system_rhs.l2_norm());

    TrilinosWrappers::SolverGMRES solver(solver_control);

    pcout << "  Solving the linear system..." << std::endl;

    try {
        solver.solve(system_matrix, solution, system_rhs, preconditioner);
    } catch (std::exception &e) {
        pcout << "Solver failed: " << e.what() << std::endl;
    }

    constraints.distribute(solution);

    setup_time += time.wall_time();
    pcout << "Time solve (" << solver_control.last_step() << " iterations)"
          << (solver_control.last_step() < 10 ? "  " : " ")
          << "(CPU/wall) " << time.cpu_time() << "s/"
          << time.wall_time() << "s\n";

    print_memory_usage("After Solve");
}


template <int dim>
void ADRProblemMB<dim>::output_results(std::string filename, const unsigned int cycle) const {
    Timer time;

    if (triangulation.n_global_active_cells() > MAX_OUTPUT_MESH_ELEMENTS) {
        pcout << "File too big" << std::endl;
        return;
    }

    // Copy solution to a vector with ghost values for output
    TrilinosWrappers::MPI::Vector locally_relevant_solution(
        locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    locally_relevant_solution = solution;

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(locally_relevant_solution, "solution");
    data_out.build_patches(mapping);

    DataOutBase::VtkFlags flags;
    flags.compression_level = DataOutBase::CompressionLevel::best_speed;
    data_out.set_flags(flags);
    data_out.write_vtu_with_pvtu_record("./", filename, cycle, mpi_communicator, 3);

    pcout << "Time write output          (CPU/wall) "
          << time.cpu_time() << "s/" << time.wall_time() << "s\n";
}


template <int dim>
void ADRProblemMB<dim>::run(unsigned int refinement, std::string filename) {
    const unsigned int n_vect_doubles = VectorizedArray<double>::size();
    const unsigned int n_vect_bits    = 8 * sizeof(double) * n_vect_doubles;

    pcout << "===============================================" << std::endl;
    pcout << "Matrix-Based ADR Solver" << std::endl;
    pcout << "Vectorization over " << n_vect_doubles
          << " doubles = " << n_vect_bits << " bits ("
          << Utilities::System::get_current_vectorization_level() << ')'
          << std::endl;
    pcout << "===============================================" << std::endl;

    triangulation.clear();
    GridGenerator::hyper_cube(triangulation, 0., 1.);
    triangulation.refine_global(2 + refinement);

    setup_system();
    assemble_system();
    solve();
    output_results(filename, refinement);
}

#endif //PROJECT7_MATRIXBASED_ADR_PROBLEM_HPP
