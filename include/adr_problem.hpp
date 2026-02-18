#ifndef PROJECT7_MATRIXFREE_ADR_PROBLEM_HPP
#define PROJECT7_MATRIXFREE_ADR_PROBLEM_HPP

#include <general_definitions.hpp>
#include <adr_operator.hpp>
#include <prm_handler.hpp>

template <int dim>
class ADRProblem : public ADR::EllipticParamHandler<dim> {
public:
    ADRProblem();
    void run(unsigned int refinement,std::string param_filename);

private:
    void setup_system(std::string param_filename);
    void assemble_rhs();
    void solve();
    void output_results(const unsigned int cycle = 0) const;
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


template <int dim>
ADRProblem<dim>::ADRProblem() :
    ADR::EllipticParamHandler<dim>(),
#ifdef DEAL_II_WITH_P4EST
    triangulation(
        MPI_COMM_WORLD,
        Triangulation<dim>::limit_level_difference_at_vertices,
        parallel::distributed::Triangulation<dim, dim>::construct_multigrid_hierarchy
        )
#else
    triangulation(Triangulation<dim>::limit_level_difference_at_vertices)
#endif
    , fe(degree_finite_element)
    , dof_handler(triangulation)
    , setup_time(0.0)
    , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    , time_details(
        std::cout,
        false &&
        Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0
        )
{
}


//! TODO to remove
template <int dim>
void ADRProblem<dim>::print_memory_usage(const std::string &stage) const {
      struct rusage usage;
      getrusage(RUSAGE_SELF, &usage);
      double local_memory_mb = usage.ru_maxrss / 1024.0;

      // macOS correction (KB -> MB) vs Linux (already KB)
      #ifdef __APPLE__
        local_memory_mb /= 1024.0;
      #endif

      double max_memory_mb = Utilities::MPI::max(local_memory_mb, MPI_COMM_WORLD);
      double min_memory_mb = Utilities::MPI::min(local_memory_mb, MPI_COMM_WORLD);

      pcout << "  Memory (" << stage << "):        "
            << "Max: " << max_memory_mb << " MB / Min: " << min_memory_mb << " MB"
            << std::endl;
  }


template <int dim>
void ADRProblem<dim>::setup_system(std::string param_filename) {

    this->init(param_filename);

    Timer time;
    setup_time = 0;

    system_matrix.clear();
    mg_matrices.clear_elements();

    dof_handler.distribute_dofs(fe);
    dof_handler.distribute_mg_dofs();

    LOG_VAR("Number of DoFs" , dof_handler.n_dofs());

    const IndexSet locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    std::map<types::boundary_id, const Function<dim> *> boundary_functions;
    for (long unsigned int i = 0; i < this->dirichlet_bc_tags.size(); i++)
        boundary_functions[this->dirichlet_bc_tags[i]] = this->dirichlet_bc[i].get();

    VectorTools::interpolate_boundary_values(
        mapping,
        dof_handler,
        boundary_functions,
        constraints
    );
    constraints.close();

    setup_time += time.wall_time();

    time_details << "Distribute DoFs & B.C.     (CPU/wall) " << time.cpu_time() << "s/" << time.wall_time() << 's' << std::endl;
    time.restart();

    // initialize matrix free system for level 0 of MG
    {
        typename MatrixFree<dim, double>::AdditionalData additional_data;

        additional_data.tasks_parallel_scheme = MatrixFree<dim, double>::AdditionalData::none; // we don't use multi thread yet
        additional_data.mapping_update_flags = (
            update_gradients |
            update_JxW_values |
            update_quadrature_points |
            update_values
            );

        additional_data.mapping_update_flags_boundary_faces = (
            update_values |
            update_JxW_values |
            update_quadrature_points |
            update_normal_vectors |
            update_gradients
            );

        std::shared_ptr<MatrixFree<dim, double>> system_mf_storage(new MatrixFree<dim, double>());
        system_mf_storage->reinit(
            mapping,
            dof_handler,
            constraints,
            QGauss<1>(fe.degree + 1),
            additional_data
            );
        system_matrix.initialize(system_mf_storage);
    }

    system_matrix.evaluate_coefficients(this->diffusion_c, this->advection_c, this->reaction_c);

    system_matrix.initialize_dof_vector(solution);
    system_matrix.initialize_dof_vector(system_rhs);

    setup_time += time.wall_time();
    time_details << "Setup matrix-free system   (CPU/wall) " << time.cpu_time() << "s/" << time.wall_time() << 's' << std::endl;
    time.restart();

    const unsigned int nlevels = triangulation.n_global_levels();
    mg_matrices.resize(0, nlevels - 1);

    mg_constrained_dofs.initialize(dof_handler);
    // initialize dof for boundaries on level 0 of multigrid
    std::set<types::boundary_id> boundary_ids(this->dirichlet_bc_tags.begin(), this->dirichlet_bc_tags.end());
    mg_constrained_dofs.make_zero_boundary_constraints(dof_handler,boundary_ids);

    // initialize matrix free system for level>=1 of MG
    for (unsigned int level = 0; level < nlevels; ++level) {
        const IndexSet relevant_dofs = DoFTools::extract_locally_relevant_level_dofs(dof_handler, level);
        AffineConstraints<double> level_constraints;
        level_constraints.reinit(relevant_dofs);
        level_constraints.add_lines(mg_constrained_dofs.get_boundary_indices(level));
        level_constraints.close();

        typename MatrixFree<dim, float>::AdditionalData additional_data;
        additional_data.tasks_parallel_scheme = MatrixFree<dim, float>::AdditionalData::none;  // we don't use multi thread yet
        additional_data.mapping_update_flags = (
            update_gradients |
            update_JxW_values |
            update_quadrature_points |
            update_values
            );

        additional_data.mg_level = level;

        std::shared_ptr<MatrixFree<dim, float>> mg_mf_storage_level(new MatrixFree<dim, float>());
        mg_mf_storage_level->reinit(
            mapping,
            dof_handler,
            level_constraints,
            QGauss<1>(fe.degree + 1),
            additional_data
        );

        mg_matrices[level].initialize(
            mg_mf_storage_level,
            mg_constrained_dofs,
            level
            );

        mg_matrices[level].evaluate_coefficients(this->diffusion_c,this->advection_c,this->reaction_c);
    }
    setup_time += time.wall_time();
    time_details << "Setup matrix-free levels   (CPU/wall) " << time.cpu_time() << "s/" << time.wall_time() << 's' << std::endl;

    //! TODO remove
    print_memory_usage("After Setup"); // Log memory
}


template <int dim>
void ADRProblem<dim>::assemble_rhs() {
    Timer time;

    solution = 0;
    constraints.distribute(solution);
    solution.update_ghost_values();
    system_rhs = 0;

    Table<2, VectorizedArray<double>> &mu_values = system_matrix.mu_values;
    Table<2, Tensor<1, dim, VectorizedArray<double>>> &beta_values = system_matrix.beta_values;
    Table<2, VectorizedArray<double>> &gamma_values = system_matrix.gamma_values;

    FEEvaluation<dim, degree_finite_element> phi(*system_matrix.get_matrix_free());

    for (unsigned int cell = 0; cell < system_matrix.get_matrix_free()->n_cell_batches(); ++cell) {
        phi.reinit(cell);
        phi.read_dof_values_plain(solution);
        phi.evaluate(EvaluationFlags::gradients | EvaluationFlags::values);

        for (unsigned int q = 0; q < phi.n_q_points; ++q) {
            const Point<dim,VectorizedArray<double> > p_vect = phi.quadrature_point(q);

            const auto u_val  = phi.get_value(q);
            const auto u_grad = phi.get_gradient(q);

            phi.submit_gradient(-mu_values(cell, q) * u_grad, q);
            phi.submit_value(-beta_values(cell, q) * u_grad -gamma_values(cell, q) * u_val, q);

            VectorizedArray<double> f_value = 0.0;
            for (unsigned int v=0; v<VectorizedArray<double>::size(); ++v) {
                Point<dim> p;
                for (unsigned int d=0; d<dim; ++d) p[d] = p_vect[d][v];
                f_value[v] = this->force_term.value(p);
            }
            //phi.submit_value((f.value(phi.quadrature_point(q))  + neumann.value(phi.quadrature_point(q)) ) * phi.get_value(q), q);
            // We submit the value of the force term at the quadrature point.
            phi.submit_value(f_value, q);
        }

        // This performs the actual integration with the values (multiplication by test functions and weights)
        phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);

        phi.distribute_local_to_global(system_rhs);
    }

    FEFaceEvaluation<dim, degree_finite_element> phi_face(*system_matrix.get_matrix_free());

    for (unsigned int b_face = 0; b_face < system_matrix.get_matrix_free()->n_boundary_face_batches(); ++b_face) {
        auto face_id = system_matrix.get_matrix_free()->get_boundary_id(b_face);
        auto pos = std::find(this->neumann_bc_tags.begin(),this->neumann_bc_tags.end(),face_id);
        if (pos != this->neumann_bc_tags.end()) {
            phi_face.reinit(b_face);
            auto neumann_index = pos - this->neumann_bc_tags.begin();
            for (unsigned int q = 0; q < phi_face.n_q_points; ++q) {
                const Point<dim,VectorizedArray<double> > p_vect = phi_face.quadrature_point(q);
                VectorizedArray<double> neumann_value = 0.0;
                for (unsigned int v=0; v<VectorizedArray<double>::size(); ++v) {
                    Point<dim> p;
                    for (unsigned int d=0; d<dim; ++d) p[d] = p_vect[d][v];
                    neumann_value[v] = this->neumann_bc[neumann_index]->value(p) / this->diffusion_c.value(p); // nabla( u ) * n = gamma / mu = h
                }
                phi_face.submit_value(neumann_value, q);
            }

            // Integrate (multiply by test function v and Jacobian dS)
            phi_face.integrate(EvaluationFlags::values);
            phi_face.distribute_local_to_global(system_rhs);
        }
    }

    system_rhs.compress(VectorOperation::add);

    setup_time += time.wall_time();
    time_details << "Assemble right hand side   (CPU/wall) " << time.cpu_time() << "s/" << time.wall_time() << 's' << std::endl;

    print_memory_usage("After Assembly"); // Log memory
}


template <int dim>
void ADRProblem<dim>::solve() {
    Timer time;
    MGTransferMatrixFree<dim, float> mg_transfer(mg_constrained_dofs);
    mg_transfer.build(dof_handler);
    setup_time += time.wall_time();

    //! TODO implement Chebyshev instead
    using SmootherType = JacobiSmoother<LevelMatrixType>;
    using VectorTypeMG = LinearAlgebra::distributed::Vector<float>;
    mg::SmootherRelaxation<SmootherType,VectorTypeMG> mg_smoother;

    MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
    smoother_data.resize(0, triangulation.n_global_levels() - 1);

    for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level){
        mg_matrices[level].compute_diagonal();
        smoother_data[level].relaxation = 1.0;
    }
    mg_smoother.initialize(mg_matrices, smoother_data);

    MGCoarseGridApplySmoother<LinearAlgebra::distributed::Vector<float>> mg_coarse;
    mg_coarse.initialize(mg_smoother);

    mg::Matrix<LinearAlgebra::distributed::Vector<float>> mg_matrix(mg_matrices);

    MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<LevelMatrixType>> mg_interface_matrices;
    mg_interface_matrices.resize(0, triangulation.n_global_levels() - 1);

    for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
        mg_interface_matrices[level].initialize(mg_matrices[level]);

    mg::Matrix<LinearAlgebra::distributed::Vector<float>> mg_interface(mg_interface_matrices);

    Multigrid<LinearAlgebra::distributed::Vector<float>> mg(
        mg_matrix,
        mg_coarse,
        mg_transfer,
        mg_smoother,
        mg_smoother
        );

    mg.set_edge_matrices(mg_interface, mg_interface);

    PreconditionMG<dim,LinearAlgebra::distributed::Vector<float>,MGTransferMatrixFree<dim, float>> preconditioner(dof_handler, mg, mg_transfer);

    //! TODO put as problem
    SolverControl solver_control(this->max_iters, this->epsilon * system_rhs.l2_norm());

    typename SolverGMRES<LinearAlgebra::distributed::Vector<double>>::AdditionalData gmres_data;
    //! TODO try both and see which is best (empirically) and relation to true error for both
    gmres_data.right_preconditioning = true;

    SolverGMRES<LinearAlgebra::distributed::Vector<double>> gmres(solver_control, gmres_data);

    setup_time += time.wall_time();
    time_details << "MG build smoother time     (CPU/wall) " << time.cpu_time() << "s/" << time.wall_time() << "s\n";
    pcout << "Total setup time               (wall) " << setup_time << "s\n";

    time.reset();
    time.start();

    //! TODO check correctness of the following with BCs
    constraints.distribute(solution);

    try {
        gmres.solve(system_matrix, solution, system_rhs, preconditioner);
    } catch (std::exception &e) {
        pcout << "Solver failed: " << e.what() << std::endl;
    }

    constraints.distribute(solution);

    pcout << "Time solve (" << solver_control.last_step() << " iterations)" << (solver_control.last_step() < 10 ? "  " : " ") << "(CPU/wall) " << time.cpu_time() << "s/" << time.wall_time() << "s\n";

    print_memory_usage("After Solve"); // Log memory
  }


template <int dim>
void ADRProblem<dim>::output_results(const unsigned int cycle) const {
    Timer time;

    // do not output for big meshes
    if (triangulation.n_global_active_cells() > MAX_OUTPUT_MESH_ELEMENTS) {
        pcout << "File too big" << std::endl;
        return;
    }

    DataOut<dim> data_out;

    solution.update_ghost_values();
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
    data_out.build_patches(mapping);

    DataOutBase::VtkFlags flags;
    flags.compression_level = DataOutBase::CompressionLevel::best_speed;
    data_out.set_flags(flags);
    data_out.write_vtu_with_pvtu_record("./", this->output_filename, cycle, MPI_COMM_WORLD, 1);

    pcout << "Time write output          (CPU/wall) " << time.cpu_time() << "s/" << time.wall_time() << "s\n";
}


//! TODO extract for to be on the user side
template <int dim>
void ADRProblem<dim>::run(unsigned int refinement,std::string param_filename) {
    const unsigned int n_vect_doubles = VectorizedArray<double>::size();
    const unsigned int n_vect_bits    = 8 * sizeof(double) * n_vect_doubles;

    pcout << "Vectorization over " << n_vect_doubles
            << " doubles = " << n_vect_bits << " bits ("
            << Utilities::System::get_current_vectorization_level() << ')'
            << std::endl;

    triangulation.clear(); //clear previous data to allow re-initialization
    GridGenerator::hyper_cube(triangulation, 0., 1., true);
    triangulation.refine_global(3 + refinement);

    setup_system(param_filename);
    assemble_rhs();
    solve();
    output_results(refinement);
}

#endif //PROJECT7_MATRIXFREE_ADR_PROBLEM_HPP