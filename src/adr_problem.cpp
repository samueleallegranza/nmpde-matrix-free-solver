#include <adr_problem.hpp>

template <int dim>
ADRProblem<dim>::ADRProblem()
#ifdef DEAL_II_WITH_P4EST
    : triangulation(
        MPI_COMM_WORLD,
        Triangulation<dim>::limit_level_difference_at_vertices,
        parallel::distributed::Triangulation<dim, dim>::construct_multigrid_hierarchy
        )
#else
    : triangulation(Triangulation<dim>::limit_level_difference_at_vertices)
#endif
    , fe(degree_finite_element)
    , dof_handler(triangulation)
    , setup_time(0.0)
    , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    , time_details(
        std::cout,
        false &&
        Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0
        ) {

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
void ADRProblem<dim>::setup_system() {

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

    VectorTools::interpolate_boundary_values(
        mapping,
        dof_handler,
        0,
        DirichletBoundaryCondition<dim>(),
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

    system_matrix.evaluate_coefficients();

    system_matrix.initialize_dof_vector(solution);
    system_matrix.initialize_dof_vector(system_rhs);

    setup_time += time.wall_time();
    time_details << "Setup matrix-free system   (CPU/wall) " << time.cpu_time() << "s/" << time.wall_time() << 's' << std::endl;
    time.restart();

    const unsigned int nlevels = triangulation.n_global_levels();
    mg_matrices.resize(0, nlevels - 1);

    const std::set<types::boundary_id> dirichlet_boundary_ids = {0};
    mg_constrained_dofs.initialize(dof_handler);
    // initialize dof for boundaries on level 0 of multigrid
    mg_constrained_dofs.make_zero_boundary_constraints(dof_handler,dirichlet_boundary_ids);

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

        mg_matrices[level].evaluate_coefficients();
    }
    setup_time += time.wall_time();
    time_details << "Setup matrix-free levels   (CPU/wall) " << time.cpu_time() << "s/" << time.wall_time() << 's' << std::endl;

    //! TODO remove
    print_memory_usage("After Setup"); // Log memory
}


template <int dim>
void ADRProblem<dim>::assemble_rhs() {
    Timer time;

    system_rhs = 0;
    FEEvaluation<dim, degree_finite_element> phi(*system_matrix.get_matrix_free());
    ForceTerm<dim> f;
    for (unsigned int cell = 0; cell < system_matrix.get_matrix_free()->n_cell_batches(); ++cell) {
        phi.reinit(cell);
        phi.evaluate(EvaluationFlags::values);

        //! TODO verify correctness
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
            phi.submit_value(f(cell, q) * phi.get_value(), q);

        phi.integrate(EvaluationFlags::values);
        phi.distribute_local_to_global(system_rhs);
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
    const int max_iterations = 500;
    const double tolerance = 1.0e-12;
    SolverControl solver_control(max_iterations, tolerance * system_rhs.l2_norm());

    typename SolverGMRES<LinearAlgebra::distributed::Vector<double>>::AdditionalData gmres_data;
    //! TODO try both and see which is best (empirically) and relation to true error for both
    gmres_data.right_preconditioning = true;

    SolverGMRES<LinearAlgebra::distributed::Vector<double>> gmres(solver_control, gmres_data);

    setup_time += time.wall_time();
    time_details << "MG build smoother time     (CPU/wall) " << time.cpu_time() << "s/" << time.wall_time() << "s\n";
    pcout << "Total setup time               (wall) " << setup_time << "s\n";

    time.reset();
    time.start();

    //! TODO
    // old is :
    // constraints.set_zero(solution);
    // changed to:
    constraints.distribute(solution);

    // B. Calculate the contribution of these boundary values to the system: A * u_bc
    LinearAlgebra::distributed::Vector<double> boundary_contribution;
    system_matrix.initialize_dof_vector(boundary_contribution);
    system_matrix.vmult(boundary_contribution, solution);

    // C. Adjust the RHS: f_new = f_old - A * u_bc
    // We are now solving for the correction: A * delta_u = f_new
    system_rhs.add(-1.0, boundary_contribution);

    // D. Reset solution boundary DoFs to zero for the solve phase.
    // The solver will compute the correction 'delta_u' which must be 0 at boundaries.
    constraints.set_zero(solution);

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
void ADRProblem<dim>::output_results(const unsigned int cycle,std::string filename) const {
    Timer time;
    // do not output for big meshes
    if (triangulation.n_global_active_cells() > 1000000) {
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
    data_out.write_vtu_with_pvtu_record("./", filename, cycle, MPI_COMM_WORLD, 3);

    time_details << "Time write output          (CPU/wall) " << time.cpu_time() << "s/" << time.wall_time() << "s\n";
}


//! TODO extract for to be on the user side
template <int dim>
void ADRProblem<dim>::run() {
    const unsigned int n_vect_doubles = VectorizedArray<double>::size();
    const unsigned int n_vect_bits    = 8 * sizeof(double) * n_vect_doubles;

    pcout << "Vectorization over " << n_vect_doubles
            << " doubles = " << n_vect_bits << " bits ("
            << Utilities::System::get_current_vectorization_level() << ')'
            << std::endl;

    for (unsigned int cycle = 0; cycle < 5; ++cycle) {
        pcout << "Cycle " << cycle << std::endl;

        if (cycle == 0)
          {
            GridGenerator::hyper_cube(triangulation, 0., 1.);
            triangulation.refine_global(2);
          }
        triangulation.refine_global(1);
        setup_system();
        assemble_rhs();
        solve();
        output_results(cycle);
        pcout << std::endl;
      };
  }
}