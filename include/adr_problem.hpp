#ifndef PROJECT7_MATRIXFREE_ADR_PROBLEM_HPP
#define PROJECT7_MATRIXFREE_ADR_PROBLEM_HPP

#include <general_definitions.hpp>
#include <default_coefficient.hpp>
#include <adr_operator.hpp>

template <int dim>
class ADRProblem {
public:
    ADRProblem(std::string param_filename);
    void run(unsigned int refinement, std::string filename);

private:
    void setup_system();
    void assemble_rhs();
    void solve();
    void output_results(std::string filename, const unsigned int cycle = 0) const;
    void print_memory_usage(const std::string &stage) const; // Added
    void declare_parameters();

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

    ParameterHandler prm;
    int max_iterations;
    double tolerance;
    bool use_jacobi,use_gmres;
    FunctionParser<dim> diffu_c;
    FunctionParser<dim> advec_c;
    FunctionParser<dim> react_c;
    FunctionParser<dim> force;
    FunctionParser<dim> dirichlet;
    FunctionParser<dim> neumann;
};


template <int dim>
ADRProblem<dim>::ADRProblem(std::string param_filename)
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
        )
    , diffu_c(1)
    , advec_c(dim)
    , react_c(1)
    , force(1)
    , dirichlet(1)
    , neumann(1)
{
    declare_parameters();
    prm.parse_input(param_filename);

    prm.enter_subsection("Algebraic Solver");
    {
        max_iterations = prm.get_integer("Max iterations");
        tolerance = prm.get_double("Tolerance");
        use_jacobi = prm.get("Preconditioner") == "Jacobi";
        use_gmres = prm.get("Solver") == "GMRES";
        prm.leave_subsection();
    }
    prm.enter_subsection("Data");
        std::map<std::string, double> constants;
        constants["pi"] = numbers::PI;
        std::string variables = diffu_c.default_variable_names();

        std::string diffu_str = prm.get("Diffusion Coefficient");
        diffu_c.initialize(
            variables,
            diffu_str,
            constants
        );

        std::vector<std::string> advec_strs(dim);
        if constexpr (dim >= 1)
            advec_strs[0] = prm.get("Advection Coefficient x value");
        if constexpr (dim >= 2)
            advec_strs[1] = prm.get("Advection Coefficient y value");
        if constexpr (dim >= 3)
            advec_strs[2] = prm.get("Advection Coefficient z value");

        advec_c.initialize(
            variables,
            advec_strs,
            constants
        );

        std::string react_str = prm.get("Reaction Coefficient");
        react_c.initialize(
            variables,
            react_str,
            constants
        );

        std::string force_str = prm.get("Force term");
        force.initialize(
            variables,
            force_str,
            constants
        );

        std::string dirichlet_str = prm.get("Dirichlet Boundary function");
        dirichlet.initialize(
            variables,
            dirichlet_str,
            constants
        );

        std::string neumann_str = prm.get("Neumann Boundary function");
        neumann.initialize(
            variables,
            neumann_str,
            constants
        );
    prm.leave_subsection();

    LOG_TITLE("Parameters were read successfully");
    LOG_VAR("Max number of iters",max_iterations);
    LOG_VAR("Tolerance",tolerance);
    LOG_VAR("Preconditioner", (use_jacobi ? "Jacobi" : "Chebyshev"));
    LOG_VAR("Solver", (use_gmres ? "GMRES" : "CG"));
    LOG_VAR("mu(x,y,z)",diffu_str);
    if constexpr (dim >= 1)
        LOG_VAR("beta_x(x,y,z)",advec_strs[0]);
    if constexpr (dim >= 2)
        LOG_VAR("beta_y(x,y,z)",advec_strs[1]);
    if constexpr (dim >= 3)
        LOG_VAR("beta_z(x,y,z)",advec_strs[2]);
    LOG_VAR("gamma(x,y,z)",react_str);
    LOG_VAR("f(x,y,z)",force_str);
    LOG_VAR("g(x,y,z)",dirichlet_str);
    LOG_VAR("h(x,y,z)",neumann_str);
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

template<int dim>
void ADRProblem<dim>::declare_parameters() {
    prm.enter_subsection("Algebraic Solver");
    {
        prm.declare_entry(
            "Max iterations",
            "500",
            Patterns::Integer(),
            "The maximum number of iterations to be performed by the linear solver"
        );
        prm.declare_entry(
            "Tolerance",
            "1.0e-12",
            Patterns::Double(),
            "The tolerance for the linear solver (it will be multiplied by the L^2 norm of the rhs)"
        );
        prm.declare_entry(
            "Solver",
            "GMRES",
            Patterns::Selection("GMRES|CG"),
            "The linear solver to be used"
        );
        prm.declare_entry(
            "Preconditioner",
            "Jacobi",
            Patterns::Selection("Jacobi|Chebyshev"),
            "The preconditioner to be used with the linear solver"
        );
    }
    prm.leave_subsection();

    prm.enter_subsection("Data");
    {
        prm.declare_entry(
            "Diffusion Coefficient",
            "1",
            Patterns::Anything(),
            "The diffusion coefficient of the PDE",
            true
        );
        prm.declare_entry(
            "Reaction Coefficient",
            "0",
            Patterns::Anything(),
            "The reaction coefficient of the PDE",
            false
        );
        prm.declare_entry(
            "Advection Coefficient x value",
            "0",
            Patterns::Anything(),
            "The x value of the advection coefficient of the PDE",
            false
        );
        prm.declare_entry(
            "Advection Coefficient y value",
            "0",
            Patterns::Anything(),
            "The y value of the advection coefficient of the PDE",
            false
        );
        prm.declare_entry(
            "Advection Coefficient z value",
            "0",
            Patterns::Anything(),
            "The z value of the advection coefficient of the PDE",
            false
        );
        prm.declare_entry(
            "Force term",
            "0",
            Patterns::Anything(),
            "The force term (aka the rhs of the equation: Lu=f in Omega )",
            false
        );
        prm.declare_entry(
            "Dirichlet Boundary function",
            "0",
            Patterns::Anything(),
            "The function to be applied at the Dirichlet boundary",
            false
        );
        prm.declare_entry(
            "Neumann Boundary function",
            "0",
            Patterns::Anything(),
            "The function to be applied at the Neumann boundary",
            false
        );
    }
    prm.leave_subsection();
    // #if defined(BUILD_TYPE_DEBUG)
    // prm.print_parameters("default_parameters.xml",ParameterHandler::XML);
    // #endif
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
        dirichlet,
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

    system_matrix.evaluate_coefficients(diffu_c, advec_c, react_c);

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

        mg_matrices[level].evaluate_coefficients(diffu_c,advec_c,react_c);
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
    for (unsigned int cell = 0; cell < system_matrix.get_matrix_free()->n_cell_batches(); ++cell) {
        phi.reinit(cell);
        //phi.evaluate(EvaluationFlags::values); We don't need this because we are assembly the RHS and we don't need to
        //"evaluate" anything from a solution vector.
        for (unsigned int q = 0; q < phi.n_q_points; ++q) {
            const Point<dim,VectorizedArray<double> > p_vect =phi.quadrature_point(q);
            VectorizedArray<double> f_value = 0.0;
            for (unsigned int v=0; v<VectorizedArray<double>::size(); ++v) {
                Point<dim> p;
                for (unsigned int d=0; d<dim; ++d) p[d] = p_vect[d][v];
                f_value[v] = force.value(p);
            }
            //phi.submit_value((f.value(phi.quadrature_point(q))  + neumann.value(phi.quadrature_point(q)) ) * phi.get_value(q), q);
            // We submit the value of the force term at the quadrature point.
            phi.submit_value(f_value, q);
        }

        // This performs the actual integration with the values (multiplication by test functions and weights)
        phi.integrate(EvaluationFlags::values);

        phi.distribute_local_to_global(system_rhs);
    }

    FEFaceEvaluation<dim, degree_finite_element> phi_face(*system_matrix.get_matrix_free());

    const types::boundary_id neumann_id = 1;

    for (unsigned int b_face = 0; b_face < system_matrix.get_matrix_free()->n_boundary_face_batches(); ++b_face) {
        if (system_matrix.get_matrix_free()->get_boundary_id(b_face) == neumann_id) {
            phi_face.reinit(b_face);
            for (unsigned int q = 0; q < phi_face.n_q_points; ++q) {
                const Point<dim,VectorizedArray<double> > p_vect = phi_face.quadrature_point(q);
                VectorizedArray<double> neumann_value = 0.0;
                for (unsigned int v=0; v<VectorizedArray<double>::size(); ++v) {
                    Point<dim> p;
                    for (unsigned int d=0; d<dim; ++d) p[d] = p_vect[d][v];
                    neumann_value[v] = neumann.value(p);
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
void ADRProblem<dim>::output_results(std::string filename, const unsigned int cycle) const {
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
    data_out.write_vtu_with_pvtu_record("./", filename, cycle, MPI_COMM_WORLD, 1);

    pcout << "Time write output          (CPU/wall) " << time.cpu_time() << "s/" << time.wall_time() << "s\n";
}


//! TODO extract for to be on the user side
template <int dim>
void ADRProblem<dim>::run(unsigned int refinement, std::string filename) {
    const unsigned int n_vect_doubles = VectorizedArray<double>::size();
    const unsigned int n_vect_bits    = 8 * sizeof(double) * n_vect_doubles;

    pcout << "Vectorization over " << n_vect_doubles
            << " doubles = " << n_vect_bits << " bits ("
            << Utilities::System::get_current_vectorization_level() << ')'
            << std::endl;

    triangulation.clear(); //clear previous data to allow re-initialization
    GridGenerator::hyper_cube(triangulation, 0., 1.);
    triangulation.refine_global(2 + refinement);

    for (const auto &cell : triangulation.active_cell_iterators()){
        for (const auto &face : cell->face_iterators()){
            if (face->at_boundary()) {
                const Point<dim> face_center = face->center();
                if (std::abs(face_center[0] - 0.0) < 1e-10 || std::abs(face_center[1] - 0.0) < 1e-10) {
                    face->set_boundary_id(0);
                }

                else if (std::abs(face_center[0] - 1.0) < 1e-10 || std::abs(face_center[1] - 1.0) < 1e-10) {
                    face->set_boundary_id(1);
                }
            }
        }
    }

    setup_system();
    assemble_rhs();
    solve();
    output_results(filename, refinement);
}

#endif //PROJECT7_MATRIXFREE_ADR_PROBLEM_HPP