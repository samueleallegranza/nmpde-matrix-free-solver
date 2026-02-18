#ifndef PROJECT7_MATRIXFREE_ADR_PROBLEM_HPP
#define PROJECT7_MATRIXFREE_ADR_PROBLEM_HPP

#include <general_definitions.hpp>
#include <adr_operator.hpp>
#include <prm_handler.hpp>

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
class ADRMBProblem : public ADR::EllipticParamHandler<dim> {
public:
    ADRMBProblem();
    void run(unsigned int refinement,std::string param_filename);

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
    void print_memory_usage(const std::string &stage) const; // Added

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


template <int dim>
ADRMBProblem<dim>::ADRMBProblem() :
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
{}


//! TODO to remove
template <int dim>
void ADRMBProblem<dim>::print_memory_usage(const std::string &stage) const {
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
template<class Iterator>
void ADRMBProblem<dim>::cell_worker(const Iterator &cell, ScratchData<dim> &scratch_data, CopyData &copy_data) {
    FEValues<dim> &fe_values = scratch_data.fe_values;
    FEFaceValues<dim> &fe_f_values = scratch_data.fe_f_values;
    fe_values.reinit(cell);

    const unsigned int dofs_per_cell = fe_values.get_fe().n_dofs_per_cell();
    const unsigned int n_q_points= fe_values.get_quadrature().size();

    copy_data.reinit(cell, dofs_per_cell);

    for (unsigned int q = 0; q < n_q_points; ++q) {
        auto x_q = fe_values.quadrature_point(q);
        auto diffusion = this->diffusion_c.value(x_q);
        Tensor<1,dim> advection;
        advection[0] = this->advection_c.value(x_q,0);
        if constexpr (dim >= 2)
            advection[1] = this->advection_c.value(x_q,1);
        if constexpr (dim >= 3)
            advection[2] = this->advection_c.value(x_q,2);
        auto reaction = this->reaction_c.value(x_q);

        for (unsigned int i = 0; i < dofs_per_cell; ++i){
            for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                auto nabla_phi_i = fe_values.shape_grad(i, q);
                auto nabla_phi_j = fe_values.shape_grad(j, q);
                auto phi_i = fe_values.shape_value(i, q);
                auto phi_j = fe_values.shape_value(j, q);

                copy_data.cell_matrix(i, j) +=
                    (
                        diffusion * nabla_phi_i * nabla_phi_j +
                        advection * nabla_phi_j * phi_i +
                        reaction * phi_j * phi_i
                    ) * fe_values.JxW(q);
            }
            copy_data.cell_rhs(i) += this->force_term.value(x_q) * fe_values.shape_value(i, q) * fe_values.JxW(q);
        }
    }

    if (cell->at_boundary()) {
        for (unsigned int face_id = 0 ; face_id < cell->n_faces(); ++face_id) {
            auto face = cell->face(face_id);
            fe_f_values.reinit(cell,face_id);
            auto pos = std::find(this->neumann_bc_tags.begin(),this->neumann_bc_tags.end(),face->boundary_id());
            if (face->at_boundary() && pos != this->neumann_bc_tags.end()) {
                fe_f_values.reinit(cell,face_id);
                auto neumann_index = pos - this->neumann_bc_tags.begin();
                for (unsigned int q=0; q<fe_f_values.get_quadrature().size(); ++q) {
                    auto x_q = fe_f_values.quadrature_point(q);
                    for (unsigned int i = 0; i<dofs_per_cell; ++i) {
                        copy_data.cell_rhs(i) += this->neumann_bc[neumann_index]->value(x_q) * fe_f_values.shape_value(i,q) * fe_f_values.JxW(q);
                    }
                }
            }
        }
    }
}


template <int dim>
void ADRMBProblem<dim>::setup_system(std::string param_filename) {
    this->init(param_filename);

    Timer time;
    setup_time = 0;

    dof_handler.distribute_dofs(fe);
    dof_handler.distribute_mg_dofs();

    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    std::set<types::boundary_id> dirichlet_boundary_ids(this->dirichlet_bc_tags.begin(),this->dirichlet_bc_tags.end());

    std::map<types::boundary_id, const Function<dim> *> boundary_functions;
    for (long unsigned int i = 0; i < this->dirichlet_bc_tags.size(); i++)
        boundary_functions[this->dirichlet_bc_tags[i]] = this->dirichlet_bc[i].get();

    VectorTools::interpolate_boundary_values(
        dof_handler,
        boundary_functions,
        constraints
    );
    constraints.close();

    time_details << "Distribute DoFs & B.C.     (CPU/wall) " << time.cpu_time() << "s/" << time.wall_time() << 's' << std::endl;
    time.restart();

    DynamicSparsityPattern dsp_level_0(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp_level_0, constraints);
    sparsity_pattern.copy_from(dsp_level_0);

    system_matrix.reinit(sparsity_pattern);

    mg_constrained_dofs.clear();
    mg_constrained_dofs.initialize(dof_handler);
    mg_constrained_dofs.make_zero_boundary_constraints(dof_handler, dirichlet_boundary_ids);

    setup_time += time.wall_time();
    time_details << "Setup matrix-based system   (CPU/wall) " << time.cpu_time() << "s/" << time.wall_time() << 's' << std::endl;
    time.restart();

    const unsigned int n_levels = triangulation.n_levels();

    mg_interface_matrices.resize(0, n_levels - 1);
    mg_matrices.resize(0, n_levels - 1);
    mg_sparsity_patterns.resize(0, n_levels - 1);
    mg_interface_sparsity_patterns.resize(0, n_levels - 1);

    for (unsigned int level = 0; level < n_levels; ++level) {
        DynamicSparsityPattern dsp_level_sp(dof_handler.n_dofs(level),dof_handler.n_dofs(level));
        MGTools::make_sparsity_pattern(dof_handler, dsp_level_sp, level);

        mg_sparsity_patterns[level].copy_from(dsp_level_sp);
        mg_matrices[level].reinit(mg_sparsity_patterns[level]);

        DynamicSparsityPattern dsp_level_interface(dof_handler.n_dofs(level),dof_handler.n_dofs(level));
        MGTools::make_interface_sparsity_pattern(
            dof_handler,
            mg_constrained_dofs,
            dsp_level_interface,
            level
        );
        mg_interface_sparsity_patterns[level].copy_from(dsp_level_interface);
        mg_interface_matrices[level].reinit(mg_interface_sparsity_patterns[level]);
    }

    setup_time += time.wall_time();
    time_details << "Setup matrix-based levels   (CPU/wall) " << time.cpu_time() << "s/" << time.wall_time() << 's' << std::endl;

    //! TODO remove
    print_memory_usage("After Setup"); // Log memory
}

template<int dim>
void ADRMBProblem<dim>::assemble_system() {

    MappingQ1<dim> mapping;

    auto cell_worker = [&](
        const typename DoFHandler<dim>::active_cell_iterator &cell,
          ScratchData<dim> & scratch_data,
          CopyData & copy_data
    ) {
          this->cell_worker(cell, scratch_data, copy_data);
    };

    auto copier = [&](const CopyData &cd) {
        this->constraints.distribute_local_to_global(
            cd.cell_matrix,
            cd.cell_rhs,
            cd.local_dof_indices,
            system_matrix,
            system_rhs
        );
    };

    const unsigned int n_gauss_points = degree_finite_element + 1;

    ScratchData<dim> scratch_data(
        mapping,
        fe,
        n_gauss_points,
        update_values | update_gradients | update_JxW_values | update_quadrature_points
    );

    MeshWorker::mesh_loop(
        dof_handler.begin_active(),
        dof_handler.end(),
        cell_worker,
        copier,
        scratch_data,
        CopyData(),
        MeshWorker::assemble_own_cells
    );
}

template<int dim>
void ADRMBProblem<dim>::assemble_multigrid() {

    MappingQ1<dim> mapping;

    const unsigned int n_levels = triangulation.n_levels();

    std::vector<AffineConstraints<double>> boundary_constraints(n_levels);

    for (unsigned int level = 0; level < n_levels; ++level){
        const IndexSet dofset = DoFTools::extract_locally_relevant_level_dofs(dof_handler, level);
        boundary_constraints[level].reinit(dofset);
        boundary_constraints[level].add_lines(mg_constrained_dofs.get_refinement_edge_indices(level));
        boundary_constraints[level].add_lines(mg_constrained_dofs.get_boundary_indices(level));
        boundary_constraints[level].close();
    }

    auto cell_worker = [&](
        const typename DoFHandler<dim>::level_cell_iterator &cell,
        ScratchData<dim> &scratch_data,
        CopyData &copy_data
    ) {
        this->cell_worker(cell, scratch_data, copy_data);
    };

    auto copier = [&](const CopyData &cd) {
        boundary_constraints[cd.level].distribute_local_to_global(
            cd.cell_matrix,
            cd.local_dof_indices,
            mg_matrices[cd.level]
        );

        const unsigned int dofs_per_cell = cd.local_dof_indices.size();
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
                if (
                    mg_constrained_dofs.is_interface_matrix_entry(
                        cd.level,
                        cd.local_dof_indices[i],
                        cd.local_dof_indices[j]
                    )
                ) {
                    mg_interface_matrices[cd.level].add(
                        cd.local_dof_indices[i],
                        cd.local_dof_indices[j],
                        cd.cell_matrix(i, j)
                    );
                }
    };

    const unsigned int n_gauss_points = degree_finite_element + 1;

    ScratchData<dim> scratch_data(
        mapping,
        fe,
        n_gauss_points,
        update_values | update_gradients | update_JxW_values | update_quadrature_points
    );

    MeshWorker::mesh_loop(
        dof_handler.begin_mg(),
        dof_handler.end_mg(),
        cell_worker,
        copier,
        scratch_data,
        CopyData(),
        MeshWorker::assemble_own_cells
    );
}

template<int dim>
void ADRMBProblem<dim>::refine_grid() {

    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

    KellyErrorEstimator<dim>::estimate(
        dof_handler,
        QGauss<dim - 1>(degree_finite_element + 2),
        std::map<types::boundary_id, const Function<dim> *>(),
        solution,
        estimated_error_per_cell
    );
    GridRefinement::refine_and_coarsen_fixed_number(
        triangulation,
        estimated_error_per_cell,
        0.3,
        0.03
    );
    triangulation.execute_coarsening_and_refinement();
}

template <int dim>
void ADRMBProblem<dim>::solve() {

    Timer time;

    MGTransferPrebuilt<Vector<double>> mg_transfer(mg_constrained_dofs);
    mg_transfer.build(dof_handler);

    FullMatrix<double> coarse_matrix;
    coarse_matrix.copy_from(mg_matrices[0]);

    MGCoarseGridHouseholder<double, Vector<double>> coarse_grid_solver;
    coarse_grid_solver.initialize(coarse_matrix);

    using Smoother = PreconditionJacobi<SparseMatrix<double>>;
    mg::SmootherRelaxation<Smoother, Vector<double>> mg_smoother;

    mg_smoother.initialize(mg_matrices);

    mg_smoother.set_steps(2);
    mg_smoother.set_symmetric(false);

    mg::Matrix<Vector<double>> mg_matrix(mg_matrices);
    mg::Matrix<Vector<double>> mg_interface_up(mg_interface_matrices);
    mg::Matrix<Vector<double>> mg_interface_down(mg_interface_matrices);
    Multigrid<Vector<double>> mg(
        mg_matrix,
        coarse_grid_solver,
        mg_transfer,
        mg_smoother,
        mg_smoother
    );
    mg.set_edge_matrices(mg_interface_down, mg_interface_up);

    PreconditionMG<dim, Vector<double>, MGTransferPrebuilt<Vector<double>>> preconditioner(dof_handler, mg, mg_transfer);

    time_details << "MG build smoother time     (CPU/wall) " << time.cpu_time() << "s/" << time.wall_time() << "s\n";
    pcout << "Total setup time               (wall) " << setup_time << "s\n";

    time.reset();
    time.start();

    SolverControl solver_control(this->max_iters, this->epsilon * system_rhs.l2_norm());

    typename SolverGMRES<Vector<double>>::AdditionalData gmres_data;
    //! TODO try both and see which is best (empirically) and relation to true error for both
    gmres_data.right_preconditioning = true;

    SolverGMRES<Vector<double>> solver(solver_control,gmres_data);

    solution = 0;

    solver.solve(system_matrix, solution, system_rhs, preconditioner);
    constraints.distribute(solution);

    pcout << "Time solve (" << solver_control.last_step() << " iterations)" << (solver_control.last_step() < 10 ? "  " : " ") << "(CPU/wall) " << time.cpu_time() << "s/" << time.wall_time() << "s\n";

    print_memory_usage("After Solve"); // Log memory
  }


template <int dim>
void ADRMBProblem<dim>::output_results(const unsigned int cycle) const {
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
void ADRMBProblem<dim>::run(unsigned int refinement,std::string param_filename) {
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
    assemble_system();
    assemble_multigrid();
    solve();
    output_results(refinement);
}

#endif //PROJECT7_MATRIXFREE_ADR_PROBLEM_HPP