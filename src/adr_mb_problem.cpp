#include <general_definitions.hpp>
#include <prm_handler.hpp>
#include <adr_mb_problem.hpp>

namespace MatrixBasedADR {
    using namespace dealii;

    template <int dim>
    ADRMBProblem<dim>::ADRMBProblem() :
        ADR::ADRParamHandler<dim>(),
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
        , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {}


    template <int dim>
        void ADRMBProblem<dim>::print_memory_usage() const {
        LOG_IMPORTANT("Memory Consumption")

        // Grid and DoFs (Base mesh connectivity)
        std::size_t mem_tria = triangulation.memory_consumption();
        std::size_t mem_dofs = dof_handler.memory_consumption();

        // System Matrix (Matrix-Free Level 0)
        // This includes precomputed Shape Gradients, JxW, Normal Vectors, etc.
        std::size_t mem_mb_sys = system_matrix.memory_consumption();

        // Multigrid Levels (Matrix-Free Levels > 0)
        std::size_t mem_mb_mg = 0;
        for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level) {
            mem_mb_mg += mg_matrices[level].memory_consumption();
        }

        // Vectors (Solution + RHS)
        std::size_t mem_vec = solution.memory_consumption() + system_rhs.memory_consumption();

        // Convert to MB for readability
        const double to_MB = 1.0 / (1024.0 * 1024.0);

        LOG_FIT(fmt::format("{}Triangulation:        {}{:2.9f} MB",GREEN,YELLOW,(mem_tria * to_MB)),90);
        LOG_FIT(fmt::format("{}DoFHandler:           {}{:2.9f} MB",GREEN,YELLOW,(mem_dofs * to_MB)),90);
        LOG_FIT(fmt::format("{}Matrix:               {}{:2.9f} MB",GREEN,YELLOW,(mem_mb_sys * to_MB)),90);
        LOG_FIT(fmt::format("{}Multigrid matrices:   {}{:2.9f} MB",GREEN,YELLOW,(mem_mb_mg * to_MB)),90);
        LOG_FIT(fmt::format("{}Vectors (Sol+RHS):    {}{:2.9f} MB",GREEN,YELLOW,(mem_vec * to_MB)),90);
        std::size_t  total = mem_tria + mem_dofs + mem_mb_sys + mem_mb_mg + mem_vec;
        LOG_FIT(fmt::format("{}{}TOTAL:                {}{:2.9f} MB",BOLD,RED,YELLOW,(total * to_MB)),94);
    }

    template<int dim>
    void ADRMBProblem<dim>::print_memory_usage_to_file(const unsigned int refinement) const {
        if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) != 0) return;

        // Grid and DoFs (Base mesh connectivity)
        std::size_t mem_tria = triangulation.memory_consumption();
        std::size_t mem_dofs = dof_handler.memory_consumption();

        // System Matrix (Matrix-Free Level 0)
        // This includes precomputed Shape Gradients, JxW, Normal Vectors, etc.
        std::size_t mem_mb_sys = system_matrix.memory_consumption();

        // Multigrid Levels (Matrix-Free Levels > 0)
        std::size_t mem_mb_mg = 0;
        for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level) {
            mem_mb_mg += mg_matrices[level].memory_consumption();
        }

        // Vectors (Solution + RHS)
        std::size_t mem_vec = solution.memory_consumption() + system_rhs.memory_consumption();

        // Convert to MB for readability
        const double to_MB = 1.0 / (1024.0 * 1024.0);

        std::ofstream outfile;
        if (!std::filesystem::exists(this->output_filename+".csv")) {
            outfile.open(this->output_filename+".csv",std::ios_base::out);
            outfile << "# Matrix Based Memory Consumption" << "\n";
            outfile << "Refinement,Dofs,Triangulation,DoFHandler,Matrix,Multigrid matrices,Vectors,TOTAL" << "\n";

        } else {
            outfile.open(this->output_filename+".csv",std::ios_base::app);
        }


        std::size_t  total = mem_tria + mem_dofs + mem_mb_sys + mem_mb_mg + mem_vec;

        outfile << refinement << "," << dof_handler.n_dofs() << "," << std::setprecision(10)
                << (mem_tria * to_MB) << ","
                << (mem_dofs * to_MB) << ","
                << (mem_mb_sys * to_MB) << ","
                << (mem_mb_mg * to_MB) << ","
                << (mem_vec * to_MB) << ","
                << (total * to_MB)
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
            Tensor<1,dim> advection = to_tensor(this->advection_c,x_q);
            auto reaction = this->reaction_c.value(x_q);
            auto force = this->force_term.value(x_q);

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
                copy_data.cell_rhs(i) += force * fe_values.shape_value(i, q) * fe_values.JxW(q);
            }
        }

        if (cell->at_boundary()) {
            for (unsigned int face_id = 0 ; face_id < cell->n_faces(); ++face_id) {

                auto face = cell->face(face_id);
                fe_f_values.reinit(cell,face_id);

                auto pos = std::find(
                    this->neumann_bc_tags.begin(),
                    this->neumann_bc_tags.end(),
                    face->boundary_id()
                );

                if (face->at_boundary() && pos != this->neumann_bc_tags.end()) {
                    fe_f_values.reinit(cell,face_id);
                    auto neumann_index = pos - this->neumann_bc_tags.begin();
                    for (unsigned int q=0; q<fe_f_values.get_quadrature().size(); ++q) {
                        auto x_q = fe_f_values.quadrature_point(q);
                        for (unsigned int i = 0; i<dofs_per_cell; ++i) {
                            copy_data.cell_rhs(i) +=
                                this->neumann_bc[neumann_index]->value(x_q) *
                                fe_f_values.shape_value(i,q) *
                                fe_f_values.JxW(q);
                        }
                    }
                }
            }
        }
    }


    template <int dim>
    void ADRMBProblem<dim>::setup_system(std::string param_filename) {
        this->init(param_filename);

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

        DynamicSparsityPattern dsp_level_0(dof_handler.n_dofs(), dof_handler.n_dofs());
        DoFTools::make_sparsity_pattern(dof_handler, dsp_level_0, constraints);
        sparsity_pattern.copy_from(dsp_level_0);

        system_matrix.reinit(sparsity_pattern);

        mg_constrained_dofs.clear();
        mg_constrained_dofs.initialize(dof_handler);
        mg_constrained_dofs.make_zero_boundary_constraints(dof_handler, dirichlet_boundary_ids);

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

        SolverControl solver_control(this->max_iters, this->epsilon * system_rhs.l2_norm());
        solution = 0;

        try {
            if (this->symmetric_solver) {
                // CG solver for dealii::Vector
                SolverCG<Vector<double>> cg(solver_control);
                cg.solve(system_matrix, solution, system_rhs, preconditioner);
            } else {
                // GMRES solver for dealii::Vector
                typename SolverGMRES<Vector<double>>::AdditionalData gmres_data;
                //! TODO try both and see which is best (empirically) and relation to true error for both
                gmres_data.right_preconditioning = true;

                SolverGMRES<Vector<double>> gmres(solver_control, gmres_data);
                gmres.solve(system_matrix, solution, system_rhs, preconditioner);
            }
        } catch (std::exception &e) {
            pcout << "Solver failed: " << e.what() << std::endl;
        }

        constraints.distribute(solution);

    }


    template <int dim>
    void ADRMBProblem<dim>::output_results(const unsigned int cycle) const {

        // do not output for big meshes
        // if (triangulation.n_global_active_cells() > MAX_OUTPUT_MESH_ELEMENTS) {
        //     pcout << "File too big" << std::endl;
        //     return;
        // }

        DataOut<dim> data_out;

        solution.update_ghost_values();
        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(solution, "solution");
        data_out.build_patches(mapping);

        DataOutBase::VtkFlags flags;
        flags.compression_level = DataOutBase::CompressionLevel::best_speed;
        data_out.set_flags(flags);
        data_out.write_vtu_with_pvtu_record("./", this->output_filename, cycle, MPI_COMM_WORLD, 1);

    }


    template <int dim>
    void ADRMBProblem<dim>::run(std::string param_filename) {
        // const unsigned int n_vect_doubles = VectorizedArray<double>::size();
        // const unsigned int n_vect_bits    = 8 * sizeof(double) * n_vect_doubles;
        //
        // pcout << "Vectorization over " << n_vect_doubles
        //         << " doubles = " << n_vect_bits << " bits ("
        //         << Utilities::System::get_current_vectorization_level() << ')'
        //         << std::endl;

        this->init(param_filename);

        this->output_filename += "_mb";

        std::ofstream outfile;
        if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) outfile.open(this->output_filename+"_time.csv",std::ios_base::out);
        if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) outfile << "# Matrix Free Time Details" << "\n";
        if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) outfile << "Refinement,Setup,Assembly,Solve,TOTAL" << "\n";

        for (auto ref : this->refinements) {
            LOG_VAR("Refinement", ref);
            if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) outfile << ref << ",";
            triangulation.clear(); //clear previous data to allow re-initialization
            GridGenerator::hyper_cube(triangulation, 0., 1., true);
            triangulation.refine_global(ref);


            Timer timer;
            double total_time = 0.0;

            LOG("Setup")
            timer.start();
            setup_system(param_filename);
            timer.stop();
            total_time += timer.wall_time();
            if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) outfile << timer.wall_time() << ",";

            LOG("Assemble")
            timer.start();
            assemble_system();
            assemble_multigrid();
            timer.stop();
            total_time += timer.wall_time();
            if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) outfile << timer.wall_time() << ",";

            LOG("Solve")
            timer.start();
            solve();
            timer.stop();
            total_time += timer.wall_time();
            if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) outfile << timer.wall_time() << ",";

            if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) outfile << total_time << std::endl;

            print_memory_usage();
            print_memory_usage_to_file(ref);
            output_results(ref);
        }
    }
}