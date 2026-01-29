/* ---------------------------------------------------------------------
 *
 * Matrix-free Advection-Diffusion-Reaction Solver + Memory Benchmarking
 *
 * Solves: - div(mu * grad(u)) + div(beta * u) + gamma * u = f
 *
 * ---------------------------------------------------------------------
 */

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/memory_consumption.h> // Included for memory stats

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/diagonal_matrix.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <iostream>
#include <fstream>
#include <sys/resource.h> // POSIX standard for getrusage

namespace Step37_ADR
{
  using namespace dealii;

  const unsigned int degree_finite_element = 2;
  const unsigned int dimension             = 3;

  // --------------------------------------------------------------------------
  // Coefficient Class
  // --------------------------------------------------------------------------
  template <int dim, typename number>
  class Coefficient
  {
  public:
    static number get_mu(const Point<dim> &p)
    {
       return 1. / (0.05 + 2. * p.square());
    }

    static Tensor<1, dim, number> get_beta(const Point<dim> &)
    {
      Tensor<1, dim, number> beta;
      beta[0] = 1.0;
      beta[1] = 0.5;
      if (dim > 2) beta[2] = 0.0;
      return beta;
    }

    static number get_div_beta(const Point<dim> &)
    {
      return 0.0;
    }

    static number get_gamma(const Point<dim> &)
    {
      return 1.0;
    }
  };


  // --------------------------------------------------------------------------
  // Matrix-Free Operator
  // --------------------------------------------------------------------------
  template <int dim, int fe_degree, typename number>
  class ADROperator
    : public MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>
  {
  public:
    using value_type = number;
    using vector_type = LinearAlgebra::distributed::Vector<number>;

    ADROperator();

    void clear() override;

    void evaluate_coefficients();

    virtual void compute_diagonal() override;

  private:
    virtual void apply_add(
      LinearAlgebra::distributed::Vector<number> &      dst,
      const LinearAlgebra::distributed::Vector<number> &src) const override;

    void
    local_apply(const MatrixFree<dim, number> &                   data,
                LinearAlgebra::distributed::Vector<number> &      dst,
                const LinearAlgebra::distributed::Vector<number> &src,
                const std::pair<unsigned int, unsigned int> &cell_range) const;

    void local_compute_diagonal(
      const MatrixFree<dim, number> &              data,
      LinearAlgebra::distributed::Vector<number> & dst,
      const unsigned int &                         dummy,
      const std::pair<unsigned int, unsigned int> &cell_range) const;

    Table<2, VectorizedArray<number>>                  mu_values;
    Table<2, Tensor<1, dim, VectorizedArray<number>>>  beta_values;
    Table<2, VectorizedArray<number>>                  gamma_plus_div_beta_values;
  };


  template <int dim, int fe_degree, typename number>
  ADROperator<dim, fe_degree, number>::ADROperator()
    : MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>()
  {}


  template <int dim, int fe_degree, typename number>
  void ADROperator<dim, fe_degree, number>::clear()
  {
    mu_values.reinit(0, 0);
    beta_values.reinit(0, 0);
    gamma_plus_div_beta_values.reinit(0, 0);
    MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>::clear();
  }


  template <int dim, int fe_degree, typename number>
  void ADROperator<dim, fe_degree, number>::evaluate_coefficients()
  {
    const unsigned int n_cells = this->data->n_cell_batches();
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> phi(*this->data);

    mu_values.reinit(n_cells, phi.n_q_points);
    beta_values.reinit(n_cells, phi.n_q_points);
    gamma_plus_div_beta_values.reinit(n_cells, phi.n_q_points);

    for (unsigned int cell = 0; cell < n_cells; ++cell)
      {
        phi.reinit(cell);
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          {
            const Point<dim, VectorizedArray<number>> p = phi.quadrature_point(q);
            
            for (unsigned int v = 0; v < VectorizedArray<number>::size(); ++v)
            {
               Point<dim> p_real;
               for(unsigned int d=0; d<dim; ++d) p_real[d] = p[d][v];

               mu_values(cell, q)[v] = Coefficient<dim, double>::get_mu(p_real);
               
               Tensor<1, dim, double> beta = Coefficient<dim, double>::get_beta(p_real);
               for(unsigned int d=0; d<dim; ++d) beta_values(cell, q)[d][v] = beta[d];

               double div_beta = Coefficient<dim, double>::get_div_beta(p_real);
               double gamma    = Coefficient<dim, double>::get_gamma(p_real);
               
               gamma_plus_div_beta_values(cell, q)[v] = gamma + div_beta;
            }
          }
      }
  }


  template <int dim, int fe_degree, typename number>
  void ADROperator<dim, fe_degree, number>::local_apply(
    const MatrixFree<dim, number> &                   data,
    LinearAlgebra::distributed::Vector<number> &      dst,
    const LinearAlgebra::distributed::Vector<number> &src,
    const std::pair<unsigned int, unsigned int> &     cell_range) const
  {
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> phi(data);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi.reinit(cell);
        phi.read_dof_values(src);
        
        phi.evaluate(EvaluationFlags::gradients | EvaluationFlags::values);

        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          {
            const auto u_val  = phi.get_value(q);
            const auto u_grad = phi.get_gradient(q);

            phi.submit_gradient(mu_values(cell, q) * u_grad, q);

            auto value_term = beta_values(cell, q) * u_grad + 
                              gamma_plus_div_beta_values(cell, q) * u_val;
            
            phi.submit_value(value_term, q);
          }

        phi.integrate(EvaluationFlags::gradients | EvaluationFlags::values);
        phi.distribute_local_to_global(dst);
      }
  }


  template <int dim, int fe_degree, typename number>
  void ADROperator<dim, fe_degree, number>::apply_add(
    LinearAlgebra::distributed::Vector<number> &      dst,
    const LinearAlgebra::distributed::Vector<number> &src) const
  {
    this->data->cell_loop(&ADROperator::local_apply, this, dst, src);
  }


  template <int dim, int fe_degree, typename number>
  void ADROperator<dim, fe_degree, number>::compute_diagonal()
  {
    this->inverse_diagonal_entries.reset(
      new DiagonalMatrix<LinearAlgebra::distributed::Vector<number>>());
    LinearAlgebra::distributed::Vector<number> &inverse_diagonal =
      this->inverse_diagonal_entries->get_vector();
    this->data->initialize_dof_vector(inverse_diagonal);
    unsigned int dummy = 0;
    this->data->cell_loop(&ADROperator::local_compute_diagonal,
                          this,
                          inverse_diagonal,
                          dummy);

    this->set_constrained_entries_to_one(inverse_diagonal);

    for (unsigned int i = 0; i < inverse_diagonal.locally_owned_size(); ++i)
      {
        if (std::abs(inverse_diagonal.local_element(i)) > 1e-15)
            inverse_diagonal.local_element(i) = 1. / inverse_diagonal.local_element(i);
        else
            inverse_diagonal.local_element(i) = 1.0;
      }
  }


  template <int dim, int fe_degree, typename number>
  void ADROperator<dim, fe_degree, number>::local_compute_diagonal(
    const MatrixFree<dim, number> &             data,
    LinearAlgebra::distributed::Vector<number> &dst,
    const unsigned int &,
    const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> phi(data);

    AlignedVector<VectorizedArray<number>> diagonal(phi.dofs_per_cell);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi.reinit(cell);
        for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
          {
            for (unsigned int j = 0; j < phi.dofs_per_cell; ++j)
              phi.submit_dof_value(VectorizedArray<number>(), j);
            phi.submit_dof_value(make_vectorized_array<number>(1.), i);

            phi.evaluate(EvaluationFlags::gradients | EvaluationFlags::values);
            
            for (unsigned int q = 0; q < phi.n_q_points; ++q)
              {
                auto val = phi.get_value(q);
                auto grad = phi.get_gradient(q);

                phi.submit_gradient(mu_values(cell, q) * grad, q);
                
                auto value_term = beta_values(cell, q) * grad + 
                                  gamma_plus_div_beta_values(cell, q) * val;
                phi.submit_value(value_term, q);
              }
            
            phi.integrate(EvaluationFlags::gradients | EvaluationFlags::values);
            diagonal[i] = phi.get_dof_value(i);
          }
        for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
          phi.submit_dof_value(diagonal[i], i);
        phi.distribute_local_to_global(dst);
      }
  }

  // --------------------------------------------------------------------------
  // Custom Jacobi Smoother for Non-Symmetric Multigrid
  // --------------------------------------------------------------------------
  template <typename MatrixType>
  class JacobiSmoother : public DiagonalMatrix<typename MatrixType::vector_type>
  {
  public:
      using VectorType = typename MatrixType::vector_type;
      using value_type = typename VectorType::value_type;

      struct AdditionalData
      {
          double relaxation = 1.0;
      };

      void initialize(const MatrixType &matrix, const AdditionalData &data)
      {
          this->matrix = &matrix;
          this->relaxation = data.relaxation;
          this->get_vector() = matrix.get_matrix_diagonal_inverse()->get_vector();
      }

      void step(VectorType &dst, const VectorType &src) const
      {
          VectorType tmp;
          matrix->initialize_dof_vector(tmp);
          matrix->vmult(tmp, dst);
          tmp.sadd(-1.0, 1.0, src);

          VectorType correction;
          matrix->initialize_dof_vector(correction);
          this->vmult(correction, tmp);
          
          dst.add(relaxation, correction);
      }

      void Tstep(VectorType &dst, const VectorType &src) const
      {
           step(dst, src);
      }
      
  private:
      const MatrixType *matrix;
      double relaxation;
  };


  // --------------------------------------------------------------------------
  // Problem Solver Class
  // --------------------------------------------------------------------------
  template <int dim>
  class ADRProblem
  {
  public:
    ADRProblem();
    void run();

  private:
    void setup_system();
    void assemble_rhs();
    void solve();
    void output_results(const unsigned int cycle) const;
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
  ADRProblem<dim>::ADRProblem()
#ifdef DEAL_II_WITH_P4EST
    : triangulation(MPI_COMM_WORLD,
                    Triangulation<dim>::limit_level_difference_at_vertices,
                    parallel::distributed::Triangulation<
                      dim, dim>::construct_multigrid_hierarchy)
#else
    : triangulation(Triangulation<dim>::limit_level_difference_at_vertices)
#endif
    , fe(degree_finite_element)
    , dof_handler(triangulation)
    , setup_time(0.)
    , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    , time_details(std::cout,
                   false &&
                     Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {}

  template <int dim>
  void ADRProblem<dim>::print_memory_usage(const std::string &stage) const
  {
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
  void ADRProblem<dim>::setup_system()
  {
    Timer time;
    setup_time = 0;

    system_matrix.clear();
    mg_matrices.clear_elements();

    dof_handler.distribute_dofs(fe);
    dof_handler.distribute_mg_dofs();

    pcout << "Number of degrees of freedom: " << dof_handler.n_dofs()
          << std::endl;

    const IndexSet locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(dof_handler);

    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    
    VectorTools::interpolate_boundary_values(
      mapping, dof_handler, 0, Functions::ZeroFunction<dim>(), constraints);
    constraints.close();
    setup_time += time.wall_time();
    time_details << "Distribute DoFs & B.C.     (CPU/wall) " << time.cpu_time()
                 << "s/" << time.wall_time() << 's' << std::endl;
    time.restart();

    {
      typename MatrixFree<dim, double>::AdditionalData additional_data;
      additional_data.tasks_parallel_scheme =
        MatrixFree<dim, double>::AdditionalData::none;
      additional_data.mapping_update_flags =
        (update_gradients | update_JxW_values | update_quadrature_points | update_values);
      std::shared_ptr<MatrixFree<dim, double>> system_mf_storage(
        new MatrixFree<dim, double>());
      system_mf_storage->reinit(mapping,
                                dof_handler,
                                constraints,
                                QGauss<1>(fe.degree + 1),
                                additional_data);
      system_matrix.initialize(system_mf_storage);
    }

    system_matrix.evaluate_coefficients();

    system_matrix.initialize_dof_vector(solution);
    system_matrix.initialize_dof_vector(system_rhs);

    setup_time += time.wall_time();
    time_details << "Setup matrix-free system   (CPU/wall) " << time.cpu_time()
                 << "s/" << time.wall_time() << 's' << std::endl;
    time.restart();

    const unsigned int nlevels = triangulation.n_global_levels();
    mg_matrices.resize(0, nlevels - 1);

    const std::set<types::boundary_id> dirichlet_boundary_ids = {0};
    mg_constrained_dofs.initialize(dof_handler);
    mg_constrained_dofs.make_zero_boundary_constraints(dof_handler,
                                                       dirichlet_boundary_ids);

    for (unsigned int level = 0; level < nlevels; ++level)
      {
        const IndexSet relevant_dofs =
          DoFTools::extract_locally_relevant_level_dofs(dof_handler, level);
        AffineConstraints<double> level_constraints;
        level_constraints.reinit(relevant_dofs);
        level_constraints.add_lines(
          mg_constrained_dofs.get_boundary_indices(level));
        level_constraints.close();

        typename MatrixFree<dim, float>::AdditionalData additional_data;
        additional_data.tasks_parallel_scheme =
          MatrixFree<dim, float>::AdditionalData::none;
        additional_data.mapping_update_flags =
          (update_gradients | update_JxW_values | update_quadrature_points | update_values);
        additional_data.mg_level = level;
        std::shared_ptr<MatrixFree<dim, float>> mg_mf_storage_level(
          new MatrixFree<dim, float>());
        mg_mf_storage_level->reinit(mapping,
                                    dof_handler,
                                    level_constraints,
                                    QGauss<1>(fe.degree + 1),
                                    additional_data);

        mg_matrices[level].initialize(mg_mf_storage_level,
                                      mg_constrained_dofs,
                                      level);
        mg_matrices[level].evaluate_coefficients();
      }
    setup_time += time.wall_time();
    time_details << "Setup matrix-free levels   (CPU/wall) " << time.cpu_time()
                 << "s/" << time.wall_time() << 's' << std::endl;

    print_memory_usage("After Setup"); // Log memory
  }


  template <int dim>
  void ADRProblem<dim>::assemble_rhs()
  {
    Timer time;

    system_rhs = 0;
    FEEvaluation<dim, degree_finite_element> phi(
      *system_matrix.get_matrix_free());
    for (unsigned int cell = 0;
         cell < system_matrix.get_matrix_free()->n_cell_batches();
         ++cell)
      {
        phi.reinit(cell);
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          phi.submit_value(make_vectorized_array<double>(1.0), q);
        phi.integrate(EvaluationFlags::values);
        phi.distribute_local_to_global(system_rhs);
      }
    system_rhs.compress(VectorOperation::add);

    setup_time += time.wall_time();
    time_details << "Assemble right hand side   (CPU/wall) " << time.cpu_time()
                 << "s/" << time.wall_time() << 's' << std::endl;
                 
    print_memory_usage("After Assembly"); // Log memory
  }


  template <int dim>
  void ADRProblem<dim>::solve()
  {
    Timer                            time;
    MGTransferMatrixFree<dim, float> mg_transfer(mg_constrained_dofs);
    mg_transfer.build(dof_handler);
    setup_time += time.wall_time();
    
    using SmootherType = JacobiSmoother<LevelMatrixType>;
    
    mg::SmootherRelaxation<SmootherType,
                           LinearAlgebra::distributed::Vector<float>>
                                                         mg_smoother;
    MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
    smoother_data.resize(0, triangulation.n_global_levels() - 1);
    
    for (unsigned int level = 0; level < triangulation.n_global_levels();
         ++level)
      {
        mg_matrices[level].compute_diagonal();
        smoother_data[level].relaxation = 1.0; 
      }
    mg_smoother.initialize(mg_matrices, smoother_data);

    MGCoarseGridApplySmoother<LinearAlgebra::distributed::Vector<float>>
      mg_coarse;
    mg_coarse.initialize(mg_smoother);

    mg::Matrix<LinearAlgebra::distributed::Vector<float>> mg_matrix(
      mg_matrices);

    MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<LevelMatrixType>>
      mg_interface_matrices;
    mg_interface_matrices.resize(0, triangulation.n_global_levels() - 1);
    for (unsigned int level = 0; level < triangulation.n_global_levels();
         ++level)
      mg_interface_matrices[level].initialize(mg_matrices[level]);
    mg::Matrix<LinearAlgebra::distributed::Vector<float>> mg_interface(
      mg_interface_matrices);

    Multigrid<LinearAlgebra::distributed::Vector<float>> mg(
      mg_matrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);
    
    mg.set_edge_matrices(mg_interface, mg_interface);

    PreconditionMG<dim,
                   LinearAlgebra::distributed::Vector<float>,
                   MGTransferMatrixFree<dim, float>>
      preconditioner(dof_handler, mg, mg_transfer);

    SolverControl solver_control(100, 1e-12 * system_rhs.l2_norm());
    
    typename SolverGMRES<LinearAlgebra::distributed::Vector<double>>::AdditionalData gmres_data;
    gmres_data.right_preconditioning = true;

    SolverGMRES<LinearAlgebra::distributed::Vector<double>> gmres(solver_control, gmres_data);
    
    setup_time += time.wall_time();
    time_details << "MG build smoother time     (CPU/wall) " << time.cpu_time()
                 << "s/" << time.wall_time() << "s\n";
    pcout << "Total setup time               (wall) " << setup_time << "s\n";

    time.reset();
    time.start();
    constraints.set_zero(solution);
    
    try {
        gmres.solve(system_matrix, solution, system_rhs, preconditioner);
    } catch (std::exception &e) {
        pcout << "Solver failed: " << e.what() << std::endl;
    }

    constraints.distribute(solution);

    pcout << "Time solve (" << solver_control.last_step() << " iterations)"
          << (solver_control.last_step() < 10 ? "  " : " ") << "(CPU/wall) "
          << time.cpu_time() << "s/" << time.wall_time() << "s\n";

    print_memory_usage("After Solve"); // Log memory
  }


  template <int dim>
  void ADRProblem<dim>::output_results(const unsigned int cycle) const
  {
    Timer time;
    if (triangulation.n_global_active_cells() > 1000000)
      return;

    DataOut<dim> data_out;

    solution.update_ghost_values();
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
    data_out.build_patches(mapping);

    DataOutBase::VtkFlags flags;
    flags.compression_level = DataOutBase::CompressionLevel::best_speed;
    data_out.set_flags(flags);
    data_out.write_vtu_with_pvtu_record(
      "./", "adr_mf_solution", cycle, MPI_COMM_WORLD, 3);

    time_details << "Time write output          (CPU/wall) " << time.cpu_time()
                 << "s/" << time.wall_time() << "s\n";
  }


  template <int dim>
  void ADRProblem<dim>::run()
  {
    {
      const unsigned int n_vect_doubles = VectorizedArray<double>::size();
      const unsigned int n_vect_bits    = 8 * sizeof(double) * n_vect_doubles;

      pcout << "Vectorization over " << n_vect_doubles
            << " doubles = " << n_vect_bits << " bits ("
            << Utilities::System::get_current_vectorization_level() << ')'
            << std::endl;
    }

    for (unsigned int cycle = 0; cycle < 5; ++cycle)
      {
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


int main(int argc, char *argv[])
{
  try
    {
      using namespace Step37_ADR;

      Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

      ADRProblem<dimension> adr_problem;
      adr_problem.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}