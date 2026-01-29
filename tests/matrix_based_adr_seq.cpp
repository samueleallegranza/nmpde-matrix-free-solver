/* ---------------------------------------------------------------------
 * Matrix-Based ADR Benchmark (Serial)
 *
 * Modified to match the refinement cycles and timing output of the
 * Matrix-Free implementation for direct performance comparison.
 * --------------------------------------------------------------------- */

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h> // Added for benchmarking

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <iostream>

using namespace dealii;

// --------------------------------------------------------------------------
// 1. Coefficients
// --------------------------------------------------------------------------
template <int dim>
class Coefficient
{
public:
  static double get_mu(const Point<dim> &p)
  {
     return 1. / (0.05 + 2. * p.square());
  }

  static Tensor<1, dim> get_beta(const Point<dim> &)
  {
    Tensor<1, dim> beta;
    beta[0] = 1.0;
    beta[1] = 0.5;
    if (dim > 2) beta[2] = 0.0;
    return beta;
  }

  static double get_gamma(const Point<dim> &)
  {
    return 1.0;
  }
};

// --------------------------------------------------------------------------
// 2. Benchmarking Solver Class
// --------------------------------------------------------------------------
template <int dim>
class SimpleADRBenchmark
{
public:
  SimpleADRBenchmark();
  void run();

private:
  void setup_system();
  void assemble_system();
  void solve();
  void output_results(const unsigned int cycle) const;

  Triangulation<dim>        triangulation;
  FE_Q<dim>                 fe;
  DoFHandler<dim>           dof_handler;

  AffineConstraints<double> constraints;
  SparsityPattern           sparsity_pattern;
  SparseMatrix<double>      system_matrix;

  Vector<double>            solution;
  Vector<double>            system_rhs;
  
  // Timer for benchmarking
  Timer                     timer;
};

template <int dim>
SimpleADRBenchmark<dim>::SimpleADRBenchmark()
  : fe(2) 
  , dof_handler(triangulation)
{}

template <int dim>
void SimpleADRBenchmark<dim>::setup_system()
{
  timer.restart(); // Start Setup Timer

  dof_handler.distribute_dofs(fe);

  std::cout << "  Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());

  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           Functions::ZeroFunction<dim>(),
                                           constraints);
  constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);

  timer.stop(); // Stop Setup Timer
  std::cout << "  Setup time:                (CPU/wall) " 
            << timer.cpu_time() << "s/" << timer.wall_time() << "s" << std::endl;
}

template <int dim>
void SimpleADRBenchmark<dim>::assemble_system()
{
  timer.restart(); // Start Assembly Timer

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

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      cell_matrix = 0;
      cell_rhs    = 0;

      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          const Point<dim> p = fe_values.quadrature_point(q);
          const double dx    = fe_values.JxW(q);

          const double mu         = Coefficient<dim>::get_mu(p);
          const Tensor<1,dim> beta = Coefficient<dim>::get_beta(p);
          const double gamma      = Coefficient<dim>::get_gamma(p);

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              const double         phi_i      = fe_values.shape_value(i, q);
              const Tensor<1, dim> grad_phi_i = fe_values.shape_grad(i, q);

              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  const double         phi_j      = fe_values.shape_value(j, q);
                  const Tensor<1, dim> grad_phi_j = fe_values.shape_grad(j, q);

                  cell_matrix(i, j) +=
                    (mu * grad_phi_j * grad_phi_i +       
                     (beta * grad_phi_j) * phi_i +        
                     gamma * phi_j * phi_i                
                    ) * dx;
                }
              cell_rhs(i) += 1.0 * phi_i * dx;
            }
        }

      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(cell_matrix,
                                             cell_rhs,
                                             local_dof_indices,
                                             system_matrix,
                                             system_rhs);
    }
  
  timer.stop(); // Stop Assembly Timer
  std::cout << "  Assembly time:             (CPU/wall) " 
            << timer.cpu_time() << "s/" << timer.wall_time() << "s" << std::endl;
}

template <int dim>
void SimpleADRBenchmark<dim>::solve()
{
  timer.restart(); // Start Solve Timer

  SolverControl solver_control(1000, 1e-12 * system_rhs.l2_norm());
  SolverGMRES<> solver(solver_control);

  SparseILU<double> preconditioner;
  preconditioner.initialize(system_matrix, SparseILU<double>::AdditionalData());

  solver.solve(system_matrix, solution, system_rhs, preconditioner);

  constraints.distribute(solution);

  timer.stop(); // Stop Solve Timer
  std::cout << "  Solve time (" << solver_control.last_step() << " it):      (CPU/wall) " 
            << timer.cpu_time() << "s/" << timer.wall_time() << "s" << std::endl;
}

template <int dim>
void SimpleADRBenchmark<dim>::output_results(const unsigned int cycle) const
{
  Timer output_timer; // Separate timer for output just to be safe
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
  data_out.build_patches();

    std::string suffix = std::to_string(cycle);
    while (suffix.size() < 3)
        suffix = "0" + suffix;
    std::ofstream output("adr_bf_seq_solution_" + suffix + ".vtu");
  data_out.write_vtu(output);
}

template <int dim>
void SimpleADRBenchmark<dim>::run()
{
  std::cout << "Running Matrix-Based Benchmark (Dimension " << dim << ")" << std::endl;

  // Loop cycles 0 to 4 (5 total cycles)
  for (unsigned int cycle = 0; cycle < 5; ++cycle)
  {
      std::cout << "Cycle " << cycle << std::endl;

      // Exact Refinement Strategy Logic from Matrix-Free:
      // Cycle 0: HyperCube + Refine(2) + Refine(1) = Refine(3)
      // Cycle 1: Refine(1) -> Refine(4)
      if (cycle == 0)
      {
          GridGenerator::hyper_cube(triangulation, 0, 1);
          triangulation.refine_global(2);
      }
      triangulation.refine_global(1);

      setup_system();
      assemble_system();
      solve();
      output_results(cycle);
      std::cout << std::endl;
  }
}

// --------------------------------------------------------------------------
// Main
// --------------------------------------------------------------------------
int main()
{
  try
    {
      SimpleADRBenchmark<3> problem; // Set to 3D to match Step37_ADR
      problem.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << "Exception: " << exc.what() << std::endl;
      return 1;
    }
  return 0;
}