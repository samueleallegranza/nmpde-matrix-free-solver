// src/advection_diffusion_problem.h

#ifndef ADVECTION_DIFFUSION_PROBLEM_H
#define ADVECTION_DIFFUSION_PROBLEM_H

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/convergence_table.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include "../include/matrix_free_operator.h"
#include "../include/matrix_based_operator.h"

#include <fstream>
#include <iostream>

using namespace dealii;

/**
 * Coefficient functions for the PDE
 */
template <int dim>
class DiffusionCoefficient : public Function<dim>
{
public:
  virtual double value(const Point<dim> & p,
                      const unsigned int component = 0) const override;
};

template <int dim>
class AdvectionCoefficient : public TensorFunction<1, dim>
{
public:
  virtual Tensor<1, dim> value(const Point<dim> &p) const override;
};

template <int dim>
class ReactionCoefficient : public Function<dim>
{
public:
  virtual double value(const Point<dim> & p,
                      const unsigned int component = 0) const override;
};

template <int dim>
class RightHandSide : public Function<dim>
{
public:
  virtual double value(const Point<dim> & p,
                      const unsigned int component = 0) const override;
};

template <int dim>
class DirichletBoundaryValues : public Function<dim>
{
public:
  virtual double value(const Point<dim> & p,
                      const unsigned int component = 0) const override;
};

template <int dim>
class NeumannBoundaryValues : public Function<dim>
{
public:
  virtual double value(const Point<dim> & p,
                      const unsigned int component = 0) const override;
};

/**
 * Main problem class
 */
template <int dim, int fe_degree>
class AdvectionDiffusionProblem
{
public:
  AdvectionDiffusionProblem();

  void run();

private:
  void make_grid();
  void setup_system();
  void assemble_rhs();
  void solve_matrix_free();
  void solve_matrix_based();
  void compute_error() const;
  void output_results(const unsigned int cycle);
  void print_performance_metrics();

  // Mesh and DoFHandler
  Triangulation<dim> triangulation;
  FE_Q<dim>          fe;
  DoFHandler<dim>    dof_handler;

  // Constraints
  AffineConstraints<double> constraints;

  // Matrix-free structures
  std::shared_ptr<MatrixFree<dim, double>> matrix_free_data;
  MatrixFreeAdvectionDiffusion<dim, fe_degree, double> matrix_free_operator;

  // Matrix-based structures
  MatrixBasedAdvectionDiffusion<dim> matrix_based_operator;

  // Solution and RHS vectors
  LinearAlgebra::distributed::Vector<double> solution_matrix_free;
  LinearAlgebra::distributed::Vector<double> solution_matrix_based;
  LinearAlgebra::distributed::Vector<double> system_rhs;
  Vector<double>                             system_rhs_standard;

  // Coefficient functions
  std::shared_ptr<DiffusionCoefficient<dim>>  mu_function;
  std::shared_ptr<AdvectionCoefficient<dim>>  beta_function;
  std::shared_ptr<ReactionCoefficient<dim>>   gamma_function;
  std::shared_ptr<RightHandSide<dim>>         rhs_function;

  // Performance tracking
  TimerOutput computing_timer;
  ConvergenceTable convergence_table;

  struct PerformanceData
  {
    unsigned int n_dofs;
    double setup_time_mf;
    double setup_time_mb;
    double solve_time_mf;
    double solve_time_mb;
    unsigned int n_iterations_mf;
    unsigned int n_iterations_mb;
    double memory_mf;
    double memory_mb;
  };

  std::vector<PerformanceData> performance_data;
};

#endif // ADVECTION_DIFFUSION_PROBLEM_H