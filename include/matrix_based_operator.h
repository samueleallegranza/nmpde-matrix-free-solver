// include/matrix_based_operator.h

#ifndef MATRIX_BASED_OPERATOR_H
#define MATRIX_BASED_OPERATOR_H

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_values.h>

using namespace dealii;

/**
 * Traditional matrix-based operator for comparison
 * Assembles and stores the full system matrix
 */
template <int dim>
class MatrixBasedAdvectionDiffusion
{
public:
  /**
   * Constructor
   */
  MatrixBasedAdvectionDiffusion();

  /**
   * Clear all data
   */
  void clear();

  /**
   * Initialize sparsity pattern
   */
  void initialize_sparsity_pattern(
    const DoFHandler<dim> &           dof_handler,
    const AffineConstraints<double> & constraints);

  /**
   * Assemble the system matrix
   */
  void assemble_matrix(
    const DoFHandler<dim> &                        dof_handler,
    const Quadrature<dim> &                        quadrature,
    const AffineConstraints<double> &              constraints,
    const std::shared_ptr<const Function<dim>>     mu_function,
    const std::shared_ptr<const TensorFunction<1, dim>> beta_function,
    const std::shared_ptr<const Function<dim>>     gamma_function);

  /**
   * Get reference to system matrix
   */
  const SparseMatrix<double> &get_system_matrix() const;

  /**
   * Get reference to system matrix (non-const)
   */
  SparseMatrix<double> &get_system_matrix();

private:
  /**
   * Sparsity pattern
   */
  SparsityPattern sparsity_pattern;

  /**
   * System matrix
   */
  SparseMatrix<double> system_matrix;
};

// ====================== Implementation ====================== 

template <int dim>
MatrixBasedAdvectionDiffusion<dim>::MatrixBasedAdvectionDiffusion()
{}

template <int dim>
void MatrixBasedAdvectionDiffusion<dim>::clear()
{
  // TODO: Clear sparsity pattern and system matrix
  system_matrix.clear();
  sparsity_pattern.reinit(0, 0, 0);
}

template <int dim>
void MatrixBasedAdvectionDiffusion<dim>::initialize_sparsity_pattern(
  const DoFHandler<dim> &           dof_handler,
  const AffineConstraints<double> & constraints)
{
  // TODO: Create dynamic sparsity pattern
  // DynamicSparsityPattern dsp(dof_handler.n_dofs());
  // DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
  
  // TODO: Copy to compressed sparsity pattern
  // sparsity_pattern.copy_from(dsp);
  
  // TODO: Initialize system matrix
  // system_matrix.reinit(sparsity_pattern);
}

template <int dim>
void MatrixBasedAdvectionDiffusion<dim>::assemble_matrix(
  const DoFHandler<dim> &                        dof_handler,
  const Quadrature<dim> &                        quadrature,
  const AffineConstraints<double> &              constraints,
  const std::shared_ptr<const Function<dim>>     mu_function,
  const std::shared_ptr<const TensorFunction<1, dim>> beta_function,
  const std::shared_ptr<const Function<dim>>     gamma_function)
{
  // TODO: Standard matrix assembly using FEValues
  // FEValues<dim> fe_values(...);
  // FullMatrix<double> cell_matrix;
  // std::vector<types::global_dof_index> local_dof_indices;
  
  // for (const auto &cell : dof_handler.active_cell_iterators())
  // {
  //   cell_matrix = 0;
  //   fe_values.reinit(cell);
  //   
  //   for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
  //   {
  //     // TODO: Evaluate coefficients at quadrature point
  //     // const auto mu = mu_function->value(fe_values.quadrature_point(q));
  //     // const auto beta = beta_function->value(fe_values.quadrature_point(q));
  //     // const auto gamma = gamma_function->value(fe_values.quadrature_point(q));
  //     
  //     for (unsigned int i = 0; i < fe_values.dofs_per_cell; ++i)
  //     {
  //       for (unsigned int j = 0; j < fe_values.dofs_per_cell; ++j)
  //       {
  //         // TODO: Compute cell_matrix(i,j) = ∫ [ μ∇φi·∇φj + β·∇φj φi + γφiφj ] dx
  //       }
  //     }
  //   }
  //   
  //   cell->get_dof_indices(local_dof_indices);
  //   constraints.distribute_local_to_global(cell_matrix,
  //                                          local_dof_indices,
  //                                          system_matrix);
  // }
}

template <int dim>
const SparseMatrix<double> &
MatrixBasedAdvectionDiffusion<dim>::get_system_matrix() const
{
  return system_matrix;
}

template <int dim>
SparseMatrix<double> &
MatrixBasedAdvectionDiffusion<dim>::get_system_matrix()
{
  return system_matrix;
}

#endif // MATRIX_BASED_OPERATOR_H