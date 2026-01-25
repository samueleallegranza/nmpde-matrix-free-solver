// include/matrix_free_operator.h

#ifndef MATRIX_FREE_OPERATOR_H
#define MATRIX_FREE_OPERATOR_H

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

using namespace dealii;

/**
 * Matrix-free operator for the advection-diffusion-reaction equation:
 * -∇·(μ∇u) + ∇·(βu) + γu = f
 */
template <int dim, int fe_degree, typename number>
class MatrixFreeAdvectionDiffusion
{
public:
  /**
   * Constructor
   */
  MatrixFreeAdvectionDiffusion();

  /**
   * Clear all data
   */
  void clear();

  /**
   * Initialize the operator with matrix-free data and coefficients
   */
  void initialize(
    std::shared_ptr<const MatrixFree<dim, number>> matrix_free_data,
    const std::shared_ptr<const Function<dim>>     mu_function,
    const std::shared_ptr<const TensorFunction<1, dim>> beta_function,
    const std::shared_ptr<const Function<dim>>     gamma_function);

  /**
   * Matrix-vector product: dst = A * src
   */
  void vmult(LinearAlgebra::distributed::Vector<number> &      dst,
             const LinearAlgebra::distributed::Vector<number> &src) const;

  /**
   * Transposed matrix-vector product (if needed)
   */
  void Tvmult(LinearAlgebra::distributed::Vector<number> &      dst,
              const LinearAlgebra::distributed::Vector<number> &src) const;

  /**
   * Compute diagonal for preconditioning
   */
  void compute_diagonal();

  /**
   * Get inverse diagonal for Jacobi preconditioning
   */
  const LinearAlgebra::distributed::Vector<number> &
  get_matrix_diagonal_inverse() const;

private:
  /**
   * Local cell operation for matrix-free evaluation
   */
  void local_apply(
    const MatrixFree<dim, number> &                   data,
    LinearAlgebra::distributed::Vector<number> &      dst,
    const LinearAlgebra::distributed::Vector<number> &src,
    const std::pair<unsigned int, unsigned int> &     cell_range) const;

  /**
   * Local cell operation for diagonal computation
   */
  void local_compute_diagonal(
    const MatrixFree<dim, number> &                   data,
    LinearAlgebra::distributed::Vector<number> &      dst,
    const unsigned int &                              dummy,
    const std::pair<unsigned int, unsigned int> &     cell_range) const;

  /**
   * Matrix-free data structure
   */
  std::shared_ptr<const MatrixFree<dim, number>> matrix_free_data;

  /**
   * Coefficient functions evaluated at quadrature points
   */
  Table<2, VectorizedArray<number>> mu_coefficients;
  Table<2, Tensor<1, dim, VectorizedArray<number>>> beta_coefficients;
  Table<2, VectorizedArray<number>> gamma_coefficients;

  /**
   * Inverse diagonal for preconditioning
   */
  LinearAlgebra::distributed::Vector<number> inverse_diagonal;
};

// ====================== Implementation ====================== 

template <int dim, int fe_degree, typename number>
MatrixFreeAdvectionDiffusion<dim, fe_degree, number>::
MatrixFreeAdvectionDiffusion()
{}

template <int dim, int fe_degree, typename number>
void MatrixFreeAdvectionDiffusion<dim, fe_degree, number>::clear()
{
  // TODO: Clear all data structures
}

template <int dim, int fe_degree, typename number>
void MatrixFreeAdvectionDiffusion<dim, fe_degree, number>::initialize(
  std::shared_ptr<const MatrixFree<dim, number>> mf_data,
  const std::shared_ptr<const Function<dim>>     mu_function,
  const std::shared_ptr<const TensorFunction<1, dim>> beta_function,
  const std::shared_ptr<const Function<dim>>     gamma_function)
{
  // TODO: Store matrix-free data
  // TODO: Evaluate coefficient functions at all quadrature points
  // TODO: Store coefficients in tables for vectorized access
}

template <int dim, int fe_degree, typename number>
void MatrixFreeAdvectionDiffusion<dim, fe_degree, number>::vmult(
  LinearAlgebra::distributed::Vector<number> &      dst,
  const LinearAlgebra::distributed::Vector<number> &src) const
{
  // TODO: Apply matrix-free operator using cell_loop
  // dst.zero_out_ghost_values();
  // matrix_free_data->cell_loop(&MatrixFreeAdvectionDiffusion::local_apply,
  //                             this, dst, src);
}

template <int dim, int fe_degree, typename number>
void MatrixFreeAdvectionDiffusion<dim, fe_degree, number>::Tvmult(
  LinearAlgebra::distributed::Vector<number> &      dst,
  const LinearAlgebra::distributed::Vector<number> &src) const
{
  // TODO: For symmetric problems, this is same as vmult
  vmult(dst, src);
}

template <int dim, int fe_degree, typename number>
void MatrixFreeAdvectionDiffusion<dim, fe_degree, number>::
local_apply(
  const MatrixFree<dim, number> &                   data,
  LinearAlgebra::distributed::Vector<number> &      dst,
  const LinearAlgebra::distributed::Vector<number> &src,
  const std::pair<unsigned int, unsigned int> &     cell_range) const
{
  // TODO: Implement local cell operation
  // FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> phi(data);
  
  // for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  // {
  //   phi.reinit(cell);
  //   phi.read_dof_values(src);
  //   phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
  //
  //   for (unsigned int q = 0; q < phi.n_q_points; ++q)
  //   {
  //     // TODO: Compute -∇·(μ∇u) + ∇·(βu) + γu
  //     // phi.submit_gradient(..., q);
  //     // phi.submit_value(..., q);
  //   }
  //
  //   phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
  //   phi.distribute_local_to_global(dst);
  // }
}

template <int dim, int fe_degree, typename number>
void MatrixFreeAdvectionDiffusion<dim, fe_degree, number>::compute_diagonal()
{
  // TODO: Compute diagonal entries for Jacobi preconditioning
  // matrix_free_data->initialize_dof_vector(inverse_diagonal);
  // MatrixFreeTools::compute_diagonal(...);
  // Invert diagonal entries
}

template <int dim, int fe_degree, typename number>
void MatrixFreeAdvectionDiffusion<dim, fe_degree, number>::
local_compute_diagonal(
  const MatrixFree<dim, number> &                   data,
  LinearAlgebra::distributed::Vector<number> &      dst,
  const unsigned int &                              ,
  const std::pair<unsigned int, unsigned int> &     cell_range) const
{
  // TODO: Local diagonal computation
}

template <int dim, int fe_degree, typename number>
const LinearAlgebra::distributed::Vector<number> &
MatrixFreeAdvectionDiffusion<dim, fe_degree, number>::
get_matrix_diagonal_inverse() const
{
  return inverse_diagonal;
}

#endif // MATRIX_FREE_OPERATOR_H