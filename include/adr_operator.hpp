#ifndef PROJECT7_MATRIXFREE_ADR_OPERATOR_HPP
#define PROJECT7_MATRIXFREE_ADR_OPERATOR_HPP

#include <general_definitions.hpp>
#include <default_coefficient.hpp>
using namespace dealii;

template <int dim, int fe_degree, typename number>
class ADROperator : public MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>> {
public:
    using value_type = number;
    using vector_type = LinearAlgebra::distributed::Vector<number>;

    ADROperator();

    void clear() override;

    void evaluate_coefficients(
        const DiffusionCoefficient<dim> &diffu_f,
        const AdvectionCoefficient<dim> &advec_f,
        const ReactionCoefficient<dim> &react_f
        );

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
    Table<2, VectorizedArray<number>>                  gamma_values;
};

template <typename MatrixType>
class JacobiSmoother : public DiagonalMatrix<typename MatrixType::vector_type> {
public:
    using VectorType = typename MatrixType::vector_type;
    using value_type = typename VectorType::value_type;

    struct AdditionalData {
        double relaxation = 1.0;
    };

    void initialize(const MatrixType &matrix, const AdditionalData &data);

    void step(VectorType &dst, const VectorType &src) const;

    void Tstep(VectorType &dst, const VectorType &src) const;

private:
    const MatrixType *matrix;
    double relaxation;
};

#endif //PROJECT7_MATRIXFREE_ADR_OPERATOR_HPP