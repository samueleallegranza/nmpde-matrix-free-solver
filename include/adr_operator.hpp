/**
 * Library implementing a Matrix-Free operator and a Jacobi preconditioner for Matrix-Free systems
 * @file adr_operator.hpp
 * @sa adr_problem.hpp
 */

#ifndef PROJECT7_MATRIXFREE_ADR_OPERATOR_HPP
#define PROJECT7_MATRIXFREE_ADR_OPERATOR_HPP

#include <general_definitions.hpp>

/**
 * Namespace for Matrix free ADR solvers
 * @namespace MatrixFreeADR
 */
namespace MatrixFreeADR {
    using namespace dealii;

    /**
     * @brief a class that implements the action of the matrix on a vector
     * @tparam dim the dimension of the problem
     * @tparam fe_degree the final element degree
     * @tparam number the number type to be used (float for preconditioner, double otherwise)
     */
    template <int dim, int fe_degree, typename number>
    class ADROperator : public MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>> {
    public:
        using value_type = number;
        using vector_type = LinearAlgebra::distributed::Vector<number>;

        ADROperator();

        /// @brief clear the matrix free data
        void clear() override;

        /**
         * @brief evaluates coefficients on the cells and saves them in a Table
         *
         * @param diffu_f The 1D function that represent the diffusion coefficient
         * @param advec_f The 3D function that represent the advection coefficient
         * @param react_f The 1D function that represent the reaction coefficient
         */
        void evaluate_coefficients(
            const FunctionParser<dim> &diffu_f,
            const FunctionParser<dim> &advec_f,
            const FunctionParser<dim> &react_f
            );

        /**
         * @brief Computes the diagonal of the matrix
         * @note Used for Jacobi
         * @sa local_compute_diagonal
         */
        virtual void compute_diagonal() override;

        /**
         * @brief Computes how much memory is occupied by the Matrix-Free operator
         * @return The memory consumed by the operator
         */
        std::size_t memory_consumption() const override;

        /// @brief a table associating to each cell a SIMD ready collection of values of the diffusion coefficient
        Table<2, VectorizedArray<number>> mu_values;
        /// @brief a table associating to each cell a SIMD ready collection of tensors of the advection coefficient
        Table<2, Tensor<1, dim, VectorizedArray<number>>> beta_values;
        /// @brief a table associating to each cell a SIMD ready collection of values of the reaction coefficient
        Table<2, VectorizedArray<number>> gamma_values;

    private:
        /**
         * @brief Applies the Operator on a vector
         *
         * This method implements the action of the matrix operator on a vector.
         * essentially it calls MatrixFree::cell_loop() to loop over the vector
         * and local_apply() on each cell.
         * @param dst The destination vector
         * @param src The source vector
         * @sa local_apply
         */
        virtual void apply_add(
          LinearAlgebra::distributed::Vector<number> &      dst,
          const LinearAlgebra::distributed::Vector<number> &src) const override;

        /**
         * @brief Applies the Operator on the cells
         * @param data The data of the matrix free operator
         * @param dst The destination vector
         * @param src The source vector
         * @param cell_range The cells on which to apply the operator
         * @sa apply_add
         */
        void
        local_apply(const MatrixFree<dim, number> &                   data,
                    LinearAlgebra::distributed::Vector<number> &      dst,
                    const LinearAlgebra::distributed::Vector<number> &src,
                    const std::pair<unsigned int, unsigned int> &cell_range) const;

        /**
         * @brief Computes the diagonal on the operator to be used with Jacobi
         * @param data The data of the matrix free operator
         * @param dst The destination vector
         * @param dummy not used
         * @param cell_range The cells on which to apply the operator
         */
        void local_compute_diagonal(
          const MatrixFree<dim, number> &              data,
          LinearAlgebra::distributed::Vector<number> & dst,
          const unsigned int &                         dummy,
          const std::pair<unsigned int, unsigned int> &cell_range) const;
    };


    /**
     * @brief Implements the Jacobi Smoother for Matrix-Free Operators
     * @tparam MatrixType The type of the operator
     */
    template <typename MatrixType>
    class JacobiSmoother : public DiagonalMatrix<typename MatrixType::vector_type> {
    public:
        using VectorType = typename MatrixType::vector_type;
        using value_type = typename VectorType::value_type;

        /**
         * @brief additional data to be passed to the smoother
         */
        struct AdditionalData {
            /// @brief the relaxation coefficient
            double relaxation = 1.0;
        };

        /**
         * @brief Initializes the smoother
         * @param matrix The matrix to extract the diagonal from
         * @param data Additional data for the jacobi smooter
         * @sa AdditionalData
         */
        void initialize(const MatrixType &matrix, const AdditionalData &data);

        /**
         * @brief Perform one step of the preconditioned Jacobi iteration
         * @param dst The destination vector
         * @param src The source vector
         */
        void step(VectorType &dst, const VectorType &src) const;

        /**
         * @brief Perform one transposed step of the preconditioned Jacobi iteration
         * @param dst The destination vector
         * @param src The source vector
         */
        void Tstep(VectorType &dst, const VectorType &src) const;

    private:
        /// @brief The matrix
        const MatrixType *matrix;
        /// @brief The relaxation parameters
        double relaxation;
    };

    template <int dim, int fe_degree, typename number>
    ADROperator<dim, fe_degree, number>::ADROperator()
        : MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>() {
    }


    template <int dim, int fe_degree, typename number>
    void ADROperator<dim, fe_degree, number>::clear() {
        mu_values.reinit(0, 0);
        beta_values.reinit(0, 0);
        gamma_values.reinit(0, 0);
        MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>::clear();
    }


    template <int dim, int fe_degree, typename number>
    void ADROperator<dim, fe_degree, number>::evaluate_coefficients(
        const FunctionParser<dim> &diffu_f,
        const FunctionParser<dim> &advec_f,
        const FunctionParser<dim> &react_f
        ) {
        const unsigned int n_cells = this->data->n_cell_batches();
        FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> phi(*this->data);

        mu_values.reinit(n_cells, phi.n_q_points);
        beta_values.reinit(n_cells, phi.n_q_points);
        gamma_values.reinit(n_cells, phi.n_q_points);

        for (unsigned int cell = 0; cell < n_cells; ++cell) {
            phi.reinit(cell);
            for (unsigned int q = 0; q < phi.n_q_points; ++q) {
                const Point<dim, VectorizedArray<number>> p_vect = phi.quadrature_point(q);
                VectorizedArray<number> diffu_value;
                Tensor<1, dim, VectorizedArray<number>> advec_value;
                VectorizedArray<number> react_value;
                for (unsigned int v=0; v<VectorizedArray<number>::size(); ++v) {
                    Point<dim> p;
                    for (unsigned int d=0; d<dim; ++d) p[d] = p_vect[d][v];
                    diffu_value[v] = diffu_f.value(p);
                    advec_value[0][v] = advec_f.value(p,0);
                    if constexpr (dim >= 2) advec_value[1][v] = advec_f.value(p,1);
                    if constexpr (dim >= 3) advec_value[2][v] = advec_f.value(p,2);
                    react_value[v] = react_f.value(p);
                }
                mu_values(cell, q) = diffu_value;
                beta_values(cell, q) = advec_value;
                gamma_values(cell, q) = react_value;
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

        for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {

            AssertDimension(mu_values.size(0), data.n_cell_batches());
            AssertDimension(mu_values.size(1), phi.n_q_points);
            AssertDimension(beta_values.size(0), data.n_cell_batches());
            AssertDimension(beta_values.size(1), phi.n_q_points);
            AssertDimension(gamma_values.size(0), data.n_cell_batches());
            AssertDimension(gamma_values.size(1), phi.n_q_points);

            phi.reinit(cell);
            phi.read_dof_values(src);

            phi.evaluate(EvaluationFlags::gradients | EvaluationFlags::values);

            for (unsigned int q = 0; q < phi.n_q_points; ++q) {
                const auto u_val  = phi.get_value(q);
                const auto u_grad = phi.get_gradient(q);

                phi.submit_gradient(mu_values(cell, q) * u_grad, q);

                /**
                 *  \nabla\phi\mu\nabla\phi + \phi\beta\nabla\phi + \phi\gamma\phi
                 */
                auto value_term =
                    beta_values(cell, q) * u_grad +
                    gamma_values(cell, q) * u_val;

                phi.submit_value(value_term, q);
            }

            phi.integrate(EvaluationFlags::gradients | EvaluationFlags::values);
            phi.distribute_local_to_global(dst);
        }
    }


    template <int dim, int fe_degree, typename number>
    void ADROperator<dim, fe_degree, number>::apply_add(
        LinearAlgebra::distributed::Vector<number> &      dst,
        const LinearAlgebra::distributed::Vector<number> &src
        ) const {
        this->data->cell_loop(&ADROperator::local_apply, this, dst, src);
    }


    template <int dim, int fe_degree, typename number>
    void ADROperator<dim, fe_degree, number>::compute_diagonal() {

        this->inverse_diagonal_entries.reset(new DiagonalMatrix<LinearAlgebra::distributed::Vector<number>>());

        auto &inverse_diagonal = this->inverse_diagonal_entries->get_vector();
        this->data->initialize_dof_vector(inverse_diagonal);

        unsigned int dummy = 0;
        this->data->cell_loop(
        &ADROperator::local_compute_diagonal,
                  this,
                  inverse_diagonal,
                  dummy
            );

        this->set_constrained_entries_to_one(inverse_diagonal);

        for (unsigned int i = 0; i < inverse_diagonal.locally_owned_size(); ++i) {
            //! TODO maybe use std::abs(inverse_diagonal.local_element(i)) > 1e-15 inside assert
            if (inverse_diagonal.local_element(i) != 0.0)
                inverse_diagonal.local_element(i) = 1. / inverse_diagonal.local_element(i);
            else
                inverse_diagonal.local_element(i) = 1.;

            //   if (std::abs(inverse_diagonal.local_element(i)) > 1e-15)
            //       inverse_diagonal.local_element(i) = 1. / inverse_diagonal.local_element(i);
            //   else
            //       inverse_diagonal.local_element(i) = 1.0;
        }
    }


    template <int dim, int fe_degree, typename number>
    std::size_t ADROperator<dim, fe_degree, number>::memory_consumption() const {
        std::size_t memory = 0;

        memory += MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>::memory_consumption();
        memory += mu_values.memory_consumption();
        memory += beta_values.memory_consumption();
        memory += gamma_values.memory_consumption();

        return memory;
    }


    template <int dim, int fe_degree, typename number>
    void ADROperator<dim, fe_degree, number>::local_compute_diagonal(
        const MatrixFree<dim, number> &             data,
        LinearAlgebra::distributed::Vector<number> &dst,
        const unsigned int &,
        const std::pair<unsigned int, unsigned int> &cell_range
        ) const {

        FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> phi(data);

        AlignedVector<VectorizedArray<number>> diagonal(phi.dofs_per_cell);

        for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {

            AssertDimension(mu_values.size(0), data.n_cell_batches());
            AssertDimension(mu_values.size(1), phi.n_q_points);
            AssertDimension(beta_values.size(0), data.n_cell_batches());
            AssertDimension(beta_values.size(1), phi.n_q_points);
            AssertDimension(gamma_values.size(0), data.n_cell_batches());
            AssertDimension(gamma_values.size(1), phi.n_q_points);

            phi.reinit(cell);

            for (unsigned int i = 0; i < phi.dofs_per_cell; ++i) {

                for (unsigned int j = 0; j < phi.dofs_per_cell; ++j) {
                    phi.submit_dof_value(VectorizedArray<number>(), j);
                }

                phi.submit_dof_value(make_vectorized_array<number>(1.), i);

                phi.evaluate(EvaluationFlags::gradients | EvaluationFlags::values);

                for (unsigned int q = 0; q < phi.n_q_points; ++q) {
                    const auto u_val  = phi.get_value(q);
                    const auto u_grad = phi.get_gradient(q);

                    phi.submit_gradient(mu_values(cell, q) * u_grad, q);

                    /**
                     *  \nabla\phi\mu\nabla\phi + \phi\beta\nabla\phi + \phi\gamma\phi
                     */
                    auto value_term =
                        beta_values(cell, q) * u_grad +
                        gamma_values(cell, q) * u_val;

                    phi.submit_value(value_term, q);
                }

                phi.integrate(EvaluationFlags::gradients | EvaluationFlags::values);
                diagonal[i] = phi.get_dof_value(i);
            }

            for (unsigned int i = 0; i < phi.dofs_per_cell; ++i) {
                phi.submit_dof_value(diagonal[i], i);
            }
            phi.distribute_local_to_global(dst);
        }
    }


    template<typename MatrixType>
    void JacobiSmoother<MatrixType>::initialize(const MatrixType &matrix, const AdditionalData &data) {
        this->matrix = &matrix;
        this->relaxation = data.relaxation;
        this->get_vector() = matrix.get_matrix_diagonal_inverse()->get_vector();
    }

    template<typename MatrixType>
    void JacobiSmoother<MatrixType>::step(VectorType &dst, const VectorType &src) const {
        /**
         *  dst initially contains x_k (and later x_{k+1})
         *  src initially contains b (the rhs)
         */
        VectorType tmp;
        matrix->initialize_dof_vector(tmp);
        // tmp = A * x_k (using local apply for matrix free)
        matrix->vmult(tmp, dst);
        // -1.0*tmp +1.0*src = -(A * x_k) + b = b-Ax_k = r (saved in tmp)
        tmp.sadd(-1.0, 1.0, src);

        VectorType correction;
        matrix->initialize_dof_vector(correction);
        // correction = D^{-1} * tmp = D^{-1} * r
        this->vmult(correction, tmp);
        // dst = dst + relaxation * correction = x_k + \omega*D^{-1} * r
        dst.add(relaxation, correction);
    }

    template<typename MatrixType>
    void JacobiSmoother<MatrixType>::Tstep(VectorType &dst, const VectorType &src) const {
        step(dst, src);
    }
}

#endif //PROJECT7_MATRIXFREE_ADR_OPERATOR_HPP