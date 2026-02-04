#ifndef PROJECT7_MATRIXFREE_DEFAULT_COEFFICIENT_HPP
#define PROJECT7_MATRIXFREE_DEFAULT_COEFFICIENT_HPP

#include <general_definitions.hpp>
using namespace dealii;

template <int dim>
class DiffusionCoefficient : public Function<dim> {
public:
    virtual double value(const Point<dim> & p,
                 const unsigned int component = 0) const override;

    template <typename number>
    number value(const Point<dim, number> &p,
                 const unsigned int        component = 0) const;
};

template <int dim>
class ReactionCoefficient : public Function<dim> {
public:
    virtual double value(const Point<dim> & p,
                 const unsigned int component = 0) const override;

    template <typename number>
    number value(const Point<dim, number> &p,
                 const unsigned int        component = 0) const;
};

template <int dim>
class AdvectionCoefficient : public Function<dim> {
public:
    virtual void vector_value(const Point<dim> &p, Vector<double> &values) const override;

    template <typename number>
    Tensor<1, dim, number> vector_value(const Point<dim, number> &p) const;
};

template <int dim>
class DirichletBoundaryCondition : public Function<dim> {
public:
    virtual double value(const Point<dim> & p,
                 const unsigned int component = 0) const override;

    template <typename number>
    number value(const Point<dim, number> &p,
                 const unsigned int        component = 0) const;
};

template <int dim>
class NeumannBoundaryCondition : public Function<dim> {
public:
    virtual double value(const Point<dim> & p,
                 const unsigned int component = 0) const override;

    template <typename number>
    number value(const Point<dim, number> &p,
                 const unsigned int        component = 0) const;
};

template <int dim>
class ForceTerm : public Function<dim> {
public:
    virtual double value(const Point<dim> & p,
                 const unsigned int component = 0) const override;

    template <typename number>
    number value(const Point<dim, number> &p,
                 const unsigned int        component = 0) const;
};

template <int dim> template< typename number>
number DiffusionCoefficient<dim>::value(const Point<dim, number> &p, const unsigned int /*component*/) const {
    return 1. / (0.05 + 2. * p.square());
}

template <int dim>
double DiffusionCoefficient<dim>::value(const Point<dim> & p, const unsigned int component) const {
    return value<double>(p, component);
}

template <int dim> template< typename number>
number ReactionCoefficient<dim>::value(const Point<dim, number> &p, const unsigned int /*component*/) const {
    return 2.0;
}

template <int dim>
double ReactionCoefficient<dim>::value(const Point<dim> & p, const unsigned int component) const {
    return value<double>(p, component);
}

/*template <int dim> template< typename number>
number AdvectionCoefficient<dim>::value(const Point<dim, number> &p, const unsigned int component) const {
    if (component == 0)
        return 0.1 * p[0];
    if (component == 1)
        return 0.1 * p[0];
    return 0.0;
}

template <int dim>
double AdvectionCoefficient<dim>::value(const Point<dim> & p, const unsigned int component) const {
    return value<double>(p, component);
}*/

template <int dim> template <typename number>
Tensor<1, dim, number> AdvectionCoefficient<dim>::vector_value(const Point<dim, number> &p) const
{
    Tensor<1, dim, number> beta;

    beta[0] = number(0.25) * p[0];
    beta[1] = number(0.25) * p[0];
    if (dim > 2) beta[2] = number(0.0);

    return beta;
}

template <int dim>
void AdvectionCoefficient<dim>::vector_value(const Point<dim> &p, Vector<double> &values) const {
    // call our template version
    Tensor<1, dim, double> beta = this->template vector_value<double>(p);

    // results are copied into the Vector object
    for (unsigned int d = 0; d < dim; ++d)
        values[d] = beta[d];
}

template <int dim> template<typename number>
number DirichletBoundaryCondition<dim>::value(const Point<dim, number> &p, const unsigned int /*component*/) const {
    return 0.0;
}

template <int dim>
double DirichletBoundaryCondition<dim>::value(const Point<dim> & p, const unsigned int component) const {
    return value<double>(p, component);
}

template <int dim> template< typename number>
number NeumannBoundaryCondition<dim>::value(const Point<dim, number> &p, const unsigned int /*component*/) const {
    return 0.0;
}

template <int dim>
double NeumannBoundaryCondition<dim>::value(const Point<dim> & p, const unsigned int component) const {
    return value<double>(p, component);
}

template <int dim> template< typename number>
number ForceTerm<dim>::value(const Point<dim, number> &p, const unsigned int /*component*/) const {
    return 1.0;
}

template <int dim>
double ForceTerm<dim>::value(const Point<dim> & p, const unsigned int component) const {
    return value<double>(p, component);
}

#endif //PROJECT7_MATRIXFREE_DEFAULT_COEFFICIENT_HPP