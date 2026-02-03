#include <default_coefficient.hpp>

template <int dim> template< typename number>
number DiffusionCoefficient<dim>::value(const Point<dim, number> &p, const unsigned int /*component*/) const {
    return 1. / (0.05 + 2. * p.square());
}

template <int dim>
double DiffusionCoefficient<dim>::value(const Point<dim> & p, const unsigned int component) const {
    return value<double>(p, component);
}

template <int dim> template< typename number>
Tensor<1, dim, number> AdvectionCoefficient<dim>::value(const Point<dim, number> &p, const unsigned int /*component*/) const {
    Tensor<1, dim, number> beta;
    beta[0] = 1.0;
    beta[1] = 0.5;
    if (dim > 2) beta[2] = 0.0;
    return beta;
}

template <int dim>
Tensor<1, dim, double> AdvectionCoefficient<dim>::value(const Point<dim> & p, const unsigned int component) const {
    return value<double>(p, component);
}

template <int dim> template< typename number>
number ReactionCoefficient<dim>::value(const Point<dim, number> &p, const unsigned int /*component*/) const {
    return 1.0;
}

template <int dim>
double ReactionCoefficient<dim>::value(const Point<dim> & p, const unsigned int component) const {
    return value<double>(p, component);
}

template <int dim> template< typename number>
number DirichletBoundaryCondition<dim>::value(const Point<dim, number> &p, const unsigned int /*component*/) const {
    return 0.0;
}

template <int dim>
double DirichletBoundaryCondition<dim>::value(const Point<dim> & p, const unsigned int component) const {
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