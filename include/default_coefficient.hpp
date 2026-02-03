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
class AdvectionCoefficient : public Function<dim> {
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
number AdvectionCoefficient<dim>::value(const Point<dim, number> &p, const unsigned int /*component*/) const {
    // Tensor<1, dim, number> beta;
    // beta[0] = 1.0;
    // beta[1] = 0.5;
    // if (dim > 2) beta[2] = 0.0;
    return 2.0;
}

template <int dim>
double AdvectionCoefficient<dim>::value(const Point<dim> & p, const unsigned int component) const {
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