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
    virtual double value(const Point<dim> & p,
                 const unsigned int component = 0) const override;

    template <typename number>
    number value(const Point<dim, number> &p,
                 const unsigned int        component = 0) const;

    virtual void vector_value(const Point<dim> &p, Vector<double> &values) const override;

    template <typename number>
    void vector_value(const Point<dim> &p, Vector<number> &values) const;
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

template <int dim> template< typename number>
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
}

template <int dim> template< typename number>
void AdvectionCoefficient<dim>::vector_value(const Point<dim> &p, Vector<number> &values) const {
    values[0] = 0.25 * p[0];
    values[1] = 0.25 * p[0];
    values[2] = 0.0;
}

template <int dim>
void AdvectionCoefficient<dim>::vector_value(const Point<dim> &p, Vector<double> &values) const {
    return vector_value<double>(p,values);
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