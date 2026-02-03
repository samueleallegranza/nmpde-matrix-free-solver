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
    virtual Tensor<1, dim, double> value(const Point<dim> & p,
                 const unsigned int component = 0) const override;

    template <typename number>
    Tensor<1, dim, number> value(const Point<dim, number> &p,
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
class ForceTerm : public Function<dim> {
public:
    virtual double value(const Point<dim> & p,
                 const unsigned int component = 0) const override;

    template <typename number>
    number value(const Point<dim, number> &p,
                 const unsigned int        component = 0) const;
};


#endif //PROJECT7_MATRIXFREE_DEFAULT_COEFFICIENT_HPP