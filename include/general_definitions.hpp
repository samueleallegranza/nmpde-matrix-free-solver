/**
 * Library with general definitions used all throughout the project
 * @file general_definitions.hpp
 */
#ifndef PROJECT7_MATRIXFREE_GENERAL_DEFINITIONS_HPP
#define PROJECT7_MATRIXFREE_GENERAL_DEFINITIONS_HPP

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/memory_consumption.h> // Included for memory stats

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>


#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/mg_coarse.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <deal.II/base/function_parser.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/meshworker/mesh_loop.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>

#include <sys/resource.h>
#include <filesystem>
#include <iostream>
#include<string>
#include <fstream>
#include <fmt/format.h>

/// @def The maximum number of mesh elements that is acceptable to request to print the files for
#define MAX_OUTPUT_MESH_ELEMENTS 1000000

/// The degree of the finite element to use for all solvers. Quadrature degrees are usually this plus one
const unsigned int degree_finite_element = 2;

/// @def The escape sequence to print in green on the shell
#define GREEN   "\033[32m"

/// @def The escape sequence to print in yellow on the shell
#define YELLOW  "\033[33m"

/// @def The escape sequence to print in blue on the shell
#define BLUE  "\033[34m"

/// @def The escape sequence to print in cyan on the shell
#define CYAN    "\033[36m"

/// @def The escape sequence to print in red on the shell
#define RED    "\033[31m"

/// @def The escape sequence to reset foreground colors on the shell
#define RESET   "\033[0m"

/// @def The escape sequence to print in bold on the shell
#define BOLD    "\033[1m"


/// @def A macro that expands to the file name of each file as an std::string
#define FILE_NAME fmt::format("{:<30}",std::filesystem::path(__FILE__).filename().string())
/// @def A prefix to use before logs containing the file name and line
#define LOG_PREFIX  CYAN "LOG " << BLUE << FILE_NAME << CYAN << " at line " << BLUE << fmt::format("{:>04d}",__LINE__) << CYAN << ":\t [ "
/// @def A suffix to use after the logs. This resets the colors.
#define LOG_SUFFIX  " ]" << RESET


#if !defined(BUILD_TYPE_DEBUG)

    #define LOG_IMPORTANT(title) ;
    #define LOG_VAR(name,val) ;
    #define LOG(val) ;
    #define LOG_TITLE(title) ;
    #define LOG_ANY(formt,arg) ;
    #define LOG_FIT(val,space)  ;

#else

    /**
     * @def A Macro to print a beautifully spaced Title in red
     * @param title The title to be printed
     */
    #define LOG_IMPORTANT(title) pcout << LOG_PREFIX << CYAN \
        << "================ " << RED << fmt::format("{:^46}",title) << CYAN << " ================" \
        << LOG_SUFFIX <<std::endl;

    /**
     * @def A Macro to print a beautifully spaced Title
     * @param title The title to be printed
     */
    #define LOG_TITLE(title) pcout << LOG_PREFIX << CYAN \
        << "================ " << fmt::format("{:^46}",title) << " ================" \
        << LOG_SUFFIX <<std::endl;

    /**
     * @def A Macro to print a beautifully spaced name for a variable followed by a value
     * @param name a string to be prefixed to the value
     * @param val the value of a variable
     */
    #define LOG_VAR(name,val) pcout << LOG_PREFIX << GREEN \
        << fmt::format("{:<90}",fmt::format("{}{} == {}{}",name, CYAN, YELLOW, val)) \
        << CYAN << LOG_SUFFIX <<std::endl;

    /**
     * @def A Macro to print a beautifully spaced string
     * @param val a string
     */
    #define LOG(val) pcout << LOG_PREFIX << GREEN \
        << fmt::format("{:<80}",val) \
        << CYAN << LOG_SUFFIX <<std::endl;

    /**
     * @def A Macro to print a beautifully spaced string, spacing is adjusted to account for
     * @param val a string
     * @param space the spacing, the length needs to account for the length of the escape characters
     */
    #define LOG_FIT(val,space) pcout << LOG_PREFIX << YELLOW \
        << fmt::format("{:<"+std::to_string(space)+"}",val) \
        << CYAN << LOG_SUFFIX <<std::endl;
    /**
     * @def A Macro to print a variable between formatted text
     * @param formt the fmt format to be used
     * @param arg the argument to be printed
     */
    #define LOG_ANY(formt,arg) pcout << LOG_PREFIX << YELLOW \
            << fmt::format("{:<80}",fmt::format(formt,arg)) \
            << CYAN << LOG_SUFFIX <<std::endl;

#endif

/**
 * @brief Converts a 3-dimensional function to a vector
 * Utility to make the code more readable,
 * functions needs to be turned into tensors to multiply them with deal.ii test functions
 * @tparam dim the dimension of the tensor
 * @param function the function to be transformed
 * @param p the point at which the function is computed
 * @return the tensor that the function represents at point p
 */
template <int dim>
dealii::Tensor<1,dim> to_tensor(const dealii::Function<dim> &function,const dealii::Point<dim> &p) {
    dealii::Tensor<1,dim> tensor_value;

    if constexpr (dim >= 1) tensor_value[0] = function.value(p,0);
    if constexpr (dim >= 2) tensor_value[1] = function.value(p,1);
    if constexpr (dim >= 3) tensor_value[2] = function.value(p,2);

    return tensor_value;
}

#endif //PROJECT7_MATRIXFREE_GENERAL_DEFINITIONS_HPP