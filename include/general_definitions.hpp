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

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <deal.II/base/function_parser.h>
#include <deal.II/base/parameter_handler.h>

#include <sys/resource.h>
#include <filesystem>
#include <iostream>
#include <fstream>

const unsigned int degree_finite_element = 2;
const unsigned int dimension             = 3;

#ifdef CGAL_DISABLE_ROUNDING_MATH_CHECK
#undef CGAL_DISABLE_ROUNDING_MATH_CHECK
#endif

#define MAX_OUTPUT_MESH_ELEMENTS 1000000

#if !defined(BUILD_TYPE_DEBUG)
#define LOG_VAR(name,val) ;
#define LOG(val) ;
#else
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE  "\033[34m"
#define CYAN    "\033[36m"
#define RESET   "\033[0m"

#define FILE_NAME std::filesystem::path(__FILE__).filename().string()
#define LOG_PREFIX  CYAN "LOG " << BLUE << FILE_NAME << CYAN << " at line " << BLUE << __LINE__ << CYAN << ": [ "
#define LOG_SUFFIX  " ]" << RESET
#define LOG_TITLE(title) pcout << LOG_PREFIX << CYAN \
<< "======================== " << title   << " ========================" \
<< LOG_SUFFIX <<std::endl;
#define LOG_VAR(name,val) pcout << LOG_PREFIX << GREEN << name << CYAN << " == " << YELLOW << val << CYAN << LOG_SUFFIX <<std::endl;
#define LOG(val) pcout << LOG_PREFIX << YELLOW << val << CYAN << LOG_SUFFIX <<std::endl;
#endif

using namespace dealii;

#endif //PROJECT7_MATRIXFREE_GENERAL_DEFINITIONS_HPP