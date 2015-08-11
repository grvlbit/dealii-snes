/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2012 - 2014 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Author : Gunnar Jansen, University of Neuchatel, 2015
 * based on a the work of: Sven Wetterauer, University of Heidelberg, 2012
 */



#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <petscsnes.h>

#include <fstream>
#include <iostream>


#include <deal.II/numerics/solution_transfer.h>

namespace Step15
{
  using namespace dealii;

  template <int dim>
  class MinimalSurfaceProblem
  {
  public:
    MinimalSurfaceProblem ();
    ~MinimalSurfaceProblem ();

    void run ();

  private:
    void setup_system (const bool initial_step);
    void set_boundary_values ();

    void assemble_system (PETScWrappers::Vector &present_solution);
    void assemble_rhs (PETScWrappers::Vector &present_solution,PETScWrappers::Vector &system_rhs);

    static PetscErrorCode FormFunction(SNES snes , Vec x, Vec f, void* ctx);
    static PetscErrorCode FormJacobian(SNES snes, Vec x, Mat* jac, Mat* B,
                                       MatStructure *flag, void* ctx);

    Triangulation<dim>   triangulation;

    DoFHandler<dim>      dof_handler;
    FE_Q<dim>            fe;

    ConstraintMatrix     hanging_node_constraints;

    SparsityPattern      sparsity_pattern;
    PETScWrappers::SparseMatrix system_matrix;

    PETScWrappers::Vector       present_solution;
    PETScWrappers::Vector       system_rhs;
  };



  template <int dim>
  class BoundaryValues : public Function<dim>
  {
  public:
    BoundaryValues () : Function<dim>() {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
  };


  template <int dim>
  double BoundaryValues<dim>::value (const Point<dim> &p,
                                     const unsigned int /*component*/) const
  {
    return std::sin(2 * numbers::PI * (p[0]+p[1]));
  }


  template <int dim>
  PetscErrorCode MinimalSurfaceProblem<dim>::FormFunction(SNES snes , Vec x, Vec f, void* ctx)
  {
     auto p_ctx = reinterpret_cast<MinimalSurfaceProblem<dim>*>(ctx);
     PETScWrappers::Vector x_wrap(x);
     PETScWrappers::Vector f_wrap(f);

     p_ctx->assemble_rhs(x_wrap,f_wrap);
     return 0;
  }

  template <int dim>
  PetscErrorCode MinimalSurfaceProblem<dim>::FormJacobian(SNES snes, Vec x, Mat* jac, Mat* B,
                                                    MatStructure *flag, void* ctx)
  {
    auto p_ctx = reinterpret_cast<MinimalSurfaceProblem<dim>*>(ctx);

    PETScWrappers::Vector x_wrap(x);
    p_ctx->assemble_system(x_wrap);

    /*
       Assemble matrix
    */
    PetscErrorCode ierr;
    ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    if (jac != B) {
      ierr = MatAssemblyBegin(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    }
    return 0;
  }



  template <int dim>
  MinimalSurfaceProblem<dim>::MinimalSurfaceProblem ()
    :
    dof_handler (triangulation),
    fe (2)
  {}



  template <int dim>
  MinimalSurfaceProblem<dim>::~MinimalSurfaceProblem ()
  {
    dof_handler.clear ();
  }



  template <int dim>
  void MinimalSurfaceProblem<dim>::setup_system (const bool initial_step)
  {
    if (initial_step)
      {
        dof_handler.distribute_dofs (fe);
        present_solution.reinit (dof_handler.n_dofs());
      }

    hanging_node_constraints.clear ();
    DoFTools::make_hanging_node_constraints (dof_handler,
                                             hanging_node_constraints);

    VectorTools::interpolate_boundary_values (dof_handler,
                                      0,
                                      ZeroFunction<dim>(),
                                      hanging_node_constraints);
    hanging_node_constraints.close ();

    system_rhs.reinit (dof_handler.n_dofs());

    CompressedSparsityPattern c_sparsity(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern (dof_handler, c_sparsity);

    hanging_node_constraints.condense (c_sparsity);

    sparsity_pattern.copy_from(c_sparsity);
    system_matrix.reinit (sparsity_pattern);
  }


  template <int dim>
  void MinimalSurfaceProblem<dim>::assemble_system (PETScWrappers::Vector &present_solution)
  {
    const QGauss<dim>  quadrature_formula(3);

    system_matrix = 0;

    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_gradients         |
                             update_quadrature_points |
                             update_JxW_values);

    const unsigned int           dofs_per_cell = fe.dofs_per_cell;
    const unsigned int           n_q_points    = quadrature_formula.size();

    FullMatrix<double>           cell_matrix (dofs_per_cell, dofs_per_cell);

    std::vector<Tensor<1, dim> > old_solution_gradients(n_q_points);

    std::vector<types::global_dof_index>    local_dof_indices (dofs_per_cell);

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell)
      {
        cell_matrix = 0;

        fe_values.reinit (cell);

        fe_values.get_function_gradients(present_solution,
                                         old_solution_gradients);

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          {
            const double coeff
              = 1.0 / std::sqrt(1 +
                                old_solution_gradients[q_point] *
                                old_solution_gradients[q_point]);

            for (unsigned int i=0; i<dofs_per_cell; ++i)
              {
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                  {
                    cell_matrix(i, j) += (fe_values.shape_grad(i, q_point)
                                          * coeff
                                          * (fe_values.shape_grad(j, q_point)
                                             -
                                             coeff * coeff
                                             * (fe_values.shape_grad(j, q_point)
                                                *
                                                old_solution_gradients[q_point])
                                             * old_solution_gradients[q_point]
                                            )
                                          * fe_values.JxW(q_point));
                  }
              }
          }

        cell->get_dof_indices (local_dof_indices);
        hanging_node_constraints
                     .distribute_local_to_global (cell_matrix,
                                                  local_dof_indices,
                                                  system_matrix);
      }


      system_matrix.compress(VectorOperation::add);

  }


  template <int dim>
  void MinimalSurfaceProblem<dim>::assemble_rhs(
    PETScWrappers::Vector &present_solution, PETScWrappers::Vector &system_rhs)
  {
    const QGauss<dim>  quadrature_formula(3);

    system_rhs = 0;

    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_gradients         |
                             update_quadrature_points |
                             update_JxW_values);

    const unsigned int           dofs_per_cell = fe.dofs_per_cell;
    const unsigned int           n_q_points    = quadrature_formula.size();

    FullMatrix<double>           cell_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>               cell_rhs (dofs_per_cell);

    std::vector<Tensor<1, dim> > old_solution_gradients(n_q_points);

    std::vector<types::global_dof_index>    local_dof_indices (dofs_per_cell);

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell)
      {
        cell_matrix = 0;
        cell_rhs = 0;

        fe_values.reinit (cell);

        fe_values.get_function_gradients(present_solution,
                                         old_solution_gradients);

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          {
            const double coeff
              = 1.0 / std::sqrt(1 +
                                old_solution_gradients[q_point] *
                                old_solution_gradients[q_point]);

            for (unsigned int i=0; i<dofs_per_cell; ++i)
              {
                if (hanging_node_constraints.is_inhomogeneously_constrained(local_dof_indices[i]))
                {
                  for (unsigned int j=0; j<dofs_per_cell; ++j)
                    {
                      cell_matrix(i, j) += (fe_values.shape_grad(i, q_point)
                                            * coeff
                                            * (fe_values.shape_grad(j, q_point)
                                               -
                                               coeff * coeff
                                               * (fe_values.shape_grad(j, q_point)
                                                  *
                                                  old_solution_gradients[q_point])
                                               * old_solution_gradients[q_point]
                                              )
                                            * fe_values.JxW(q_point));
                    }
                }
                cell_rhs(i) += (fe_values.shape_grad(i, q_point)
                                * coeff
                                * old_solution_gradients[q_point]
                                * fe_values.JxW(q_point));
              }
          }

        cell->get_dof_indices (local_dof_indices);
        hanging_node_constraints
                          .distribute_local_to_global (cell_rhs,
                                                       local_dof_indices,
                                                       system_rhs, cell_matrix);

        system_rhs.compress(VectorOperation::add);

      }
  }



  template <int dim>
  void MinimalSurfaceProblem<dim>::set_boundary_values ()
  {
    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values (dof_handler,
                                              0,
                                              BoundaryValues<dim>(),
                                              boundary_values);
    for (std::map<types::global_dof_index, double>::const_iterator
         p = boundary_values.begin();
         p != boundary_values.end(); ++p)
      present_solution(p->first) = p->second;
  }



  template <int dim>
  void MinimalSurfaceProblem<dim>::run ()
  {
    unsigned int refinement = 0;
    bool         first_step = true;

    GridGenerator::hyper_ball (triangulation);
    static const HyperBallBoundary<dim> boundary;
    triangulation.set_boundary (0, boundary);
    triangulation.refine_global(2);

    setup_system (true);
    set_boundary_values ();

    // SNES Stuff here
    SNES snes;

    PetscErrorCode ierr;

    ierr = SNESCreate(MPI_COMM_WORLD, &snes);
    ierr = SNESSetFunction(snes, system_rhs, FormFunction, this);
    ierr = SNESSetJacobian(snes, system_matrix, system_matrix, FormJacobian, this);
    ierr = SNESSetFromOptions(snes);

    ierr = SNESSolve(snes, NULL, present_solution);

    PetscInt its;
    SNESGetIterationNumber(snes,&its);
    PetscPrintf(MPI_COMM_WORLD,"Number of SNES iterations = %D\n",its);

    ierr = SNESDestroy(&snes);

    DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (present_solution, "solution");
    data_out.build_patches ();
    const std::string filename = "solution.vtk";
    std::ofstream output (filename.c_str());
    data_out.write_vtk (output);

  }
}


int main (int argc,char **argv)
{
  try
    {
      using namespace dealii;
      using namespace Step15;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      deallog.depth_console (0);

      MinimalSurfaceProblem<2> laplace_problem_2d;
      laplace_problem_2d.run ();

    //  MinimalSurfaceProblem<3> laplace_problem_3d;
    //  laplace_problem_3d.run ();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  return 0;
}
