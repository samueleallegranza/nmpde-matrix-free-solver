#include <general_definitions.hpp>
#include <prm_handler.hpp>

namespace ADR {
    using namespace dealii;

    template<int dim>
    void ADRParamHandler<dim>::declare_parameters() {
        prm.declare_entry(
            "Refinements",
            "5",
            Patterns::List(Patterns::Integer(1,10),1),
            "The refinement levels to test",
            true
        );
        prm.enter_subsection("Solver");
        {
            prm.declare_entry(
                "Max iters",
                "1000",
                Patterns::Integer(1,100000),
                "Max number of iters for linear solver"
            );
            prm.declare_entry(
                "Tolerance",
                "1e-10",
                Patterns::Double(),
                "Tolerance for the linear solver"
            );
            prm.declare_entry(
                "Solver type",
                "CG",
                Patterns::Selection("CG|GMRES"),
                "The solver to use"
            );
            prm.declare_entry(
                "Preconditioner",
                "Identity",
                Patterns::Selection("Jacobi|Identity|ILU|ILUT|SSOR"),
                "The type of preconditioner default is none"
            );
        }
        prm.leave_subsection();
        prm.enter_subsection("Files");
        {
            prm.declare_entry(
                "Output file",
                "./output.vtk",
                Patterns::FileName(Patterns::FileName::output),
                "A .vtk file to save the solution to"
            );
        }
        prm.leave_subsection();
        prm.enter_subsection("Functions");
        {
            prm.declare_entry(
                "Diffusion",
                "1.0",
                Patterns::Anything(),
                "Diffusion coefficient"
            );
            prm.declare_entry(
                "Advection x",
                "0.0",
                Patterns::Anything(),
                "Advection x coefficient"
            );
            prm.declare_entry(
                "Advection y",
                "0.0",
                Patterns::Anything(),
                "Advection y coefficient"
            );
            prm.declare_entry(
                "Advection z",
                "0.0",
                Patterns::Anything(),
                "Advection z coefficient"
            );
            prm.declare_entry(
                "Reaction",
                "0.0",
                Patterns::Anything(),
                "Reaction coefficient"
            );
            prm.declare_entry(
                "Force",
                "0.0",
                Patterns::Anything(),
                "Force term"
            );
            prm.declare_entry(
                "Dirichlet BC",
                "0.0",
                Patterns::List(Patterns::Anything(),0),
                "Dirichlet Boundary Condition"
            );
            prm.declare_entry(
                "Neumann BC",
                "0.0",
                Patterns::List(Patterns::Anything(),0),
                "Neumann Boundary Condition"
            );
            prm.declare_entry(
                "Dirichlet Tags",
                "0,1",
                Patterns::List(Patterns::Integer(0,9),0),
                "The Tag of the boundaries on which to apply the Dirichlet condition"
            );
            prm.declare_entry(
                "Neumann Tags",
                "0,1",
                Patterns::List(Patterns::Integer(0,9),0),
                "The Tag of the boundaries on which to apply the Neumann condition"
            );
        }
        prm.leave_subsection();
        param_initialized = true;
    }

    template<int dim>
    void ADRParamHandler<dim>::init(const String &filename) {
        if (initialized) return;

        if (!param_initialized) {
            declare_parameters();
        }

        LOG_TITLE("Initializing Elliptic parameters");
        constants["pi"] = numbers::PI;

        String diffusion_s;
        std::vector<String> advection_s(dim);
        String reaction_s;
        String force_term_s;
        std::vector<String> dirichlet_bc_s;
        std::vector<String> neumann_bc_s;

        std::vector<BoundaryIds> tags_d;
        std::vector<BoundaryIds> tags_n;

        LOG_TITLE("Reading Parameter File")
        LOG_VAR("File Name", filename)
        prm.parse_input(filename);

        std::stringstream refinements_ss(prm.get("Refinements"));
        String refinement;
        while (std::getline(refinements_ss, refinement, ',')) {
            refinements.push_back(stoi(refinement));
        }

        prm.enter_subsection("Solver");
        {
            max_iters = prm.get_integer("Max iters");
            epsilon = prm.get_double("Tolerance");
            symmetric_solver = prm.get("Solver type") == "CG";
            preconditioner = prm.get("Preconditioner");
        }
        prm.leave_subsection();
        prm.enter_subsection("Files");
        {
            output_filename = prm.get("Output file");
        }
        prm.leave_subsection();
        prm.enter_subsection("Functions");
        {
            diffusion_s = prm.get("Diffusion");
            for (int d = 0; d < dim; d++) {
                String advection_tag;
                switch (d) {
                    case 0: advection_tag = "Advection x"; break;
                    case 1: advection_tag = "Advection y"; break;
                    case 2: advection_tag = "Advection z"; break;
                    default: ;
                }
                advection_s[d] = prm.get(advection_tag);
            }
            reaction_s = prm.get("Reaction");
            force_term_s = prm.get("Force");

            String dirichlet_bc_curr, neumann_bc_curr;

            std::stringstream dirichlet_bc_ss(prm.get("Dirichlet BC"));
            std::stringstream dirichlet_tag_ss(prm.get("Dirichlet Tags"));
            while (std::getline(dirichlet_bc_ss, dirichlet_bc_curr, ',')) {
                dirichlet_bc_s.emplace_back(dirichlet_bc_curr);
            }
            while (std::getline(dirichlet_tag_ss, dirichlet_bc_curr, ',')) {
                dirichlet_bc_tags.emplace_back(static_cast<BoundaryIds>(stoi(dirichlet_bc_curr)));
            }
            for (long unsigned int i = 0; i < dirichlet_bc_tags.size(); i++) {
                dirichlet_bc.push_back(std::make_unique<FunctionParser<dim>>(1));
                dirichlet_bc[i]->initialize(variables,dirichlet_bc_s[i],constants);
            }
            std::stringstream neumann_bc_ss(prm.get("Neumann BC"));
            std::stringstream neumann_tag_ss(prm.get("Neumann Tags"));
            while (std::getline(neumann_bc_ss, neumann_bc_curr, ',')) {
                neumann_bc_s.emplace_back(neumann_bc_curr);
            }
            while (std::getline(neumann_tag_ss, neumann_bc_curr, ',')) {
                neumann_bc_tags.emplace_back(static_cast<BoundaryIds>(stoi(neumann_bc_curr)));
            }
            for (long unsigned int i = 0; i < neumann_bc_tags.size(); i++) {
                neumann_bc.push_back(std::make_unique<FunctionParser<dim>>(1));
                neumann_bc[i]->initialize(variables,neumann_bc_s[i],constants);
            }


            diffusion_c.initialize(variables,diffusion_s,constants);
            advection_c.initialize(variables,advection_s,constants);
            reaction_c.initialize(variables,reaction_s,constants);
            force_term.initialize(variables,force_term_s,constants);
        }
        prm.leave_subsection();

        LOG_TITLE("Parameters Read successful")
        LOG_VAR("Max iters",max_iters)
        LOG_VAR("Tolerance",epsilon)
        LOG_VAR("Solver", (symmetric_solver ? "CG":"GMRES") )
        LOG_VAR("Output file",output_filename)
        LOG_VAR("Diffusion Coefficient",diffusion_s)
        for (int d = 0; d < dim; d++) {
            String advection_tag;
            switch (d) {
                case 0: advection_tag = "Advection x"; break;
                case 1: advection_tag = "Advection y"; break;
                case 2: advection_tag = "Advection z"; break;
                default: ;
            }
            LOG_VAR(advection_tag, advection_s[d]);
        }
        LOG_VAR("Reaction Coefficient",reaction_s)
        LOG_VAR("Force term", (force_term_s.length() <= 66 ? force_term_s : force_term_s.substr(0,63) + "..."))
        for (long unsigned int i = 0; i < dirichlet_bc_tags.size(); i++) {
            String out = fmt::format(
                "{}Dirichlet Boundary : {}{}{},  Function: {}{}          " ,
                GREEN,
                YELLOW,dirichlet_bc_tags[i],GREEN,
                YELLOW,dirichlet_bc_s[i]
            );
            LOG_FIT(out,100);
        }
        for (long unsigned int i = 0; i < neumann_bc_tags.size(); i++) {
            String out = fmt::format(
                "{}Neumann Boundary : {}{}{},  Function: {}{}          " ,
                GREEN,
                YELLOW,neumann_bc_tags[i],GREEN,
                YELLOW,neumann_bc_s[i]
            );
            LOG_FIT(out,100);
        }
        initialized = true;
    }

    template<int dim>
    void ADRParamHandler<dim>::print_parameters(const String &filename) {
        print_parameters_as(filename,ParameterHandler::DefaultStyle);
    }

    template<int dim>
    void ADRParamHandler<dim>::print_editable_parameters(const String &filename) {
        print_parameters_as(filename,ParameterHandler::XML);
    }

    template<int dim>
    void ADRParamHandler<dim>::print_parameters_as(
        const String &filename,
        ParameterHandler::OutputStyle style
    ) {
        if (! param_initialized) {
            declare_parameters();
        }
        prm.print_parameters(filename,style);
    }

}

