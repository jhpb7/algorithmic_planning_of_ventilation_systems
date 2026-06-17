import logging
import os
import numpy as np
import pyomo.environ as pyo
import uuid
from pyomo2h5 import load_yaml, ConstraintTracker
from src.preplanning.optimise import (
    adjust_opt_problem,
    # optimal_preplanning_detailed_investcosts,
)
from src.conceptplanning.preprocessing import optimal_planning
from src.preplanning.optimise.utils import run_initial_solve
from src.preplanning.preprocessing.general_utils import (
    add_duct_zeta_flow_noise_and_dampening_from_h5,
    add_component_dimensions_from_duct_using_h5,
)



logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# silence logger of gurobipy
logging.getLogger("gurobipy").propagate = False


def prepare_config_opt(data, path_h5file, instance_topo, variable_air_volume_flag):

    data = add_duct_zeta_flow_noise_and_dampening_from_h5(data, path_h5file)
    data = add_component_dimensions_from_duct_using_h5(data, path_h5file)

    data_all_instances = data
    data = adjust_opt_problem.add_acoustically_relevant_scenarios(data, path_h5file, cs)

    model = optimal_planning.model(
        planning_mode="Configuration",
        duct_model=1,
        fan_model=1,
        branching_constraints=0,
        velocity_constraint=0,
        pressure_target_met=1,
        additional_investment_costs=1,
        reduce_fan_curves=0,
        variable_air_volume=variable_air_volume_flag,
    )

    logging.info("Loading all scenario acoustics instance...")

    model_all_acoustics = optimal_planning.model(
        planning_mode="Configuration",
        duct_model=1,
        fan_model=1,
        branching_constraints=0,
        velocity_constraint=0,
        pressure_target_met=1,
        additional_investment_costs=1,
        reduce_fan_curves=0,
        all_scenarios_acoustics=1,
        variable_air_volume=variable_air_volume_flag,
    )

    logging.info("Creating instance...")
    instance_config = adjust_opt_problem.adjust_to_control_strategy(
        cs, model=model, data=data
    )

    for e in instance_config.E_fan_station | instance_config.E_vfc:
        instance_config.ind_purchase[e].fix(instance_topo.ind_purchase[e])
    for f in instance_config.fan_set:
        instance_config.fan_ind_purchase[f].fix(instance_topo.fan_ind_purchase[f])

    instance_config_all_acoustics = adjust_opt_problem.adjust_to_control_strategy(
        cs, model=model_all_acoustics, data=data_all_instances
    )
    return instance_config, instance_config_all_acoustics


def main(cs, comment):
    tracker = ConstraintTracker()

    filename_topology = str(uuid.uuid4())
    filename_configuration = str(uuid.uuid4())

    logging.info("Loading file...")
    data = load_yaml(INFILE)

    for e in data["E_duct"][None]:
        if "max_velocity" not in data:
            data["max_velocity"] = {}
        data["max_velocity"][e] = (
            MAX_VELOCITY_VERTICAL
            if e in data["E_duct_vertical"][None]
            else MAX_VELOCITY_HORIZONTAL
        )

    velocity_constraint = 1  # if MAX_VELOCITY is not None else 0
    logging.warning("Velocity constraint is always on!")
    variable_air_volume_flag = not cs == "CAV"

    model = optimal_planning.model(
        planning_mode="Topology",
        duct_model=1,
        fan_model=1,
        branching_constraints=0,
        velocity_constraint=velocity_constraint,
        pressure_target_met=1,
        additional_investment_costs=1,
        reduce_fan_curves=0,
        variable_air_volume=variable_air_volume_flag,
    )

    logging.info("Creating instance...")
    instance_topo = adjust_opt_problem.adjust_to_control_strategy(
        cs, model=model, data=data
    )
    instance_topo = adjust_opt_problem.adjust_to_duct_constraint(
        instance_topo, MAX_VELOCITY, MAX_HEIGHT
    )

    filename = OUTFOLDER + cs + "/" + filename_topology

    # max_load_case defines which load case is removed for postprocessing
    # -- this is necessary as the maximum load case could just barely become infeasible
    # when the system is laid out for slightly lower pressure losses.
    if "max_volume_flow_scenario" in data:
        max_load_case = (
            None if cs in ["CAV", "VAV-CPC"] else data["max_volume_flow_scenario"]
        )
    else:
        max_load_case = None

    solver = pyo.SolverFactory("gurobi", solver_io="python")

    additional_annotated_dict = {
        "Optimization Type": {
            "Control Strategy": {"Content": cs},
            "Planning mode": {"Content": "Topology"},
            "Connected Optimisation": {
                "Content": filename_configuration,
                "Metadata": {
                    "Information": "Filename of Configuration Optimisation results hdf5 computed based on the results of this file"
                },
            },
        }
    }

    success = run_initial_solve(
        instance_topo,
        solver,
        tracker,
        filename,
        "Topology",
        comment,
        max_load_case,
        additional_annotated_dict=additional_annotated_dict,
        acoustics_instance=None,
    )

    if success:
        path_h5file = filename + ".h5"

        logging.info("Loading file...")

        data = load_yaml(INFILE)

        instance_config, instance_config_all_acoustics = prepare_config_opt(
            data, path_h5file, instance_topo, variable_air_volume_flag
        )

        filename = OUTFOLDER + cs + "/" + filename_configuration

        additional_annotated_dict = {
            "Optimization Type": {
                "Control Strategy": {"Content": cs},
                "Planning mode": {"Content": "Configuration"},
                "Connected Optimisation": {
                    "Content": filename_configuration,
                    "Metadata": {
                        "Information": "Filename of Topology Optimisation results hdf5 used for computing duct width and height and silencer, vfc dimensioning"
                    },
                },
            }
        }

        success = run_initial_solve(
            instance_config,
            solver,
            tracker,
            filename,
            "Configuration",
            comment,
            max_load_case,
            additional_annotated_dict=additional_annotated_dict,
            acoustics_instance=instance_config_all_acoustics,
        )

        if success:
            logging.info(f"Saved feasible solution to {filename}.h5")
        else:
            logging.warning(f"Problem infeasible. Results saved to {filename}.h5")

    ## Now pareto

    def run_pareto_loop(
        bound_expr, bound_start, obj_expr, stepsize, pareto_name, instance_topo
    ):
        success = True
        while success:
            logging.info(f"now solving with {pareto_name} Limit: {bound_start}\n\n")

            filename_topology = str(uuid.uuid4())
            filename_configuration = str(uuid.uuid4())

            @instance_topo.Constraint()
            def pareto_limit(m):
                return bound_expr(m) <= bound_start

            tracker.add(instance_topo.pareto_limit)

            filename = OUTFOLDER + cs + "/" + filename_topology

            additional_annotated_dict = {
                "Optimization Type": {
                    "Control Strategy": {"Content": cs},
                    "Planning mode": {"Content": "Topology"},
                    "Pareto Limit": {"Content": f"{pareto_name} limit: {bound_start}"},
                    "Connected Optimisation": {
                        "Content": filename_topology,
                        "Metadata": {
                            "Information": "Filename of Topology Optimisation results hdf5 used for computing duct width and height and silencer, vfc dimensioning"
                        },
                    },
                }
            }

            success = run_initial_solve(
                instance_topo,
                solver,
                tracker,
                filename,
                "Topology",
                comment,
                max_load_case,
                additional_annotated_dict=additional_annotated_dict,
                acoustics_instance=None,
            )

            tracker.delete(instance_topo.pareto_limit)
            instance_topo.del_component(pareto_limit)
            if not success:
                break

            # configuration opt

            path_h5file = filename + ".h5"

            instance_config, instance_config_all_acoustics = prepare_config_opt(
                data, path_h5file, instance_topo, variable_air_volume_flag
            )

            instance_config.min_lcc.deactivate()

            @instance_config.Objective()
            def pareto_obj_costs(model):
                return (
                    obj_expr(model)
                    + 1e-6
                    * sum(
                        model.scenario[s].sound_power_level[i, f]
                        for s in model.max_noise_scenarios
                        for (i) in model.V
                        for f in model.intervals
                    )
                    + 1e-4
                    * sum(
                        model.scenario[s].sound_pressure_level_room[v]
                        for s in model.max_noise_scenarios
                        for v in model.V_room
                    )
                )

            @instance_config.Constraint()
            def pareto_limit(m):
                return bound_expr(m) <= bound_start

            tracker.add(instance_config.pareto_limit)

            filename = OUTFOLDER + cs + "/" + filename_configuration

            additional_annotated_dict = {
                "Optimization Type": {
                    "Control Strategy": {"Content": cs},
                    "Planning mode": {"Content": "Configuration"},
                    "Pareto Limit": {"Content": f"{pareto_name} limit: {bound_start}"},
                    "Connected Optimisation": {
                        "Content": filename_topology,
                        "Metadata": {
                            "Information": "Filename of Topology Optimisation results hdf5 used for computing duct width and height and silencer, vfc dimensioning"
                        },
                    },
                }
            }

            run_initial_solve(
                instance_config,
                solver,
                tracker,
                filename,
                "Configuration",
                comment,
                max_load_case,
                additional_annotated_dict=additional_annotated_dict,
                acoustics_instance=instance_config_all_acoustics,
            )
            tracker.delete(instance_config.pareto_limit)
            instance_config.del_component(pareto_limit)

            bound_start -= stepsize

    energy_cost_pareto = (
        np.floor(instance_topo.fan_energy_costs.expr() / STEPSIZE_ENERGY)
        * STEPSIZE_ENERGY
    )
    invest_cost_pareto = (
        np.floor(instance_topo.total_invest_costs.expr() / STEPSIZE_INVEST)
        * STEPSIZE_INVEST
    )

    instance_topo.min_lcc.deactivate()

    @instance_topo.Objective()
    def min_invest_costs(m):
        return m.total_invest_costs

    run_pareto_loop(
        bound_expr=lambda m: m.fan_energy_costs,
        bound_start=energy_cost_pareto,
        obj_expr=lambda m: m.total_invest_costs,
        stepsize=STEPSIZE_ENERGY,
        pareto_name="Energy",
        instance_topo=instance_topo,
    )

    instance_topo.min_invest_costs.deactivate()

    @instance_topo.Objective()
    def min_energy_costs(m):
        return m.fan_energy_costs

    run_pareto_loop(
        bound_expr=lambda m: m.total_invest_costs,
        bound_start=invest_cost_pareto,
        obj_expr=lambda m: m.fan_energy_costs,
        stepsize=STEPSIZE_INVEST,
        pareto_name="Invest",
        instance_topo=instance_topo,
    )

def create_dir_if_not_existing(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory {path} created successfully!")
    else:
        print(f"Directory {path} already exists!")


INFILE = "opt_problems/preplanning/GPZ/standard_case.yml"
OUTFOLDER = "results/GPZ/combined/max_velocity_5_10ms_max_height_04m/pareto_opt/"
MAX_VELOCITY = None  # 5
MAX_VELOCITY_VERTICAL = 10
MAX_VELOCITY_HORIZONTAL = 5
MAX_HEIGHT = 0.4  # -999 #if None then height=width, if negative then turned off
STEPSIZE_ENERGY = 125
STEPSIZE_INVEST = 5e4

if __name__ == "__main__":
    comment = input("Enter comment for file here:\n")
    css = ["VAV-VPC"]  # , ]# "ONLY-DF",
    for cs in css:
        create_dir_if_not_existing(
            OUTFOLDER + cs,
        )
        logging.info(f"\n\nControl strategy: {cs}\n\n")
        main(cs, comment)
