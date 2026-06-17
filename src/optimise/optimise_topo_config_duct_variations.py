import logging
import os
import pyomo.environ as pyo
import uuid
from pyomo2h5 import load_yaml, ConstraintTracker
from src.optimise import (
    adjust_opt_problem,
)
from src.pyomo_models import optimal_planning
from src.optimise.utils import run_initial_solve
from src.preprocessing.general_utils import (
    add_duct_zeta_flow_noise_and_dampening_from_h5,
    add_component_dimensions_from_duct_using_h5,
)


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# silence logger of gurobipy
logging.getLogger("gurobipy").propagate = False


def main(
    max_velocity_horizontal, max_velocity_vertical, max_height, comment, outfolder_path
):
    tracker = ConstraintTracker()

    filename_topology = str(uuid.uuid4())
    filename_configuration = str(uuid.uuid4())

    logging.info("Loading file...")
    data = load_yaml(INFILE)

    for e in data["E_duct"][None]:
        if "max_velocity" not in data:
            data["max_velocity"] = {}
        data["max_velocity"][e] = (
            max_velocity_vertical
            if e in data["E_duct_vertical"][None]
            else max_velocity_horizontal
        )

    velocity_constraint = 1  # if MAX_VELOCITY is not None else 0
    print("velocity constaint is always on")
    variable_air_volume_flag = not CONTROL_STRATEGY == "CAV"

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
        capex_reduction=CAPEX_REDUCTION,
    )

    logging.info("Creating instance...")
    instance_topo = adjust_opt_problem.adjust_to_control_strategy(
        CONTROL_STRATEGY, model=model, data=data
    )
    instance_topo = adjust_opt_problem.adjust_to_duct_constraint(
        instance_topo, MAX_VELOCITY, max_height
    )

    filename = outfolder_path + filename_topology

    # max_load_case defines which load case is removed for postprocessing
    # -- this is necessary as the maximum load case could just barely become infeasible
    # when the system is laid out for slightly lower pressure losses.
    if "max_volume_flow_scenario" in data:
        max_load_case = (
            None
            if CONTROL_STRATEGY in ["CAV", "VAV-CPC"]
            else data["max_volume_flow_scenario"]
        )
    else:
        max_load_case = None

    solver = pyo.SolverFactory("gurobi", solver_io="python")

    additional_annotated_dict = {
        "Optimisation Type": {
            "Control Strategy": {"Content": CONTROL_STRATEGY},
            "Planning mode": {"Content": "Topology"},
            "Connected Optimisation": {
                "Content": filename_configuration,
                "Metadata": {
                    "Information": "Filename of Configuration Optimisation results hdf5 computed based on the results of this file"
                },
            },
        },
        "Duct Variations": {
            "Maximum duct velocity in horizontal ducts": {
                "Content": max_velocity_horizontal,
                "Metadata": {"Unit": "m/s"},
            },
            "Maximum duct velocity in vertical ducts": {
                "Content": max_velocity_vertical,
                "Metadata": {"Unit": "m/s"},
            },
            "Maximum duct height in horizontal ducts": {
                "Content": max_height,
                "Metadata": {"Unit": "m"},
            },
        },
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

    path_h5file = filename + ".h5"

    if not success:
        return success

    logging.info("Loading file...")

    data = load_yaml(INFILE)
    data = add_duct_zeta_flow_noise_and_dampening_from_h5(data, path_h5file)
    data = add_component_dimensions_from_duct_using_h5(data, path_h5file)

    data_all_instances = data
    data = adjust_opt_problem.add_acoustically_relevant_scenarios(
        data, path_h5file, CONTROL_STRATEGY
    )

    velocity_constraint = 1  # if MAX_VELOCITY is not None else 0
    ## ) optimal_preplanning_detailed_investcosts.model(
    model = optimal_planning.model(
        planning_mode="Configuration",
        duct_model=1,
        fan_model=1,
        branching_constraints=0,
        velocity_constraint=velocity_constraint,
        pressure_target_met=1,
        additional_investment_costs=1,
        reduce_fan_curves=0,
        variable_air_volume=variable_air_volume_flag,
        capex_reduction=CAPEX_REDUCTION,
    )

    logging.info("Loading all scenario acoustics instance...")

    model_all_acoustics = optimal_planning.model(
        planning_mode="Configuration",
        duct_model=1,
        fan_model=1,
        branching_constraints=0,
        velocity_constraint=velocity_constraint,
        pressure_target_met=1,
        additional_investment_costs=1,
        reduce_fan_curves=0,
        all_scenarios_acoustics=1,
        variable_air_volume=variable_air_volume_flag,
        capex_reduction=CAPEX_REDUCTION,
    )

    logging.info("Creating instance...")
    instance_config = adjust_opt_problem.adjust_to_control_strategy(
        CONTROL_STRATEGY, model=model, data=data
    )

    if FIX_ALL_FIRST_STAGE_DECISIONS:
        for e in instance_config.E_fan_station | instance_config.E_vfc:
            instance_config.ind_purchase[e].fix(instance_topo.ind_purchase[e])
        for f in instance_config.fan_set:
            instance_config.fan_ind_purchase[f].fix(instance_topo.fan_ind_purchase[f])

    instance_all_acoustics = adjust_opt_problem.adjust_to_control_strategy(
        CONTROL_STRATEGY, model=model_all_acoustics, data=data_all_instances
    )

    filename = outfolder_path + filename_configuration

    additional_annotated_dict = {
        "Optimisation Type": {
            "Control Strategy": {"Content": CONTROL_STRATEGY},
            "Planning mode": {"Content": "Configuration"},
            "Connected Optimisation": {
                "Content": filename_topology,
                "Metadata": {
                    "Information": "Filename of Topology Optimisation results hdf5 used for computing duct width and height and silencer, vfc dimensioning"
                },
            },
        },
        "Duct Variations": {
            "Maximum duct velocity in horizontal ducts": {
                "Content": max_velocity_horizontal,
                "Metadata": {"Unit": "m/s"},
            },
            "Maximum duct velocity in vertical ducts": {
                "Content": max_velocity_vertical,
                "Metadata": {"Unit": "m/s"},
            },
            "Maximum duct height in horizontal ducts": {
                "Content": max_height,
                "Metadata": {"Unit": "m"},
            },
        },
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
        acoustics_instance=instance_all_acoustics,
    )

    if success:
        logging.info(f"Saved feasible solution to {filename}.h5")
    else:
        logging.warning(f"Problem infeasible. Results saved to {filename}.h5")


def create_dir_if_not_existing(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory {path} created successfully!")
    else:
        print(f"Directory {path} already exists!")


INFILE = "yaml_opt_input_files/GPZ/standard_case.yml"
OUTFOLDER = "results/GPZ/"
MAX_VELOCITY = None  # 5
CAPEX_REDUCTION = None  # 0.5
CONTROL_STRATEGY = "VAV-VPC"
FIX_ALL_FIRST_STAGE_DECISIONS = True

if __name__ == "__main__":
    comment = input("Enter comment for file here:\n")

    max_velocity_vertical = [5, 8, 9, 10, 12]  #
    max_velocity_horizontal = [5, 8, 9, 10, 12]
    max_height = [0.4]
    for m_h in max_height:
        for mv_v in max_velocity_vertical:
            outfolder_path = (
                f"max_velocity_{mv_v}_{mv_v}ms_max_height_{m_h}m/"
                + CONTROL_STRATEGY
                + "/"
            )
            create_dir_if_not_existing(OUTFOLDER + outfolder_path)
            print(outfolder_path)
            main(mv_v, mv_v, m_h, comment, OUTFOLDER + outfolder_path)
