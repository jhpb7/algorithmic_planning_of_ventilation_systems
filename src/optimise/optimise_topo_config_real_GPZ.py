import logging
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

INFILE = "yaml_opt_input_files/GPZ/real_GPZ.yml"
OUTFOLDER = "results/GPZ/"
CONTROL_STRATEGY = "VAV-VPC"
MAX_VELOCITY = 8  # normally: 5
MAX_HEIGHT = -99  # 0.4

fix_duct_data = {
    "height": {
        ("0~4", "0~5"): 0.7,
        ("0~5", "1~1"): 0.35,
        ("0~5", "2~1"): 0.35,
        ("1~3", "1~4"): 0.35,
        ("1~4", "1-1~1"): 0.25,
        ("1~4", "1-2~1"): 0.35,
        ("1-2~3", "1-2~4"): 0.35,
        ("1-2~4", "1-2-1~1"): 0.2,
        ("1-2~4", "1-2-2~1"): 0.35,
        ("1-2-1~4", "1-2-1~5"): 0.2,
        ("1-2-2~3", "1-2-2~4"): 0.35,
        ("1-2-2~4", "1-2-2-1~1"): 0.2,
        ("1-2-2~4", "1-2-2-2~1"): 0.2,
        ("1-2-2-1~4", "1-2-2-1~5"): 0.25,
        ("1-2-2-2~4", "1-2-2-2~5"): 0.2,
        ("2~3", "2~4"): 0.35,
        ("2~4", "2-1~1"): 0.3,
        ("2~4", "2-2~1"): 0.35,
        ("2-1~4", "2-1~5"): 0.2,
        ("2-2~3", "2-2~4"): 0.35,
        ("2-2~4", "2-2-1~1"): 0.2,
        ("2-2~4", "2-2-2~1"): 0.3,
        ("2-2-1~4", "2-2-1~5"): 0.2,
        ("2-2-2~4", "2-2-2~5"): 0.2,
    },
    "width": {
        ("0~4", "0~5"): 0.85,
        ("0~5", "1~1"): 0.85,
        ("0~5", "2~1"): 0.85,
        ("1~3", "1~4"): 0.8,
        ("1~4", "1-1~1"): 0.7,
        ("1~4", "1-2~1"): 0.8,
        ("1-2~3", "1-2~4"): 0.8,
        ("1-2~4", "1-2-1~1"): 0.35,
        ("1-2~4", "1-2-2~1"): 0.8,
        ("1-2-1~4", "1-2-1~5"): 0.25,
        ("1-2-2~3", "1-2-2~4"): 0.5,
        ("1-2-2~4", "1-2-2-1~1"): 0.25,
        ("1-2-2~4", "1-2-2-2~1"): 0.5,
        ("1-2-2-1~4", "1-2-2-1~5"): 0.25,
        ("1-2-2-2~4", "1-2-2-2~5"): 0.4,
        ("2~3", "2~4"): 0.85,
        ("2~4", "2-1~1"): 0.8,
        ("2~4", "2-2~1"): 0.5,
        ("2-1~4", "2-1~5"): 0.6,
        ("2-2~3", "2-2~4"): 0.5,
        ("2-2~4", "2-2-1~1"): 0.35,
        ("2-2~4", "2-2-2~1"): 0.35,
        ("2-2-1~4", "2-2-1~5"): 0.2,
        ("2-2-2~4", "2-2-2~5"): 0.5,
    },
}

fixed_sil_dict = {
    "purchase": {
        ("0~2", "0~3"): 1,
        ("1~2", "1~3"): 0,
        ("1-1~3", "1-1~4"): 1,
        ("1-2~2", "1-2~3"): 0,
        ("1-2-1~3", "1-2-1~4"): 1,
        ("1-2-2~2", "1-2-2~3"): 0,
        ("1-2-2-1~3", "1-2-2-1~4"): 1,
        ("1-2-2-2~3", "1-2-2-2~4"): 1,
        ("2~2", "2~3"): 0,
        ("2-1~3", "2-1~4"): 1,
        ("2-2~2", "2-2~3"): 0,
        ("2-2-1~3", "2-2-1~4"): 1,
        ("2-2-2~3", "2-2-2~4"): 1,
    },
    "height": {
        ("0~2", "0~3"): 1,
        ("1-1~3", "1-1~4"): 0.2,
        ("1-2-1~3", "1-2-1~4"): 0.2,
        ("1-2-2-1~3", "1-2-2-1~4"): 0.2,
        ("1-2-2-2~3", "1-2-2-2~4"): 0.2,
        ("2-1~3", "2-1~4"): 0.2,
        ("2-2-1~3", "2-2-1~4"): 0.1,
        ("2-2-2~3", "2-2-2~4"): 0.2,
    },
    "width": {
        ("0~2", "0~3"): 1.5,
        ("1-1~3", "1-1~4"): 0.6,
        ("1-2-1~3", "1-2-1~4"): 0.4,
        ("1-2-2-1~3", "1-2-2-1~4"): 0.4,
        ("1-2-2-2~3", "1-2-2-2~4"): 0.6,
        ("2-1~3", "2-1~4"): 0.6,
        ("2-2-1~3", "2-2-1~4"): 0.3,
        ("2-2-2~3", "2-2-2~4"): 0.6,
    },
    "length": {
        ("0~2", "0~3"): 0.94,
        ("1-1~3", "1-1~4"): 1.5,
        ("1-2-1~3", "1-2-1~4"): 1.5,
        ("1-2-2-1~3", "1-2-2-1~4"): 1.5,
        ("1-2-2-2~3", "1-2-2-2~4"): 1.5,
        ("2-1~3", "2-1~4"): 1.5,
        ("2-2-1~3", "2-2-1~4"): 1.5,
        ("2-2-2~3", "2-2-2~4"): 1.5,
    },
}

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# silence logger of gurobipy
logging.getLogger("gurobipy").propagate = False


def main(fixed_duct=True, fix_silencer=True):
    comment = input("Enter comment for file here:\n")
    tracker = ConstraintTracker()

    filename_topology = str(uuid.uuid4())
    filename_configuration = str(uuid.uuid4())

    logging.info("Loading file...")
    data = load_yaml(INFILE)

    for e in data["E_duct"][None]:
        data["duct_width_max"][e] = fix_duct_data["width"][e]
        data["duct_height_max"][e] = fix_duct_data["height"][e]
        data["duct_width_min"][e] = fix_duct_data["width"][e]
        data["duct_height_min"][e] = fix_duct_data["height"][e]

    velocity_constraint = 1 if MAX_VELOCITY is not None else 0
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
    )

    logging.info("Creating instance...")
    instance = adjust_opt_problem.adjust_to_control_strategy(
        CONTROL_STRATEGY, model=model, data=data
    )
    instance = adjust_opt_problem.adjust_to_duct_constraint(
        instance, MAX_VELOCITY, MAX_HEIGHT
    )

    instance.leaf_component_decision.fix(0)

    filename = OUTFOLDER + CONTROL_STRATEGY + "/" + filename_topology

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
        }
    }

    success = run_initial_solve(
        instance,
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

    logging.info("Loading file...")

    data = load_yaml(INFILE)
    data = add_duct_zeta_flow_noise_and_dampening_from_h5(data, path_h5file)
    data = add_component_dimensions_from_duct_using_h5(data, path_h5file)

    if fix_silencer:
        for e_sil in data["E_silencer"][None]:
            if e_sil in fixed_sil_dict["width"]:
                data["silencer_width"][e_sil] = fixed_sil_dict["width"][e_sil]
                data["silencer_height"][e_sil] = fixed_sil_dict["height"][e_sil]
                data["silencer_length_min"][e_sil] = fixed_sil_dict["length"][e_sil]
                data["silencer_length_max"][e_sil] = fixed_sil_dict["length"][e_sil]

    data_all_instances = data
    data = adjust_opt_problem.add_acoustically_relevant_scenarios(
        data, path_h5file, CONTROL_STRATEGY
    )

    velocity_constraint = 1 if MAX_VELOCITY is not None else 0
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
    )

    logging.info("Creating instance...")
    instance = adjust_opt_problem.adjust_to_control_strategy(
        CONTROL_STRATEGY, model=model, data=data
    )
    if fix_silencer:
        for e in instance.E_silencer:
            instance.ind_purchase[e].fix(fixed_sil_dict["purchase"][e])

    instance_all_acoustics = adjust_opt_problem.adjust_to_control_strategy(
        CONTROL_STRATEGY, model=model_all_acoustics, data=data_all_instances
    )

    filename = OUTFOLDER + CONTROL_STRATEGY + "/" + filename_configuration

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
        }
    }

    success = run_initial_solve(
        instance,
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


if __name__ == "__main__":
    main(fix_silencer=True)
