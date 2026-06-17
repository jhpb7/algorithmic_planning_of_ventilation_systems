import logging
import pyomo.environ as pyo
from pyomo2h5 import load_yaml, ConstraintTracker
from src.preplanning.optimise import (
    adjust_opt_problem,
    # optimal_preplanning_detailed_investcosts,
)
import uuid
from src.conceptplanning.preprocessing import optimal_planning
from src.preplanning.optimise.utils import run_initial_solve
from src.preplanning.preprocessing.general_utils import add_duct_zeta_flow_noise_and_dampening_from_h5, add_component_dimensions_from_duct_using_h5


INFILE = "opt_problems/preplanning/GPZ/standard_case.yml"
OUTFOLDER = "new_solutions/GPZ/combined/real/rooms_identical_noise_limits/"
CONTROL_STRATEGY = "ODS-CC"
MAX_VELOCITY = None  # normally: 5
MAX_HEIGHT = -999 #None if None then height=width, if negative then turned off
PLANNING_MODE = "Configuration"

PATH_H5PATH = "new_solutions/GPZ/combined/real/rooms_identical_noise_limits/VAV-VPC/"
H5FILENAME = "03656486-2838-47d7-af7c-81cad251e98f"


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# silence logger of gurobipy
logging.getLogger("gurobipy").propagate = False


def main():
    comment = input("Enter comment for file here:\n")
    tracker = ConstraintTracker()

    logging.info("Loading file...")
    data = load_yaml(INFILE)

    
    full_h5file_path = PATH_H5PATH + H5FILENAME + ".h5"

    if PLANNING_MODE == "Configuration":
        data = add_duct_zeta_flow_noise_and_dampening_from_h5(data, full_h5file_path)
        data = add_component_dimensions_from_duct_using_h5(data, full_h5file_path)

        noise_limit = data["max_sound_power_level"][None]
        noise_limit = 31.4

        data["max_sound_pressure_level_room"] = {v: noise_limit for v in data["V_room"][None]}

        data_all_instances = data
        data = adjust_opt_problem.add_acoustically_relevant_scenarios(data, full_h5file_path, CONTROL_STRATEGY)

    elif PLANNING_MODE == "Topology":
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
        for e in data["E_duct"][None]:
            data["duct_width_max"][e] = fix_duct_data["width"][e]
            data["duct_height_max"][e] = fix_duct_data["height"][e]
            
            data["duct_width_min"][e] = fix_duct_data["width"][e]
            data["duct_height_min"][e] = fix_duct_data["height"][e]


    velocity_constraint = 1 if MAX_VELOCITY is not None else 0
    
    variable_air_volume_flag = not CONTROL_STRATEGY == "CAV"

    
    model = optimal_planning.model(
        planning_mode=PLANNING_MODE,
        duct_model=1,
        fan_model=1,
        branching_constraints=0,
        velocity_constraint=velocity_constraint,
        pressure_target_met=1,
        additional_investment_costs=1,
        reduce_fan_curves=0,
        variable_air_volume=variable_air_volume_flag 
    )


    logging.info("Creating instance...")
    instance = adjust_opt_problem.adjust_to_control_strategy(
        CONTROL_STRATEGY, model=model, data=data
    )

    if PLANNING_MODE == "Topology":
        instance = adjust_opt_problem.adjust_to_duct_constraint(
            instance, MAX_VELOCITY, MAX_HEIGHT
        )
        instance_all_acoustics = None

    elif PLANNING_MODE == "Configuration":
        model_all_acoustics = optimal_planning.model(
            planning_mode=PLANNING_MODE,
            duct_model=1,
            fan_model=1,
            branching_constraints=0,
            velocity_constraint=velocity_constraint,
            pressure_target_met=1,
            additional_investment_costs=1,
            reduce_fan_curves=0,
            all_scenarios_acoustics=1
        )

        logging.info("Creating all acoustics instance...")
        instance_all_acoustics = adjust_opt_problem.adjust_to_control_strategy(
            CONTROL_STRATEGY, model=model_all_acoustics, data=data_all_instances
        )

    filename = str(uuid.uuid4())

    filepath = OUTFOLDER + CONTROL_STRATEGY + "/" + filename

    # max_load_case defines which load case is removed for postprocessing
    # -- this is necessary as the maximum load case could just barely become infeasible
    # when the system is laid out for slightly lower pressure losses.
    if "max_volume_flow_scenario" in data:
        max_load_case = (
            None if CONTROL_STRATEGY in ["CAV", "VAV-CPC"] else data["max_volume_flow_scenario"]
        )
    else:
        max_load_case = None

    solver = pyo.SolverFactory("gurobi", solver_io="python")

    connected_opt_name = H5FILENAME if PLANNING_MODE == "Configuration" else ""
    additional_annotated_dict={"Optimization Type": {
                    "Control Strategy": {"Content": CONTROL_STRATEGY},
                    "Planning mode": {"Content": PLANNING_MODE},
                    "Connected Optimisation": {"Content": connected_opt_name, "Metadata": {"Information": "Filename of connected Optimisation results hdf5 (topology if this file is a configuration or vice versa)"}}}}

    success = run_initial_solve(
        instance, solver, tracker, filepath, PLANNING_MODE, comment, max_load_case, additional_annotated_dict=additional_annotated_dict, acoustics_instance=instance_all_acoustics)

    if success:
        logging.info(f"Saved feasible solution to {filename}.h5")
    else:
        logging.warning(f"Problem infeasible. Results saved to {filename}.h5")


if __name__ == "__main__":
    main()
