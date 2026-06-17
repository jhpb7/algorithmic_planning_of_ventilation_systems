import logging
import os
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


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# silence logger of gurobipy
logging.getLogger("gurobipy").propagate = False


def main(noise_limit, control_strategy, full_h5file_path, comment):

    tracker = ConstraintTracker()

    logging.info("Loading file...")
    data = load_yaml(INFILE)

    data = add_duct_zeta_flow_noise_and_dampening_from_h5(data, full_h5file_path)
    data = add_component_dimensions_from_duct_using_h5(data, full_h5file_path)

    data["max_sound_pressure_level_room"] = {v: noise_limit for v in data["V_room"][None]}

    data_all_instances = data
    data = adjust_opt_problem.add_acoustically_relevant_scenarios(data, full_h5file_path, control_strategy)


    variable_air_volume_flag = not control_strategy == "CAV"

    model = optimal_planning.model(
        planning_mode=PLANNING_MODE,
        duct_model=1,
        fan_model=1,
        branching_constraints=0,
        velocity_constraint=0,
        pressure_target_met=1,
        additional_investment_costs=1,
        reduce_fan_curves=0,
        variable_air_volume=variable_air_volume_flag 
    )


    logging.info("Creating instance...")
    instance = adjust_opt_problem.adjust_to_control_strategy(
        control_strategy, model=model, data=data
    )

    model_all_acoustics = optimal_planning.model(
        planning_mode=PLANNING_MODE,
        duct_model=1,
        fan_model=1,
        branching_constraints=0,
        velocity_constraint=0,
        pressure_target_met=1,
        additional_investment_costs=1,
        reduce_fan_curves=0,
        all_scenarios_acoustics=1
    )

    logging.info("Creating all acoustics instance...")
    instance_all_acoustics = adjust_opt_problem.adjust_to_control_strategy(
        control_strategy, model=model_all_acoustics, data=data_all_instances
    )

    filename = str(uuid.uuid4())

    filepath = OUTFOLDER + control_strategy + "/" + filename

    # max_load_case defines which load case is removed for postprocessing
    # -- this is necessary as the maximum load case could just barely become infeasible
    # when the system is laid out for slightly lower pressure losses.
    if "max_volume_flow_scenario" in data:
        max_load_case = (
            None if control_strategy in ["CAV", "VAV-CPC"] else data["max_volume_flow_scenario"]
        )
    else:
        max_load_case = None

    solver = pyo.SolverFactory("gurobi", solver_io="python")

    connected_opt_name = full_h5file_path.split("/")[-1][:-3]
    additional_annotated_dict={"Optimization Type": {
                    "Control Strategy": {"Content": control_strategy},
                    "Planning mode": {"Content": PLANNING_MODE},
                    "Pareto Limit": {"Content": f"noise limit: {noise_limit} dB"},
                    "Connected Optimisation": {"Content": connected_opt_name, "Metadata": {"Information": "Filename of connected topology Optimisation results hdf5"}}}}

    success = run_initial_solve(
        instance, solver, tracker, filepath, PLANNING_MODE, comment, max_load_case, additional_annotated_dict=additional_annotated_dict, acoustics_instance=instance_all_acoustics)

    if success:
        logging.info(f"Saved feasible solution to {filename}.h5")
    else:
        logging.warning(f"Problem infeasible. Results saved to {filename}.h5")
        # raise ValueError("Infeasible solution.")
    return success

def create_dir_if_not_existing(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory {path} created successfully!")
    else:
        print(f"Directory {path} already exists!")



INFILE = "opt_problems/preplanning/off/standard_case.yml"
OUTFOLDER = "results/off/combined/max_velocity_5_10ms_max_height_04m/rooms_identical_noise_limits_pareto/"
PLANNING_MODE = "Configuration"

if __name__ == "__main__":

    noise_limits = [37.5, 35, 30, 25]

    comment = input("Enter comment for file here:\n")

    css = ["ODS-CC"]
    previous_files_folder = "results/off/combined/max_velocity_5_10ms_max_height_04m/standard_case/"
    filename_topo = {
        "CAV": "fff81d9b-9e3b-4309-a7f7-ce55ab08dc99.h5",
        "VAV-CPC": "773fa1ec-3788-47e2-bc13-f1e28fdef057.h5",
        "VAV-VPC": "2993add0-6ccf-46e9-ba5a-f27e5c1e4dc4.h5",
        "DF-CPC": "9eea85fb-8a9b-4958-9e4e-82c91b605d29.h5",
        "ONLY-DF": "71fbfa02-8e78-4e70-9a3d-cfe5ef8017e6.h5",
        "ODS-CC": "126cb0ad-e408-4ad7-a398-6fff40ef4cae.h5"}

    for idx, cs in enumerate(css):

        for noise_limit in noise_limits:
            create_dir_if_not_existing(
                OUTFOLDER + cs,
            )
            print(f"\nControl strategy is {cs} and Noise limit is {noise_limit} dB everywhere.\n")
            success = main(noise_limit, cs, previous_files_folder + cs + "/" + filename_topo[cs], comment)
            if not success:
                break