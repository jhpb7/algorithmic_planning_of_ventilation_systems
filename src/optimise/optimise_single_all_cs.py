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
OUTFOLDER = "new_solutions/GPZ/preplanning/new_cost_model/"
CONTROL_STRATEGY = "VAV-CPC"
MAX_VELOCITY = 5
MAX_HEIGHT = None
PLANNING_MODE = "topology"

PATH_H5PATH = "new_solutions/off/preplanning/tests_all_floors/ODS-CC/"
H5FILENAME = "20b58c58-3bc9-41c7-ab01-5c8bed18187d.h5"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# silence logger of gurobipy
logging.getLogger("gurobipy").propagate = False


def main(cs, comment):
    tracker = ConstraintTracker()

    logging.info("Loading file...")
    data = load_yaml(INFILE)

    
    full_h5file_path = PATH_H5PATH + H5FILENAME

    if PLANNING_MODE == "configuration":
        data = add_duct_zeta_flow_noise_and_dampening_from_h5(data, full_h5file_path)
        data = add_component_dimensions_from_duct_using_h5(data, full_h5file_path)
        
        data_all_instances = data
        data = adjust_opt_problem.add_acoustically_relevant_scenarios(data, full_h5file_path, cs)

    velocity_constraint = 1 if MAX_VELOCITY is not None else 0
    
    variable_air_volume_flag = not cs == "CAV"

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
        cs, model=model, data=data
    )

    if PLANNING_MODE == "topology":
        instance = adjust_opt_problem.adjust_to_duct_constraint(
            instance, MAX_VELOCITY, MAX_HEIGHT
        )
        instance_all_acoustics = None

    elif PLANNING_MODE == "configuration":
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
            cs, model=model_all_acoustics, data=data_all_instances
        )
    
    filename = str(uuid.uuid4())

    filepath = OUTFOLDER + cs + "/" + filename

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

    additional_annotated_dict={"Optimization Type": {
                    "Control Strategy": {"Content": cs},
                    "Planning mode": {"Content": PLANNING_MODE},
                    "Connected Optimisation": {"Content": H5FILENAME, "Metadata": {"Information": "Filename of connected Optimisation results hdf5 (topology if this file is a configuration or vice versa)"}}}}

    success = run_initial_solve(
        instance, solver, tracker, filepath, PLANNING_MODE, comment, max_load_case, additional_annotated_dict=additional_annotated_dict, acoustics_instance=instance_all_acoustics)

    if success:
        logging.info(f"Saved feasible solution to {filename}.h5")
    else:
        logging.warning(f"Problem infeasible. Results saved to {filename}.h5")


if __name__ == "__main__":
    comment = input("Enter comment for file here:\n")
    css = ["CAV", "VAV-CPC", "VAV-VPC", "DF-CPC", "ONLY-DF", "ODS-CC"]
    for cs in css:
        print(cs)
        main(cs, comment)
