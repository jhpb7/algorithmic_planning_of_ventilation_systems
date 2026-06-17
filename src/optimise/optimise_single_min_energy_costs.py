import logging
import pyomo.environ as pyo

from pyomo2h5 import load_yaml, ConstraintTracker
from src.preplanning.optimise import (
    adjust_opt_problem,
    optimal_preplanning_detailed_investcosts,
)
from src.preplanning.optimise.utils import run_initial_solve


INFILE = "opt_problems/preplanning/off/less_rooms.yml"
OUTFOLDER = "new_solutions/off/preplanning/min_energy_costs_standard_case/"

INFILE = "opt_problems/preplanning/GPZ/standard_case.yml"
OUTFOLDER = "new_solutions/GPZ/preplanning/min_energy_costs_standard_case/"

CONTROL_STRATEGY = "ODS-CC"
MAX_VELOCITY = 5
MAX_HEIGHT = None
MAX_LOAD_CASE_NUMBER = None  # number of the maximum load case (if existing, else None)
print(f"Max load case number {MAX_LOAD_CASE_NUMBER}")

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

    velocity_constraint = 1 if MAX_VELOCITY is not None else 0
    model = optimal_preplanning_detailed_investcosts.model(
        duct_model=1,
        fan_model=1,
        branching_constraints=0,
        velocity_constraint=velocity_constraint,
        pressure_target_met=0,
    )
    print("pressure target is zero!!")

    logging.info("Creating instance...")

    n_scen = data["Scenarios"][None][0]
    # collapse scenarios to 1
    data["scenario"][1] = data["scenario"][n_scen]
    data["Scenarios"][None] = [1]
    data["time_share"] = {1: 1}

    # filter hyperplane sets
    for key in [
        "fan_hyperplanes_underestimation_specific_pre_set",
        "fan_hyperplanes_overestimation_specific_pre_set",
    ]:
        if key in data:
            data[key] = {
                (1, *k[1:]): v for k, v in data[key].items() if k[0] == n_scen
            }

    instance = adjust_opt_problem.adjust_to_control_strategy(
        CONTROL_STRATEGY, model=model, data=data
    )
    instance = adjust_opt_problem.adjust_to_duct_constraint(
        instance, MAX_VELOCITY, MAX_HEIGHT
    )

    for e in instance.E_vfc:
        instance.ind_purchase[e].value = 0
        instance.ind_purchase[e].fixed = True
    
    for e in instance.E_duct:
        instance.duct_width[e].value = instance.duct_width[e].upper
        instance.duct_width[e].fixed = True
        instance.duct_height[e].value = instance.duct_height[e].upper
        instance.duct_height[e].fixed = True

    for e in instance.fan_set:
        instance.fan_ind_purchase.value = 1.0
        instance.fan_ind_purchase.fixed = True


    instance.obj.deactivate()

    @instance.Objective()
    def min_energy_costs(model):
        return model.fan_energy_costs

    outfolder = OUTFOLDER + CONTROL_STRATEGY + "/"

    # max_load_case defines which load case is removed for postprocessing
    # -- this is necessary as the maximum load case could just barely become infeasible
    # when the system is laid out for slightly lower pressure losses.
    max_load_case = (
        None
        if CONTROL_STRATEGY in ["CAV", "VAV-CPC"]
        else MAX_LOAD_CASE_NUMBER  
    )

    solver = pyo.SolverFactory("gurobi", solver_io="python")

    success, filename = run_initial_solve(
        instance, solver, tracker, outfolder, CONTROL_STRATEGY, comment, max_load_case
    )

    if success:
        logging.info(f"Saved feasible solution to {filename}.h5")
    else:
        logging.warning(f"Problem infeasible. Results saved to {filename}.h5")


if __name__ == "__main__":
    main()
