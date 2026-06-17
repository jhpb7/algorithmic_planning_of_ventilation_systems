import logging

import pyomo.environ as pyo
import uuid
from pyomo2h5 import PyomoHDF5Saver
from src.postprocessing.postprocessing import postprocess
from src.debugging.debugging import calculate_IIS


def run_initial_solve(
    instance,
    solver,
    tracker,
    filename,
    planning_mode,
    comment,
    max_load_case,
    additional_annotated_dict=None,
    acoustics_instance=None,
):
    """Run the initial solve and save results or IIS if infeasible."""

    solver.options["LogFile"] = filename + ".log"
    logging.info("Running initial solve. Logging into filename %s", filename)
    results = solver.solve(instance, tee=True, warmstart=False)
    with PyomoHDF5Saver(filename) as saver:
        if results.solver.termination_condition in [
            pyo.TerminationCondition.infeasible,
            pyo.TerminationCondition.infeasibleOrUnbounded,
        ]:
            saver.save_annotated_dict(
                {"Comment": {"Content": comment + ", proven to be infeasible"}}
            )
            # calculate_IIS(instance, filename + "_")
            logging.warning("IIS is not computed.")
            return False
        else:
            logging.info(f"Saving instance to {filename}")

            if additional_annotated_dict:
                saver.save_annotated_dict(additional_annotated_dict)
            saver.save_annotated_dict({"Comment": {"Content": comment}})
            saver.save_instance(
                instance,
                results,
                solver_options=solver.options,
                save_constraint_flag=False,
            )
            logging.info("Postprocessing...")
            saver.save_annotated_dict(
                postprocess(
                    instance,
                    planning_mode=planning_mode,
                    max_load_case=max_load_case,
                    instance_all_acoustics=acoustics_instance,
                ),
                float_precision=4,
            )
            saver.save_tracked_constraints(tracker, "Additional_constraints")

    return True


def run_pareto_loop(
    instance,
    tracker,
    solver,
    bound_expr,
    bound_start,
    stepsize,
    bound_name,
    control_strategy,
    comment,
    outfolder,
    max_load_case,
):
    """Run Pareto optimization loop for a given bound (energy or investment)."""
    while True:
        logging.info(f"Now solving with {bound_name} ub {bound_start}")
        filename = str(uuid.uuid4())
        curr_comment = f"{control_strategy}, {bound_name} ub: {bound_start} {comment}"

        @instance.Constraint()
        def pareto_limit(m):
            return bound_expr(m) <= bound_start

        tracker.add(instance.pareto_limit)
        solver.options["LogFile"] = outfolder + filename + ".log"

        results = solver.solve(instance, tee=True, warmstart=True)
        with PyomoHDF5Saver(outfolder + filename) as saver:
            if results.solver.termination_condition in [
                pyo.TerminationCondition.infeasible,
                pyo.TerminationCondition.infeasibleOrUnbounded,
            ]:
                saver.save_annotated_dict(
                    {"Comment": {"Content": curr_comment + ", proven to be infeasible"}}
                )
                tracker.delete(instance.pareto_limit)
                instance.del_component(pareto_limit)
                break
            else:
                saver.save_annotated_dict({"Comment": {"Content": curr_comment}})
                saver.save_instance(
                    instance,
                    results,
                    solver_options=solver.options,
                    save_constraint_flag=False,
                )
                logging.info("Postprocessing...")
                saver.save_annotated_dict(
                    postprocess(instance, max_load_case=max_load_case),
                    float_precision=4,
                )
                saver.save_tracked_constraints(tracker, "Additional_constraints")

        tracker.delete(instance.pareto_limit)
        instance.del_component(pareto_limit)

        logging.info(f"Done. Saved as {filename}.h5")
        bound_start -= stepsize
