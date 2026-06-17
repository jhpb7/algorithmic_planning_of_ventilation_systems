import logging
import numpy as np
import pandas as pd

import h5py
import time

import pyomo.environ as pyo

from pyomo2h5 import PyomoHDF5Saver, load_yaml
from src.preprocessing.propagate_volume_flows import propagate_volume_flows
from src.pyomo_models import optimal_planning
from src.optimise import (
    adjust_opt_problem,
)
from src.preprocessing.general_utils import (
    find_branch_node,
    add_duct_zeta_flow_noise_and_dampening_from_h5,
    add_component_dimensions_from_duct_using_h5,
    compute_flow_noise_and_dampening_dicts_from_yaml,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# silence logger of gurobipy
logging.getLogger("gurobipy").propagate = False


def fix_instance_for_run(instance, optimised_instance, planning_mode):

    for e in optimised_instance.E_fan_station | optimised_instance.E_vfc:
        instance.ind_purchase[e].fix(optimised_instance.ind_purchase[e])

    for fan in instance.fan_ind_purchase:
        instance.fan_ind_purchase[fan].fix(optimised_instance.fan_ind_purchase[fan])

    if planning_mode == "Topology":
        for e in optimised_instance.E_duct:
            instance.duct_height[e].fix(optimised_instance.duct_height[e])
            instance.duct_width[e].fix(optimised_instance.duct_width[e])

    elif planning_mode == "Configuration":
        for e in optimised_instance.E_silencer:
            instance.ind_purchase[e].fix(optimised_instance.ind_purchase[e])

            instance.silencer_length[e].fix(optimised_instance.silencer_length[e])
            instance.number_of_splitters[e].fix(
                optimised_instance.number_of_splitters[e]
            )
    else:
        raise KeyError(
            f"Planning mode should either be 'Topology' or 'Configuration' - was {planning_mode}"
        )

    instance.min_lcc.deactivate()

    if planning_mode == "Configuration":

        instance.binary_switch_if_noise_exceeds_limit = pyo.Var(within=pyo.Binary)

        instance.sound_pressure_level_exceedance = pyo.Var(within=pyo.NonNegativeReals)

        instance.scenario[1].limit_sound_pressure_level_in_room.deactivate()

        @instance.Constraint()
        def exceedance_binary_costs(m):
            return (
                m.sound_pressure_level_exceedance
                <= m.max_sound_power_level * m.binary_switch_if_noise_exceeds_limit
            )

        @instance.Constraint(instance.V_room)
        def def_sound_pressure_level_exceedance(m, v):
            return (
                m.sound_pressure_level_exceedance
                >= m.scenario[1].sound_pressure_level_room[v]
                - m.max_sound_pressure_level_room[v]
            )

        @instance.Objective(sense=pyo.minimize)
        def min_energy_consumption_allow_acoustic_exceedance(instance):
            return (
                instance.fan_power_consumption
                + 1e-6
                * sum(
                    instance.scenario[1].sound_pressure_level_room[v]
                    for v in instance.V_room
                )
                + 1e-4 * instance.sound_pressure_level_exceedance
                + 1e7 * instance.binary_switch_if_noise_exceeds_limit
            )

    else:

        @instance.Objective(sense=pyo.minimize)
        def min_energy_consumption(instance):
            return instance.fan_power_consumption

    return instance


def fix_data_for_run(data, q_vals):
    data["Scenarios"] = {None: [1]}
    data["scenario"] = {1: {"volume_flow": {}}}
    data["scenario"][1]["volume_flow"] = {
        key: val["mean"] / 3600 for key, val in q_vals.items()
    }
    data["time_share"] = {1: 1}
    data = propagate_volume_flows(data)

    return data


def build_df(scenario_dict, power_dict, power_lb_dict, sound_pressure_gaps):
    """
    scenario_dict: like {'scenario': {'0': {'room': {'1-1~4': {'mean': 729.0}, ...}}, '1': {...}, ...}}
                       or just {'0': {...}, ...} (works either way)
    power_dict:     like {0: 1317.31, 1: 1317.83, ...}
    """

    power_col = "power consumption in W"
    sound_pressure_gap_col = "sound pressure gap in dB"

    # allow both shapes: {'scenario': {...}} or {...}
    scenarios = scenario_dict.get("scenario", scenario_dict)

    # normalize keys to int
    scenarios = {int(k): v for k, v in scenarios.items()}
    power = {int(k): v for k, v in power_dict.items()}
    sound_pressure_gaps = {int(k): v for k, v in sound_pressure_gaps.items()}

    time_share = {int(k): v for k, v in scenario_dict["time_share"].items()}

    # collect all room names across scenarios
    all_rooms = sorted(
        {room_name for s in scenarios.values() for room_name in s["room"].keys()}
    )

    # build a row per scenario with room -> mean
    rows = {}
    for sid, s in scenarios.items():
        row = {room: s["room"].get(room, {}).get("mean", pd.NA) for room in all_rooms}
        rows[sid] = row

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "scenario"

    # attach energy cost
    df[power_col] = pd.Series(power)

    df[sound_pressure_gap_col] = pd.Series(sound_pressure_gaps)

    df["time share"] = pd.Series(time_share)

    if power_lb_dict:
        power_lb_col = "power consumption lower bound in W"
        power_lb = {int(k): v for k, v in power_lb_dict.items()}
        df[power_lb_col] = pd.Series(power_lb)

        # nice ordering: rooms first, then cost
        room_cols = [c for c in df.columns if c != power_col and c != power_lb_col]
        df = df[room_cols + [power_col, power_lb_col]]
    else:
        room_cols = [c for c in df.columns if c != power_col]
        df = df[room_cols + [power_col]]
    return df.sort_index()


def main(
    cs,
    n_points,
    original_file_path,
    input_data_yaml,
    load_case_yaml,
    planning_mode="Topology",
    path_topo_h5file=None,
    noise_limit=None,
):

    solver = pyo.SolverFactory("gurobi_direct", solver_io="python")

    variable_air_volume_flag = not cs == "CAV"

    model = optimal_planning.model(
        planning_mode=planning_mode,
        duct_model=1,
        fan_model=1,
        branching_constraints=0,
        velocity_constraint=0,
        pressure_target_met=1,
        additional_investment_costs=1,
        all_scenarios_acoustics=1,
        reduce_fan_curves=0,
        variable_air_volume=variable_air_volume_flag,
    )

    # solver.options["MIPGap"] = 0.1
    solver.options["TimeLimit"] = 3 * 60  # *30
    logging.warning("Solver has time limit!")

    logging.info("Load files...")

    load_case_dict = load_yaml(load_case_yaml)

    with PyomoHDF5Saver(original_file_path, mode="r") as saver:
        optimised_instance = saver.load_instance()

    data = load_yaml(input_data_yaml)

    # if not noise_limit:
    #     raise ValueError("NOISE LIMIT NOT SET!")
    # print("MAX NOISE LEVEL SET TO 31.4!!!")
    # data["max_sound_pressure_level_room"] = {v: noise_limit for v in data["V_room"][None]}

    logging.info("Files loaded.")

    power_consumption, power_consumption_lower_bound, sound_pressure_limit_gap = (
        {},
        {},
        {},
    )

    t0 = time.time()

    for key, values in load_case_dict["scenario"].items():
        logging.info("Now running load case number %s", key)
        q_vals = values["room"]
        data = fix_data_for_run(data, q_vals)

        if planning_mode == "Configuration":
            if not path_topo_h5file:
                raise ValueError("No Path to Topology optimised hdf5 file is provided.")
            data = add_duct_zeta_flow_noise_and_dampening_from_h5(
                data, path_topo_h5file
            )
            data = add_component_dimensions_from_duct_using_h5(data, path_topo_h5file)
            data = compute_flow_noise_and_dampening_dicts_from_yaml(
                data, FIXED_DATA_FOLDER
            )

        instance = adjust_opt_problem.adjust_to_control_strategy(
            cs, model=model, data=data
        )

        if cs == "VAV-CPC":

            instance.del_component(instance.const_prepressure)

            @instance.Constraint()
            def const_prepressure(m):
                branch_node = find_branch_node(data["E"][None])
                return (
                    m.scenario[1].pressure[branch_node]
                    == optimised_instance.scenario[1].pressure[branch_node].value
                )

        instance = fix_instance_for_run(instance, optimised_instance, planning_mode)

        results = solver.solve(instance, tee=False, warmstart=True)

        if results.solver.termination_condition in [
            pyo.TerminationCondition.infeasible,
            pyo.TerminationCondition.infeasibleOrUnbounded,
        ]:
            logging.warning("Problem infeasible at %s", key)
            power_consumption[key] = -99
            power_consumption_lower_bound[key] = -999
            sound_pressure_limit_gap[key] = -999
            # calculate_IIS(instance, "key")
        else:
            sound_pressure_limit_gap[key] = max(
                instance.scenario[1].sound_pressure_level_room[v].value
                - instance.max_sound_pressure_level_room[v]
                for v in instance.V_room
            )

            power_consumption[key] = instance.fan_power_consumption.expr()
            power_consumption_lower_bound[key] = results.problem.lower_bound
        if (int(key) + 1) % 5 == 0:
            t1 = time.time()
            logging.info("Load case %s after %.2f min", key, (t1 - t0) / 60)
        if (int(key) + 1) % 20 == 0:
            percentage_feasible = (
                sum(1 for value in power_consumption.values() if value > 0)
                / len(power_consumption)
                * 100
            )
            print(f"Percentage of feasible load cases {percentage_feasible:.0f}")

    df = build_df(
        load_case_dict,
        power_consumption,
        power_consumption_lower_bound,
        sound_pressure_limit_gap,
    )

    # Convert DataFrame to HDF5 compound-compatible structured array
    compound_data = df.to_records(index=False)

    with h5py.File(original_file_path + ".h5", "a") as f:
        postprocess = f["Postprocessing"]

        try:
            mlcc = postprocess.create_group("Multi Load Case Comparison")
        except ValueError:
            mlcc = postprocess["Multi Load Case Comparison"]
            print("Does not create group again as it exists already.")
        dset = mlcc.create_dataset(
            f"Points {n_points}", data=compound_data, compression="gzip"
        )

        # columns + index as attributes
        # dset.attrs["column info"] = (
        #     "The first column names represent the room names, followed by the time share and the power consumption (in W)"
        # )
        # dset.attrs["columns"] = list(df.columns.astype(str))
        dset.attrs["Number and dimension of Sobol points"] = (
            f"{n_points} points with dimension (14 hours x R rooms)"
        )


if __name__ == "__main__":

    css = ["VAV-VPC", "DF-CPC", "VAV-VPC"]
    path_to_cs_files = "results/off/combined/max_velocity_5_10ms_max_height_04m/Monte_Carlo_Validation_acoustics/"

    original_files = {
        "k_medoid_result_a295e75f-6bdd-456c-9f16-5276f45a7fd6": "VAV-VPC",
        "7abba0cd-7f74-40c8-a52e-76f4e4a3eed0": "VAV-CPC",
        "45ee17f2-313c-4538-b7b0-fc10666fc8cc": "DF-CPC",
        # "b83b2fcd-0300-4104-88aa-4fc7a2300218": "VAV-VPC",
        # "946b9243-f772-4864-88fe-46baa631a7e3": "VAV-VPC",
        # "DF-CPC": "694f567a-21f2-421b-9f2a-73a320bd0834",
    }

    input("DID YOU CHECK THE NOISE LIMITS?!")

    input_data_yaml = "opt_problems/preplanning/off/standard_case.yml"

    folder_topo_path = (
        "results/off/combined/max_velocity_5_10ms_max_height_04m/standard_case/"
    )

    path_topo_h5files = {
        "VAV-VPC": "k_medoid_result_655aefff-118f-499f-beeb-7582aba8a19e.h5",
        "VAV-CPC": "773fa1ec-3788-47e2-bc13-f1e28fdef057.h5",
        "DF-CPC": "9eea85fb-8a9b-4958-9e4e-82c91b605d29.h5",
    }

    noise_limit = None

    input("Did you choose the right silencer width?")

    input("Did you take a new name for the MC in the h5 file?")

    FIXED_DATA_FOLDER = "data/fixed_data/OFF/"

    n_points = 128

    load_case_yaml = f"data/load_case_data/processed_OFF_distribution_sampled_load_cases_sobol_{n_points}.yml"
    # load_case_yaml = "data/load_case_data/processed_GPZ_load_cases.yml"

    for idx, (original_file, cs) in enumerate(original_files.items()):

        of_path = path_to_cs_files + cs + "/" + original_file

        path_topo_h5file = folder_topo_path + cs + "/" + path_topo_h5files[cs]
        logging.info(
            "\n\n Now computing Monte Carlo-style load cases for control strategy %s\n\n",
            cs,
        )
        logging.info("file path: %s", of_path)
        main(
            cs,
            n_points,
            of_path,
            input_data_yaml,
            load_case_yaml,
            planning_mode="Configuration",
            path_topo_h5file=path_topo_h5file,
            noise_limit=noise_limit,
        )
