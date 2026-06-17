import logging
import os
import pyomo.environ as pyo
import uuid
from pyomo2h5 import load_yaml, ConstraintTracker, PyomoHDF5Saver
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


def main(cs, comment, topo_opt_flag=1, config_opt_flag=1, filename_topo_previous=None):
    tracker = ConstraintTracker()

    if not topo_opt_flag:
        if not filename_topo_previous:
            raise ValueError(
                "Topology is not optimised but also no previously optimised file is given."
            )

    if config_opt_flag:
        filename_configuration = str(uuid.uuid4())
    else:
        filename_configuration = ""

    success = True

    velocity_constraint = 1  # if MAX_VELOCITY is not None else 0
    logging.warning("Velocity constraint is always on!")
    variable_air_volume_flag = not cs == "CAV"

    # max_load_case defines which load case is removed for postprocessing
    # -- this is necessary as the maximum load case could just barely become infeasible
    # when the system is laid out for slightly lower pressure losses.

    solver = pyo.SolverFactory("gurobi", solver_io="python")

    if topo_opt_flag:

        filename_topology = str(uuid.uuid4())

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
            cs, model=model, data=data
        )
        instance_topo = adjust_opt_problem.adjust_to_duct_constraint(
            instance_topo, MAX_VELOCITY, MAX_HEIGHT
        )

        # input("FORCED TO BUY ONE DISTRIBUTED")
        # @instance_topo.Constraint(instance_topo.Scenarios)
        # def forced_to_buy_n_fans(m, s):
        #     return sum(m.scenario[s].ind_active[e] for e in m.E_fan_station - m.E_fan_station_central) == 1
        # instance_topo.min_lcc.deactivate()


        # @instance_topo.Objective(sense=pyo.minimize)
        # def total_duct_costs(m):
        #     return m.total_duct_costs + 1e-9*m.total_invest_costs + 1e-9*m.fan_energy_costs

        # @instance_topo.Constraint(model.E_duct)
        # def height_equal_to_width(m,i,j):
        #     return m.duct_height[i,j] == m.duct_width[i,j]
        # @instance_topo.Objective(sense=pyo.minimize)
        # def min_duct_volume(m):
        #     return sum(m.duct_height[i,j] * m.duct_width[i,j] * m.duct_length[i,j] for (i,j) in m.E_duct) + 1e-9* m.fan_energy_costs + 1e-9*m.total_invest_costs

        filename = OUTFOLDER + cs + "/" + filename_topology

        if "max_volume_flow_scenario" in data:
            max_load_case = (
                None if cs in ["CAV", "VAV-CPC"] else data["max_volume_flow_scenario"]
            )
        else:
            max_load_case = None

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
    if topo_opt_flag:
        path_h5file = filename + ".h5"
    else:
        filename_topology = filename_topo_previous.split("/")[-1][:-3]
        path_h5file = filename_topo_previous
        if FIX_ALL_FIRST_STAGE_DECISIONS:
            with PyomoHDF5Saver(path_h5file, mode="r") as saver:
                instance_topo = saver.load_instance()

    if config_opt_flag and success:

        logging.info("Loading file...")

        data = load_yaml(INFILE)
        data = add_duct_zeta_flow_noise_and_dampening_from_h5(data, path_h5file)
        data = add_component_dimensions_from_duct_using_h5(data, path_h5file)

        data_all_instances = data
        data = adjust_opt_problem.add_acoustically_relevant_scenarios(
            data, path_h5file, cs
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
            all_scenarios_acoustics=0,
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
            cs, model=model, data=data
        )

        # @instance_config.Constraint(instance_config.max_noise_scenarios, instance_config.V_room)
        # def limit_noise(m, s, v):
        #     return m.scenario[s].sound_pressure_level_room[v] <= 25
        # input("NOISE LIMIT IS ACTIVE HERE!! at 25 dB")

        if FIX_ALL_FIRST_STAGE_DECISIONS:
            for e in instance_config.E_fan_station | instance_config.E_vfc:
                instance_config.ind_purchase[e].fix(instance_topo.ind_purchase[e])
            for f in instance_config.fan_set:
                instance_config.fan_ind_purchase[f].fix(
                    instance_topo.fan_ind_purchase[f]
                )

        instance_all_acoustics = adjust_opt_problem.adjust_to_control_strategy(
            cs, model=model_all_acoustics, data=data_all_instances
        )

        filename = OUTFOLDER + cs + "/" + filename_configuration

        if "max_volume_flow_scenario" in data:
            max_load_case = (
                None if cs in ["CAV", "VAV-CPC"] else data["max_volume_flow_scenario"]
            )
        else:
            max_load_case = None

        additional_annotated_dict = {
            "Optimization Type": {
                "Control Strategy": {"Content": cs},
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



INFILE = "opt_problems/preplanning/off/standard_case_one_room_with_bigger_pressure.yml"

# input(f"STARTING WITH INPUT FILE {INFILE}")
MAX_VELOCITY = None  # 5
MAX_VELOCITY_VERTICAL = 10
MAX_VELOCITY_HORIZONTAL = 5
MAX_HEIGHT = 0.4  # -999 #if None then height=width, if negative then turned off
CAPEX_REDUCTION = 1  # None
FIX_ALL_FIRST_STAGE_DECISIONS = True

OUTFOLDER = f"results/off/combined/one_room_with_bigger_pressure/max_velocity_5_10ms_max_height_04m/"


if __name__ == "__main__":
    comment = input("Enter comment for file here:\n")
    css = ["CAV", "VAV-CPC", "VAV-VPC", "DF-CPC", "ONLY-DF", "ODS-CC"]

    css = ["ODS-CC"]

    # previous_files_folder = (
    #     "results/off/combined/max_velocity_5_10ms_max_height_04m/standard_case/"
    # )
    # filename_topo_previous = {"CAV": "fff81d9b-9e3b-4309-a7f7-ce55ab08dc99.h5",
    #                           "VAV-CPC": "773fa1ec-3788-47e2-bc13-f1e28fdef057.h5",
    #                           "VAV-VPC": "2993add0-6ccf-46e9-ba5a-f27e5c1e4dc4.h5",
    #                           "DF-CPC": "9eea85fb-8a9b-4958-9e4e-82c91b605d29.h5",
    #                           "ONLY-DF": "71fbfa02-8e78-4e70-9a3d-cfe5ef8017e6.h5",
    #                           "ODS-CC": "126cb0ad-e408-4ad7-a398-6fff40ef4cae.h5"}

    topo_opt_flag = 1
    config_opt_flag = 1

    for idx, cs in enumerate(css):

        create_dir_if_not_existing(
            OUTFOLDER + cs,
        )
        print(cs)
        main(
            cs,
            comment,
            topo_opt_flag=topo_opt_flag,
            config_opt_flag=config_opt_flag)
        #     filename_topo_previous=previous_files_folder + cs + "/" + filename_topo_previous[cs],
        # )

        #
