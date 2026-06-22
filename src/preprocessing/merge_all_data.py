from pyomo2h5 import load_yaml
from pyomo2h5.yaml_handler import construct_yaml, convert_numpy_to_native


from src.preprocessing.propagate_volume_flows import propagate_volume_flows
from src.preprocessing.general_utils import (
    prepare_load_case_yaml,
    get_max_volume_flow_in_problem,
    get_fan_edge_volume_flow,
    compute_fixed_zeta_from_yaml,
    compute_flow_noise_and_dampening_dicts_from_yaml,
    compute_crosstalk_dampening,
)

from src.preprocessing.domain_utils import (
    prepare_fan_yaml,
    prepare_fan_on_edges,
    filter_hyperplanes,
    get_duct_max_dimensions,
)
from src.preprocessing.duplicate_floors_utils import (
    duplicate_floors_add_connectors,
)


BUILDING_NAME = "OFF"

OUT_FOLDER = f"yaml_opt_input_files/{BUILDING_NAME}/"
OUT_FILENAME = "standard_case"


def main():

    if BUILDING_NAME == "GPZ":
        n_floors = 1
        duplicate_floors = False

    elif BUILDING_NAME == "OFF":
        n_floors = 16
        duplicate_floors = True
        floor_entrance_node = "A0"
    else:
        raise ValueError(
            f"Building name is {BUILDING_NAME} but should be either 'GPZ' or 'OFF'"
        )

    network_data_file = f"data/network_data/{BUILDING_NAME}.yml"
    duct_data_file = f"data/duct_data/duct_hyperplanes_{BUILDING_NAME}.yml"
    fan_data_file = f"data/fan_data/fan_power_loss_hyperplanes_{BUILDING_NAME}.yml"
    fans_on_edges_file = f"data/network_data/fans_on_edges_{BUILDING_NAME}.yml"
    fixed_data_folder = f"data/fixed_data/{BUILDING_NAME}/"
    scenario_data_file = f"data/load_case_data/{BUILDING_NAME}_load_cases.yml"

    data = load_yaml(network_data_file)
    data = compute_fixed_zeta_from_yaml(data, fixed_data_folder)

    load_case_data = load_yaml(scenario_data_file)

    fans_on_edges = load_yaml(fans_on_edges_file)

    data.update(prepare_load_case_yaml(load_case_data))

    data = propagate_volume_flows(data)
    max_volume_flow_in_problem = get_max_volume_flow_in_problem(data)
    max_pressure_in_problem = data["max_pressure"][None]

    duct_data = load_yaml(duct_data_file)

    duct_edge_max_dimensions, max_width, max_height = get_duct_max_dimensions(data)

    data.update(
        filter_hyperplanes(duct_data, max_width, max_height, duct_edge_max_dimensions)
    )

    fan_data = load_yaml(fan_data_file)

    data.update(prepare_fan_on_edges(fans_on_edges))

    if duplicate_floors:
        print(f"Scaling to {n_floors} floors")
        data = duplicate_floors_add_connectors(data, n_floors, floor_entrance_node)
    data = compute_flow_noise_and_dampening_dicts_from_yaml(data, fixed_data_folder)

    if BUILDING_NAME == "OFF":
        data = compute_crosstalk_dampening(data, fixed_data_folder)

    max_volume_flow_in_problem = get_max_volume_flow_in_problem(data)
    max_pressure_in_problem = data["max_pressure"][None]
    data.update(
        prepare_fan_yaml(
            fan_data,
            fan_set=data["fan_set"][None],
            fan_edge_volume_flow=get_fan_edge_volume_flow(data),
            max_pressure_loss_in_problem=max_pressure_in_problem,
            max_volume_flow_in_problem=max_volume_flow_in_problem,
        )
    )

    yaml = construct_yaml()
    data = convert_numpy_to_native(data)
    yaml.indent(mapping=2, sequence=4, offset=2)

    with open(OUT_FOLDER + OUT_FILENAME + ".yml", "w") as f:
        yaml.dump(data, f)


if __name__ == "__main__":
    main()
