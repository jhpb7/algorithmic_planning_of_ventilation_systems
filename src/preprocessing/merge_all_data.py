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


def main():
    OUT_FOLDER = "opt_problems/preplanning/off/"
    OUT_FILENAME = "standard_case_k-medoid"
    DUPLICATE_FLOORS = (
        None  # True  # if True multiple identical copies of the same floor are added
    )
    N_FLOORS = 16

    NETWORK_DATA_FILE = "data/network_data/OFF.yml"
    DUCT_DATA_FILE = "data/duct_data/duct_hyperplanes_OFF.yml"  # OFF
    FAN_DATA_FILE = "data/fan_data/fan_power_loss_hyperplanes_OFF.yml"
    FANS_ON_EDGES_FILE = "data/network_data/fans_on_edges_OFF.yml"
    FIXED_DATA_FOLDER = "data/fixed_data/OFF/"

    FLOOR_ENTRANCE_NODE = "A0"  # hof: "zS"  # off: "A0"

    data = load_yaml(NETWORK_DATA_FILE)
    data = compute_fixed_zeta_from_yaml(data, FIXED_DATA_FOLDER)

    scenario_data_file = (
        "data/load_case_data/processed_OFF_load_cases_k_medoid_generated.yml"
    )
    load_case_data = load_yaml(scenario_data_file)

    fans_on_edges = load_yaml(FANS_ON_EDGES_FILE)

    data.update(prepare_load_case_yaml(load_case_data))

    data = propagate_volume_flows(data)
    max_volume_flow_in_problem = get_max_volume_flow_in_problem(data)
    max_pressure_in_problem = data["max_pressure"][None]

    duct_data = load_yaml(DUCT_DATA_FILE)

    duct_edge_max_dimensions, max_width, max_height = get_duct_max_dimensions(data)

    data.update(
        filter_hyperplanes(duct_data, max_width, max_height, duct_edge_max_dimensions)
    )

    fan_data = load_yaml(FAN_DATA_FILE)

    data.update(prepare_fan_on_edges(fans_on_edges))

    if DUPLICATE_FLOORS:
        print(f"Scaling to {N_FLOORS} floors")
        data = duplicate_floors_add_connectors(data, N_FLOORS, FLOOR_ENTRANCE_NODE)
    data = compute_flow_noise_and_dampening_dicts_from_yaml(data, FIXED_DATA_FOLDER)
    print("WARNING -- NO CROSSTALKING!")
    # data = compute_crosstalk_dampening(data, FIXED_DATA_FOLDER)

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
