from collections import defaultdict
from typing import List, Tuple, Set, Dict, Optional
import numpy as np

from underestimating_hyperplanes import (
    compute_tangential_planes_equation,
    tangential_planes_to_array,
    tangential_planes_to_dict,
)

Edge = Tuple[str, str]


def build_graph(
    edges: List[Edge],
) -> Tuple[Dict[str, List[str]], Dict[str, int], Set[str]]:
    """
    Builds a directed graph from the list of edges.

    Args:
        edges: A list of (u, v) edges representing directed connections.

    Returns:
        A tuple containing:
        - graph: adjacency list (node -> list of neighbors)
        - in_degree: dictionary mapping node -> in-degree count
        - all_nodes: set of all nodes in the graph
    """
    graph = defaultdict(list)
    in_degree = defaultdict(int)
    all_nodes = set()

    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1
        all_nodes.update([u, v])

    return graph, in_degree, all_nodes


def find_branch_node(edges: List[Edge]) -> str:
    """
    Finds the first branch or leaf node starting from the root.

    Args:
        edges: All edges in the DAG.

    Returns:
        The branch or leaf node reached from the root.

    Raises:
        ValueError: If no root node is found.
    """
    graph, in_degree, all_nodes = build_graph(edges)
    root = next((node for node in all_nodes if in_degree[node] == 0), None)
    if root is None:
        raise ValueError("No root found (graph might have cycles or no entry node)")

    current = root
    while True:
        neighbors = graph.get(current, [])
        if len(neighbors) != 1:
            return current
        current = neighbors[0]


def find_fan_edge(edges: List[Edge], E_fan_station: Set[Edge]) -> Optional[Edge]:
    """
    Finds the first edge from root to a branch that is in E_fan_station.

    Args:
        edges: All edges in the DAG.
        E_fan_station: Subset of edges designated as fan station edges.

    Returns:
        The first matching edge if found, otherwise None.

    Raises:
        ValueError: If no root node is found.
    """
    graph, in_degree, all_nodes = build_graph(edges)
    root = next((node for node in all_nodes if in_degree[node] == 0), None)
    if root is None:
        raise ValueError("No root found (graph might have cycles or no entry node)")

    current = root
    while True:
        neighbors = graph.get(current, [])
        if len(neighbors) != 1:
            return None  # no more path to follow
        next_node = neighbors[0]
        edge = (current, next_node)
        if edge in E_fan_station:
            return edge
        current = next_node


def get_max_volume_flow_in_problem(data):
    max_volume_flow_in_problem = 0
    for s in data["Scenarios"][None]:

        max_q_in_s = np.max(list(data["scenario"][s]["volume_flow"].values()))
        if max_q_in_s > max_volume_flow_in_problem:
            max_volume_flow_in_problem = max_q_in_s
    return max_volume_flow_in_problem


def get_fan_edge_volume_flow(data):
    fan_edge_volume_flow = defaultdict(dict)
    for s in data["Scenarios"][None]:
        fan_edge_volume_flow[s] = {}
        for e in data["E_fan_station"][None]:
            fan_edge_volume_flow[s][e] = data["scenario"][s]["volume_flow"][e]
    return fan_edge_volume_flow


def get_point_distance(value_lst):
    """
    Assuming equidistant grid values in a list, yield the difference between two neighbouring points.

    """
    sorted_lst = sorted(list(set(value_lst)))
    distance = np.diff(sorted_lst)
    if len(distance) == 0:
        return 0
    elif len(distance) == 1:
        return distance
    elif np.any(np.abs(np.diff(distance)) > 1e-3):
        raise ValueError("List values are not equally spaced.")
    return np.min(distance)


# Fan utils


def dimensionalise_coeffs(a, b, d, max_values):
    a = {
        1: a[1] / d**4 / max_values["q"] ** 2 * max_values["dp"],
        2: a[2] / d / max_values["q"] * max_values["dp"],
        3: a[3] * d**2 * max_values["dp"],
    }
    b = {
        1: b[1] / d**4 / max_values["q"] ** 3 * max_values["pel"],
        2: b[2] / d / max_values["q"] ** 2 * max_values["pel"],
        3: b[3] * d**2 / max_values["q"] * max_values["pel"],
        4: b[4] * d**5 * max_values["pel"],
        5: b[5] * max_values["pel"],
    }
    return a, b


def get_max_values(fan_data):
    limits = fan_data["limits"]
    max_values = {
        "q": limits["volume_flow"]["max_dim"],
        "dp": limits["pressure"]["max_dim"],
        "pel": limits["power"]["max_dim"],
    }
    return max_values


def prepare_fan_on_edges(data: Dict):
    result = []
    for (node_from, node_to), values in data["edges"].items():
        for p_d, weight in values.items():
            if not p_d == "total":
                (fan, speed) = p_d
                for i in range(1, weight + 1):
                    result.append([node_from, node_to, fan, speed, i])
    max_num_fans_per_fs = {}
    for edge, values in data["edges"].items():
        max_num_fans_per_fs[edge] = values["total"]
    return {
        "fan_set": {None: result},
        "max_num_fans_per_fan_station": max_num_fans_per_fs,
    }


# %% Prepare ducts


def get_duct_max_dimensions(data):
    duct_max_dimensions = {}
    for i, j in data["duct_width_max"].keys():
        duct_max_dimensions[(i, j)] = (
            data["duct_width_max"][i, j],
            data["duct_height_max"][i, j],
        )
    max_width = max(data["duct_width_max"].values())
    max_height = max(data["duct_height_max"].values())

    return duct_max_dimensions, max_width, max_height


def filter_hyperplanes(data_dict, width_thresh, height_thresh, duct_edge_data):
    filtered_dict = defaultdict(dict)

    for prefix in ["duct_friction", "duct_area2", "inverse"]:
        indices = data_dict[f"{prefix}_hyperplanes_set"]
        width_points = data_dict[f"{prefix}_point_width"]
        if not prefix == "inverse":
            height_points = data_dict[f"{prefix}_point_height"]
            # Filter based on conditions
            valid_indices = [
                idx
                for idx in indices
                if width_points[idx] < width_thresh
                and height_points[idx] < height_thresh
            ]
            filtered_dict[f"{prefix}_slope_height"] = {
                idx: data_dict[f"{prefix}_slope_height"][idx] for idx in valid_indices
            }
        else:
            valid_indices = [idx for idx in indices if width_points[idx] < width_thresh]
        # Rebuild filtered entries
        filtered_dict[f"{prefix}_hyperplanes_set"] = valid_indices
        filtered_dict[f"{prefix}_intercept"] = {
            idx: data_dict[f"{prefix}_intercept"][idx] for idx in valid_indices
        }
        filtered_dict[f"{prefix}_slope_width"] = {
            idx: data_dict[f"{prefix}_slope_width"][idx] for idx in valid_indices
        }

        for (i, j), (max_h, max_w) in duct_edge_data.items():
            if not prefix == "inverse":
                valid_indices = [
                    idx
                    for idx in indices
                    if width_points[idx] < max_w and height_points[idx] < max_h
                ]
                filtered_dict[f"{prefix}_hyperplanes_specific_pre_set"][
                    (i, j)
                ] = valid_indices

    return filtered_dict


# %% Bring load cases in correct format


def prepare_load_case_yaml(data: Dict) -> Dict:
    """
    Prepare load case yaml file

    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    load_case_data = data["scenario"]
    load_case_data = {
        idx
        + 1: {
            "volume_flow": {
                key2: value["mean"] / 3600
                for key2, value in load_case_data[key]["room"].items()
            }
        }
        for idx, key in enumerate(load_case_data.keys())
    }

    time_shares = {
        idx + 1: value for idx, value in enumerate(data["time_share"].values())
    }

    return {
        "scenario": load_case_data,
        "Scenarios": {None: list(load_case_data.keys())},
        "time_share": time_shares,
    }


def compute_planes_nd(f, grads, points: np.ndarray, variables: list[str]) -> dict:
    """Compute tangential planes for an nD function and return YAML dict.

    Args:
        f (callable): Objective function.
        grads (list[callable]): Gradient functions (one per variable).
        points (np.ndarray): Sampling points in nD space.
        variables (list[str]): Names of variables, e.g. ["width", "height"].

    Returns:
        dict: Tangential plane equations in YAML dict format.
    """
    planes = compute_tangential_planes_equation(points, f, grads)
    planes = tangential_planes_to_array(planes)
    return tangential_planes_to_dict(planes, variables)


def compute_planes_1d(f, grad, points: np.ndarray, variables: list[str]) -> dict:
    """Special case for 1D tangential planes.

    Args:
        f (callable): Objective function.
        grad (callable): Gradient function (single variable).
        points (np.ndarray): Sampling points in 1D space.
        variables (list[str]): Single variable name, e.g. ["width"].

    Returns:
        dict: Tangential plane equations in YAML dict format.
    """
    planes = compute_tangential_planes_equation(points, f, grad)
    planes = tangential_planes_to_array(planes)
    return tangential_planes_to_dict(planes, variables)


def plane_dict_from_array(index, data: dict, prefix: str, variables: list[str]) -> dict:
    """Convert tangential plane data into a structured dict with prefixed keys.

    Args:
        index (Iterable[int]): Indices of hyperplanes.
        data (dict): Plane data containing intercepts, gradients, and points.
        prefix (str): Prefix string (e.g. "duct_friction").
        variables (list[str]): Variable names (e.g. ["width", "height"]).

    Returns:
        dict: Dictionary with standardized prefixed keys for YAML export.
    """
    out = {f"{prefix}_hyperplanes_set": index}
    out[f"{prefix}_intercept"] = dict(zip(index, data["intercept"]))

    for var in variables:
        out[f"{prefix}_slope_{var}"] = dict(zip(index, data[f"grad_{var}"]))
        out[f"{prefix}_point_{var}"] = dict(zip(index, data[f"point_{var}"]))

    return out
