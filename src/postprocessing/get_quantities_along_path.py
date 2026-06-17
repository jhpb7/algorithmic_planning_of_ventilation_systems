from typing import List, Tuple, Dict
from collections import defaultdict, deque
import h5py
import ast
import numpy as np
from src.preprocessing.general_utils import parse_key
from functools import lru_cache


def decode_h5_tuple_str_into_tuple(byte_str: bytes) -> Tuple[str, ...]:
    """Decodes a byte string representing a tuple into an actual Python tuple of strings."""
    decoded = byte_str.decode("utf-8").strip("()")
    return tuple(item.strip() for item in decoded.split(","))


def decode_and_eval(h5_array) -> List[Tuple[str, str]]:
    """Decodes and evaluates HDF5 byte string array into list of tuples."""
    return [ast.literal_eval(x.decode()) for x in h5_array[:]]


def decode_list(h5_array) -> List[str]:
    """Decodes HDF5 byte string array into list of strings."""
    return [x.decode() for x in h5_array[:]]


def decode_scenarios(scenarios) -> List[int]:
    """Decodes scenario identifiers into integers."""
    return [int(s) for s in scenarios]


def build_graph(edges: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    """Builds a directed graph as adjacency list from a list of edges."""
    graph = defaultdict(list)
    for a, b in edges:
        graph[a].append(b)
    return graph


def bfs_paths(
    graph: Dict[str, List[str]], root: str, targets: List[str]
) -> Dict[str, List[str]]:
    """Finds shortest paths from the root to each target node using BFS."""
    paths = {}
    visited = set()
    queue = deque([[root]])

    while queue and set(paths.keys()) != set(targets):
        path = queue.popleft()
        node = path[-1]
        if node in visited:
            continue
        visited.add(node)

        if node in targets and node not in paths:
            paths[node] = path

        for neighbor in graph[node]:
            if neighbor not in visited:
                queue.append(path + [neighbor])

    return paths


def removed_fixed_edges(E_fixed_and_duct):
    """
    Remove a fixed edge (u, v) if v is the start of another fixed edge.
    """
    fixed_set = set(E_fixed_and_duct)
    fixed_starts = {u for u, v in fixed_set}

    return {(u, v) for (u, v) in fixed_set if v in fixed_starts}


def contract_tree_edges(E, edges_to_remove):
    """
    Contract/suppress nodes whose incoming edge is removed.

    Example:
        E = [(1, 2), (2, 3)]
        edges_to_remove = {(1, 2)}
        -> [(1, 3)]

    More generally:
        1 -> 2 -> 3 -> 4
        remove (1,2) and (2,3)
        -> (1,4)
    """
    children = defaultdict(list)
    all_nodes = set()

    for u, v in E:
        children[u].append(v)
        all_nodes.add(u)
        all_nodes.add(v)

    # A node is suppressed if the edge into it is removed
    suppressed_nodes = {v for _, v in edges_to_remove}

    @lru_cache(None)
    def first_visible_descendants(node):
        """
        Return the first descendants below 'node' that are NOT suppressed.
        If a child is suppressed too, recurse until we hit visible nodes.
        """
        out = []
        for child in children.get(node, []):
            if child in suppressed_nodes:
                out.extend(first_visible_descendants(child))
            else:
                out.append(child)
        return tuple(out)

    new_edges = set()

    for u in all_nodes:
        if u in suppressed_nodes:
            continue

        for v in children.get(u, []):
            if v in suppressed_nodes:
                for w in first_visible_descendants(v):
                    new_edges.add((u, w))
            else:
                new_edges.add((u, v))

    return sorted(new_edges)


def build_E_clean_with_rewiring(E, E_fixed_and_duct):
    edges_to_remove = removed_fixed_edges(E_fixed_and_duct)
    E_clean = contract_tree_edges(E, edges_to_remove)
    return E_clean


def extract_quantity_data(
    h5_file: h5py.File,
    quantity_name: str = "pressure",
    skip_chains_of_ducts_or_fixed=True,
) -> Tuple[List[int], List[Tuple[str, str]], List[str], Dict[int, Dict[str, float]]]:
    """Extracts scenario IDs, edges, V_ports, and pressure values from HDF5."""
    variable = h5_file["Optimisation Components"]["Variable"]
    if quantity_name == "pressure":
        scenarios = decode_scenarios(
            h5_file["Optimisation Components"]["Set"]["Scenarios"][:]
        )
    elif quantity_name == "sound_power_level":
        scenarios = decode_scenarios(
            h5_file["Optimisation Components"]["Set"]["max_noise_scenarios"][:]
        )
    E = decode_and_eval(h5_file["Optimisation Components"]["Set"]["E"])
    E_fixed = decode_and_eval(h5_file["Optimisation Components"]["Set"]["E_fixed"])
    E_duct = decode_and_eval(h5_file["Optimisation Components"]["Set"]["E_duct"])
    E_empty = decode_and_eval(h5_file["Optimisation Components"]["Set"]["E_empty"])

    V_target = decode_list(h5_file["Optimisation Components"]["Set"]["V_target"])
    V_source = decode_list(h5_file["Optimisation Components"]["Set"]["V_source"])

    # make sure V_targets are only leafs (necessary for subsequent shrinking step)
    V_target = [node for node in V_target if node not in [V_in for (V_in, V_out) in E]]

    # remove connected fixed, duct or empty for shrinking the graph
    if skip_chains_of_ducts_or_fixed:
        E_fixed_duct_empty = list(set(E_fixed + E_duct + E_empty))

        E = build_E_clean_with_rewiring(E, E_fixed_duct_empty)

    quantity = {}
    for s_key in scenarios:
        if quantity_name == "pressure":
            get_index = lambda row: row["V"].decode()
        else:
            get_index = lambda row: parse_key(row[0])

        quantity[s_key] = {
            get_index(row): row["value"]
            for row in variable["Scenario"][str(s_key)][quantity_name][:]
        }

    return scenarios, E, V_target, V_source, quantity


def process_pressure_branch_and_paths(
    file_path: str,
    quantity_name: str = "pressure",
    skip_chains_of_ducts_or_fixed: bool = True,
) -> Tuple[Dict[int, Dict[str, Dict[str, float]]], Dict[str, List[str]]]:
    """
    Extracts pressure values along room paths and builds root-to-room paths from the graph.

    Returns:
        - pressure_branch: scenario → room → node → pressure
        - root_room_paths: room → node path
    """
    quantity_along_branch: Dict[int, Dict[str, Dict[str, float]]] = {}

    with h5py.File(file_path, "r") as h5_file:
        scenarios, E, V_target, V_source, quantity_dict = extract_quantity_data(
            h5_file, quantity_name, skip_chains_of_ducts_or_fixed
        )

        graph = build_graph(E)
        root_room_paths = bfs_paths(graph, V_source[0], V_target)

        if quantity_name == "pressure":
            for scenario in scenarios:
                quantity_along_branch[scenario] = {
                    room: {
                        v: quantity_dict[scenario].get(v, np.nan)
                        for v in root_room_paths[room]
                    }
                    for room in V_target
                }
        elif quantity_name == "sound_power_level":
            for scenario in scenarios:
                quantity_along_branch[scenario] = {
                    room: {
                        (v, fi): quantity_dict[scenario].get((v, str(fi)), np.nan)
                        for fi in range(1, 9)
                        for v in root_room_paths[room]
                    }
                    for room in V_target
                }

    return quantity_along_branch, root_room_paths


def extract_pressure_change_data(
    h5_file: h5py.File,
) -> Tuple[
    Dict[int, Dict[Tuple[str, str], float]], Dict[int, Dict[Tuple[str, str], float]]
]:
    """Extracts pressure change data for duct and fixed components from HDF5."""
    scenarios = decode_scenarios(
        h5_file["Optimisation Components"]["Set"]["Scenarios"][:]
    )
    variable = h5_file["Optimisation Components"]["Variable"]
    E_duct = decode_and_eval(h5_file["Optimisation Components"]["Set"]["E_duct"])
    E_fixed = decode_and_eval(h5_file["Optimisation Components"]["Set"]["E_fixed"])

    pressure_changes_duct = {}
    pressure_changes_fixed = {}

    for s_key in scenarios:
        duct = {}
        fixed = {}
        try:
            for row in variable["Scenario"][str(s_key)]["pressure_change"][:]:
                edge = decode_h5_tuple_str_into_tuple(row["E"])
                if edge in E_duct:
                    duct[edge] = row["value"]
                elif edge in E_fixed:
                    fixed[edge] = row["value"]
        except KeyError:
            pass
        pressure_changes_duct[s_key] = duct
        pressure_changes_fixed[s_key] = fixed

    return pressure_changes_duct, pressure_changes_fixed


def process_pressure_changes(
    file_path: str,
) -> Tuple[
    Dict[int, Dict[Tuple[str, str], float]], Dict[int, Dict[Tuple[str, str], float]]
]:
    """Loads HDF5 file and extracts pressure change values for ducts and fixed parts."""
    with h5py.File(file_path, "r") as h5_file:
        return extract_pressure_change_data(h5_file)


def sum_pressure_loss_branch(
    root_room_paths: Dict[str, List[str]],
    pressure_changes: Dict[int, Dict[Tuple[str, str], float]],
    scenario: int,
) -> Dict[str, float]:
    """Sums the pressure loss along each root-to-room branch for a given scenario."""
    pressure_change_along_path = {room: 0.0 for room in root_room_paths.keys()}
    for room, path in root_room_paths.items():
        for V_in, V_out in zip(path[:-1], path[1:]):
            pressure_change_along_path[room] += pressure_changes[scenario].get(
                (V_in, V_out), 0.0
            )
    return pressure_change_along_path


def find_highest_pressure_loss_branch(
    root_room_paths: Dict[str, List[str]],
    pressure_changes: Dict[int, Dict[Tuple[str, str], float]],
) -> Tuple[Dict[int, str], Dict[int, float]]:
    """Returns the worst (highest-loss) room branch per scenario."""
    worst_branch, highest_dp = {}, {}
    for s in pressure_changes:
        dp_branch = sum_pressure_loss_branch(root_room_paths, pressure_changes, s)
        worst_branch[s] = max(dp_branch, key=dp_branch.get)
        highest_dp[s] = dp_branch[worst_branch[s]]
    return worst_branch, highest_dp


def sum_pressure_loss_branch_distribution(
    root_room_paths: Dict[str, List[str]],
    pressure_changes_duct: Dict[int, Dict[Tuple[str, str], float]],
    pressure_changes_fixed: Dict[int, Dict[Tuple[str, str], float]],
    scenario: int,
) -> Dict[str, Dict[str, float]]:
    """
    Separates pressure loss contributions from duct and fixed elements along each path.

    Returns:
        Dict[str, Dict[str, float]] with keys "duct" and "fixed", each mapping room → Δp.
    """
    pressure_change_along_path = {
        "duct": {room: 0.0 for room in root_room_paths},
        "fixed": {room: 0.0 for room in root_room_paths},
    }
    for room, path in root_room_paths.items():
        for V_in, V_out in zip(path[:-1], path[1:]):
            edge = (V_in, V_out)
            if edge in pressure_changes_duct[scenario]:
                pressure_change_along_path["duct"][room] += pressure_changes_duct[
                    scenario
                ][edge]
            elif edge in pressure_changes_fixed[scenario]:
                pressure_change_along_path["fixed"][room] += pressure_changes_fixed[
                    scenario
                ][edge]
    return pressure_change_along_path
