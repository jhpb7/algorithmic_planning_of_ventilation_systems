from collections import defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np
import h5py
import re
from src.preprocessing.fixed_element_calculation.duct_element_calculator import DuctCalc


Edge = Tuple[str, str]

import re

_tokenize = re.compile(r"[^(),\s]+").findall


def parse_key(b):
    return tuple(tok.strip("'\"") for tok in _tokenize(b.decode("utf-8")))


def build_graph(
    edges: List[Edge],
) -> Tuple[Dict[str, List[str]], Dict[str, int], Set[str]]:
    """Build adjacency list, in-degree counts, and node set for a DAG."""
    graph = defaultdict(list)
    in_degree = defaultdict(int)
    all_nodes = set()

    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1
        all_nodes.update([u, v])

    return graph, in_degree, all_nodes


def find_branch_node(edges: List[Edge]) -> str:
    """Return first branch/leaf node found when traversing from the root."""
    graph, in_degree, all_nodes = build_graph(edges)
    root = next((n for n in all_nodes if in_degree[n] == 0), None)
    if root is None:
        raise ValueError("No root found (graph may have cycles or no entry node).")

    current = root
    while True:
        neighbors = graph.get(current, [])
        if len(neighbors) != 1:
            return current
        current = neighbors[0]


def find_fan_edge(edges: List[Edge], E_fan_station: Set[Edge]) -> Edge | None:
    """Return first edge from root that is also in E_fan_station (or None)."""
    graph, in_degree, all_nodes = build_graph(edges)
    root = next((n for n in all_nodes if in_degree[n] == 0), None)
    if root is None:
        raise ValueError("No root found (graph may have cycles or no entry node).")

    current = root
    while True:
        neighbors = graph.get(current, [])
        if len(neighbors) != 1:
            return None
        next_node = neighbors[0]
        edge = (current, next_node)
        if edge in E_fan_station:
            return edge
        current = next_node


def get_max_volume_flow_in_problem(data: Dict) -> float:
    """Return maximum volume flow across all scenarios."""
    max_q = 0
    for s in data["Scenarios"][None]:
        max_q = max(max_q, np.max(list(data["scenario"][s]["volume_flow"].values())))
    return max_q


def get_fan_edge_volume_flow(data: Dict) -> Dict:
    """Return mapping: scenario -> edge -> volume flow for fan station edges."""
    fan_edge_volume_flow = defaultdict(dict)
    for s in data["Scenarios"][None]:
        fan_edge_volume_flow[s] = {}
        for e in data["E_fan_station"][None]:
            fan_edge_volume_flow[s][e] = data["scenario"][s]["volume_flow"][e]
    return fan_edge_volume_flow


def get_point_distance(value_lst: List[float]) -> float:
    """Return step size assuming values are from an equidistant grid."""
    sorted_lst = sorted(set(value_lst))
    diffs = np.diff(sorted_lst)
    if len(diffs) == 0:
        return 0
    if len(diffs) == 1:
        return diffs[0]
    if np.any(np.abs(np.diff(diffs)) > 1e-3):
        raise ValueError("List values are not equally spaced.")
    return np.min(diffs)


def compute_fixed_zeta_from_yaml(data: Dict, path_to_file: str) -> Dict:
    """Compute the fixed zeta values from yaml files and stores them into "fixed_zeta" subdict."""

    filedict = data["fixed_zeta_link"]
    zeta_dict = {
        key: DuctCalc.from_yaml(path_to_file + file).zeta
        for key, file in filedict.items()
    }
    if not None in data["fixed_zeta_central_AHU"]:
        zeta_dict.update(data["fixed_zeta_central_AHU"])

    data["fixed_zeta"] = zeta_dict
    return data


def compute_crosstalk_dampening(data, path):

    data["crosstalk_dampening_duct"] = {}
    for (v1, v2), file in data["crosstalk_link"].items():
        calc = DuctCalc.from_yaml(path + file, 0, include_acoustics=True)
        dampening_duct = np.array(calc.dampening)[1:6]
        data["crosstalk_dampening_duct"].update(
            {(i + 2, v1, v2): damp for i, damp in enumerate(dampening_duct)}
        )
    return data


def compute_flow_noise_and_dampening_dicts_from_yaml(
    data: Dict, path_to_file: str
) -> Dict:
    """Compute the fixed zeta values from yaml files and stores them into "fixed_zeta" subdict."""

    filedict = data["fixed_zeta_link"]

    if not "Scenarios" in data:
        raise KeyError(
            "No Scenarios in data dict, can't compute acoustics for fixed elements."
        )

    for s in data["Scenarios"][None]:
        data_scen = data["scenario"][s]
        res_dict = {"fixed_flow_noise": dict(), "fixed_dampening": dict()}
        for edge, file in filedict.items():
            volume_flow = data_scen["volume_flow"][edge]
            calc = DuctCalc.from_yaml(
                path_to_file + file, volume_flow=volume_flow, include_acoustics=1
            )

            res_dict["fixed_flow_noise"].update(
                {(*edge, fi + 1): val for fi, val in enumerate(calc.flow_noise)}
            )
            res_dict["fixed_dampening"].update(
                {(*edge, fi + 1): val for fi, val in enumerate(calc.dampening)}
            )

        for key in res_dict.keys():
            if key in data_scen:
                data_scen[key].update(res_dict[key])
            else:
                data_scen[key] = res_dict[key]
    return data


def read_duct_data_from_h5(h5file: str):

    def read_keys_and_values(group):
        keys = [parse_key(k) for k in group["E_duct"]]
        values = group["value"][:]
        return keys, values

    with h5py.File(h5file, "r") as f:
        oc = f["Optimisation Components"]
        var = oc["Variable"]
        param = oc["Parameter"]
        edges, widths = read_keys_and_values(var["duct_width"])
        _, heights = read_keys_and_values(var["duct_height"])
        _, lengths = read_keys_and_values(param["duct_length"])
        _, n_bendings = read_keys_and_values(param["n_duct_bendings"])

        data = {
            edge: {
                "width": w,
                "height": h,
                "length": l,
                "n_bendings": nb,
            }
            for edge, w, h, l, nb in zip(edges, widths, heights, lengths, n_bendings)
        }

    return data


def create_duct_dict(duct_data, branch_data):
    elements = dict()

    n_bendings = int(duct_data["n_bendings"])

    counter = 1
    if branch_data:
        counter += 1
        elements.update({1: branch_data})

    for i in range(counter, counter + 2 * n_bendings, 2):
        elements.update(
            {
                i: {"name": "duct", "length": duct_data["length"] / (n_bendings + 1)},
                i
                + 1: {
                    "name": "rect_bending",
                    "bending_radius": duct_data["width"],  # assuming R/w = 1
                    "bending_angle": 90,
                    "n_bendings": 1,
                },
            }
        )
    elements.update(
        {
            len(elements)
            + 1: {"name": "duct", "length": duct_data["length"] / (n_bendings + 1)}
        }
    )

    duct_dict = {
        "system": {
            "width": duct_data["width"],
            "height": duct_data["height"],
            "elements": elements,
        }
    }
    return duct_dict


def get_branch_data(edge, data, s, duct_data):

    def get_branch_values(edge):
        data_scen = data["scenario"][s]

        pred_edge = find_predecessor_edges(data["E"][None], edge[0])[0]
        all_branch_edges = find_successor_edges(data["E"][None], pred_edge[1])

        if len(all_branch_edges) != 2:
            raise ValueError(
                f"The number of branch edges of {pred_edge} should be 2 but is {len(all_branch_edges)}"
            )
        for branch_edge in all_branch_edges:
            if edge == branch_edge:
                target_branch_data = duct_data[edge]
                target_volume_flow = data_scen["volume_flow"][edge]
            else:
                neighbouring_branch_data = duct_data[branch_edge]
                neighbouring_volume_flow = data_scen["volume_flow"][branch_edge]
        main_branch_data = duct_data[pred_edge]

        main_volume_flow = data_scen["volume_flow"][pred_edge]

        element = {
            "name": "rect_branch",
            "main_width": main_branch_data["width"],
            "main_height": main_branch_data["height"],
            "main_volume_flow": main_volume_flow,
            "neighbour_width": neighbouring_branch_data["width"],
            "neighbour_height": neighbouring_branch_data["height"],
            "neighbour_volume_flow": neighbouring_volume_flow,
            "target_width": target_branch_data["width"],
            "target_height": target_branch_data["height"],
            "target_volume_flow": target_volume_flow,
        }

        return element

    def find_predecessor_edges(E, node):
        return [(e0, e1) for (e0, e1) in E if e1 == node]

    def find_successor_edges(E, node):
        return [(e0, e1) for (e0, e1) in E if e0 == node]

    if not edge in data["E_duct"][None]:
        raise KeyError(f"Edge {edge} is missing in duct edge list")
    duct_e_bend = [(i, k) for (i, j, k) in data["duct_e_branch"][None]]
    duct_e_straight = [(i, j) for (i, j, k) in data["duct_e_branch"][None]]

    if edge[0] in data["duct_t_branch_node"][None]:
        element = get_branch_values(edge)
        element.update({"direction": "bend"})
    elif edge in duct_e_bend:
        element = get_branch_values(edge)
        element.update({"direction": "bend"})
    elif edge in duct_e_straight:
        element = get_branch_values(edge)
        element.update({"direction": "straight"})
    else:
        element = None
    return element


def add_duct_zeta_flow_noise_and_dampening_from_h5(
    data: Dict, path_to_h5file: str
) -> Dict:
    """Compute the zeta values, flow noise and dampening of all duct elements from a topology optimization run. These are then added to the data dict."""

    duct_data = read_duct_data_from_h5(path_to_h5file)

    if not "Scenarios" in data:
        raise KeyError(
            "No Scenarios in data dict, can't compute acoustics for fixed elements."
        )

    if "duct_zeta" not in data:
        data["duct_zeta"] = dict()

    for s in data["Scenarios"][None]:
        data_scen = data["scenario"][s]
        res_dict = {
            "fixed_flow_noise": dict(),
            "fixed_dampening": dict(),
            "duct_zeta": dict(),
        }
        for edge, values in duct_data.items():
            branch_data = get_branch_data(edge, data, s, duct_data)
            duct_dict = create_duct_dict(values, branch_data)
            volume_flow = data_scen["volume_flow"][edge]
            calc = DuctCalc.from_yaml(
                duct_dict, volume_flow=volume_flow, include_acoustics=1
            )

            res_dict["fixed_flow_noise"].update(
                {(*edge, fi + 1): val for fi, val in enumerate(calc.flow_noise)}
            )
            res_dict["fixed_dampening"].update(
                {(*edge, fi + 1): val for fi, val in enumerate(calc.dampening)}
            )
            data["duct_zeta"].update({edge: calc.zeta})

        for key in res_dict.keys():
            if key in data_scen:
                data_scen[key].update(res_dict[key])
            else:
                data_scen[key] = res_dict[key]

    return data


def to_str(x):
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("utf-8")
    return str(x)


def is_int_like(x) -> bool:
    # bool is a subclass of int, so exclude it if you don't want True/False counted
    if isinstance(x, int) and not isinstance(x, bool):
        return True
    if isinstance(x, str):
        try:
            int(x.strip())  # handles "8", "  8 ", "+8", "-3"
            return True
        except ValueError:
            return False
    return False


def read_pressure_map(f, scen_name: str) -> Dict[str, float]:
    """Reads the pressure map for the scenario."""
    BASE_SCEN = "Optimisation Components/Variable/Scenario"
    p_path = f"{BASE_SCEN}/{scen_name}/pressure"
    d = f[p_path][()]
    if not (hasattr(d, "dtype") and d.dtype.names):
        raise ValueError(f"{p_path} is not a structured table (dtype.names missing)")
    if "V" not in d.dtype.names or "value" not in d.dtype.names:
        raise KeyError(f"{p_path} needs fields 'V' and 'value'. Got: {d.dtype.names}")
    V = [to_str(x) for x in d["V"]]
    val = np.asarray(d["value"], dtype=float)
    return {node: float(p) for node, p in zip(V, val)}


def build_parent_map(edges: List[Tuple[str, str]]) -> Dict[str, str]:
    """Builds the parent map from the edges."""
    parent: Dict[str, str] = {}
    for u, v in edges:
        parent[v] = u
        parent.setdefault(u, parent.get(u, None))
    return parent


def get_main_branch_end(graph: Dict[str, List[str]], root: str) -> str:
    """Finds the main branching node (node with out-degree != 1)."""
    cur = root
    seen = set()
    while True:
        if cur in seen:
            raise ValueError("Cycle detected (graph is not a tree).")
        seen.add(cur)
        children = graph.get(cur, [])
        if len(children) == 1:
            cur = children[0]
        else:
            break
    return cur


def path_nodes_root_to(leaf: str, parent: Dict[str, str], root: str) -> List[str]:
    """Returns the path from the root to the given leaf."""
    nodes = []
    x = leaf
    while x is not None:
        nodes.append(x)
        x = parent.get(x, None)
    nodes.reverse()
    if not nodes or nodes[0] != root:
        raise ValueError(f"Leaf {leaf!r} is not connected to root {root!r}.")
    return nodes


def last_edge_in_set(
    path_edges: List[Tuple[str, str]], S: Set[Tuple[str, str]]
) -> Tuple[str, str]:
    """Returns the last edge in the path that is present in the set S."""
    for e in reversed(path_edges):
        if e in S:
            return e
    return None


def compute_terminal_branch_pressure_loss(
    terminal_edges: List[Tuple[str, str]],
    excluded: Set[Tuple[str, str]],
    pressure_map: Dict[str, float],
) -> float:
    """Computes the pressure loss for a terminal branch."""
    pressure_loss = 0.0
    for u, v in terminal_edges:
        if (u, v) in excluded:
            continue
        if u not in pressure_map or v not in pressure_map:
            missing = u if u not in pressure_map else v
            raise KeyError(f"Missing pressure data for node {missing!r}")
        pressure_loss += pressure_map[u] - pressure_map[v]
    return pressure_loss


def compute_terminal_branch_max_pressure_changes(
    h5path: str, edges: List[Tuple[str, str]], scen: str
) -> List[Tuple[Tuple[str, str], Tuple[str, str], float]]:
    """Returns terminal branch tuples for all leaf branches and computes pressure loss."""
    BASE_SCEN = "Optimisation Components/Variable/Scenario"
    E_VFC_PATH = "Optimisation Components/Set/E_vfc_leaf"
    E_FS_PATH = "Optimisation Components/Set/E_fan_station_leaf"

    # ---- build tree structure ----
    graph, in_degree, all_nodes = build_graph(edges)
    roots = [n for n in all_nodes if in_degree.get(n, 0) == 0]
    if len(roots) != 1:
        raise ValueError(f"Expected exactly one root, found {len(roots)}: {roots}")
    root = roots[0]

    out_degree = {n: len(graph.get(n, [])) for n in all_nodes}
    parent = build_parent_map(edges)

    # ---- main branching node ----
    main_branch_end = get_main_branch_end(graph, root)

    leaves = sorted([n for n in all_nodes if out_degree.get(n, 0) == 0])

    # Read pressure data
    with h5py.File(h5path, "r") as f:
        if scen not in f[BASE_SCEN]:
            raise KeyError(f"Scenario {scen!r} not found.")
        pressure_map = read_pressure_map(f, scen)
        E_vfc_set: Set[Tuple[str, str]] = (
            set(parse_key(x) for x in f[E_VFC_PATH][()]) if E_VFC_PATH in f else set()
        )
        E_fs_set: Set[Tuple[str, str]] = (
            set(parse_key(x) for x in f[E_FS_PATH][()]) if E_FS_PATH in f else set()
        )
        excluded: Set[Tuple[str, str]] = E_vfc_set | E_fs_set

    results = {"E_vfc_leaf": dict(), "E_fan_station_leaf": dict()}
    for leaf in leaves:
        nodes = path_nodes_root_to(leaf, parent, root)

        if main_branch_end not in nodes:
            raise ValueError(
                f"main_branch_end {main_branch_end!r} not on path to leaf {leaf!r}."
            )

        # terminal branch edges: main_branch_end -> leaf
        i0 = nodes.index(main_branch_end)
        terminal_nodes = nodes[i0:]
        terminal_edges = list(zip(terminal_nodes[:-1], terminal_nodes[1:]))

        # Calculate terminal branch pressure loss
        pressure_loss = compute_terminal_branch_pressure_loss(
            terminal_edges, excluded, pressure_map
        )

        # leaf branch: last branching node -> leaf
        x = leaf
        while parent.get(x, None) is not None and out_degree.get(parent[x], 0) == 1:
            x = parent[x]  # climb until parent is branching (out_degree > 1) or root
        last_branch = parent.get(x, None) or root

        j0 = nodes.index(last_branch)
        leaf_branch_nodes = nodes[j0:]
        leaf_branch_edges = list(zip(leaf_branch_nodes[:-1], leaf_branch_nodes[1:]))

        vfc_edge = last_edge_in_set(leaf_branch_edges, E_vfc_set)
        fs_edge = last_edge_in_set(leaf_branch_edges, E_fs_set)

        results["E_vfc_leaf"].update({vfc_edge: pressure_loss})
        results["E_fan_station_leaf"].update({fs_edge: pressure_loss})

    return results


def out_adjacency(edges: List[Edge]) -> Dict[str, List[str]]:
    """
    Build outgoing adjacency: out[u] = [v1, v2, ...] for edges (u, v_i).
    """
    out: Dict[str, List[str]] = {}
    for a, b in edges:
        out.setdefault(a, []).append(b)
    return out


def in_adjacency(edges: List[Edge]) -> Dict[str, List[str]]:
    """
    Build incoming adjacency: inn[v] = [u1, u2, ...] for edges (u_i, v).
    """
    inn: Dict[str, List[str]] = {}
    for a, b in edges:
        inn.setdefault(b, []).append(a)
    return inn


def first_duct_upstream(
    edge: Edge, inn: Dict[str, List[str]], E_duct: Set[Edge]
) -> Edge | None:
    """
    Walk upstream starting at edge[0] (the start node of the given edge),
    following the unique parent chain (in-degree must be exactly 1).

    Returns the first duct edge (parent, cur) encountered, or None if the path
    ends (in-degree == 0) before finding duct.

    Raises ValueError if a branch/merge is encountered (in-degree > 1) or a cycle is detected.
    """
    cur = edge[0]
    seen: Set[str] = set()

    while True:
        if cur in seen:
            raise ValueError(f"cycle detected upstream from {edge}")
        seen.add(cur)

        prevs = inn.get(cur, [])
        if len(prevs) == 0:
            return None
        if len(prevs) > 1:
            raise ValueError(
                f"branch/merge upstream before duct from {edge} at node {cur}"
            )

        parent = prevs[0]
        e = (parent, cur)
        if e in E_duct:
            return e

        cur = parent


def first_duct_downstream(
    edge: Edge, out: Dict[str, List[str]], E_duct: Set[Edge]
) -> Edge | None:
    """
    Walk downstream starting at edge[1] (the end node of the given edge),
    following the unique child chain (out-degree must be exactly 1).

    Returns the first duct edge (cur, child) encountered, or None if the path
    ends (out-degree == 0) before finding duct.

    Raises ValueError if a branch is encountered (out-degree > 1) or a cycle is detected.
    """
    cur = edge[1]
    seen: Set[str] = set()

    while True:
        if cur in seen:
            raise ValueError(f"cycle detected downstream from {edge}")
        seen.add(cur)

        nxts = out.get(cur, [])
        if len(nxts) == 0:
            return None
        if len(nxts) > 1:
            raise ValueError(f"branch downstream before duct from {edge} at node {cur}")

        child = nxts[0]
        e = (cur, child)
        if e in E_duct:
            return e

        cur = child


def add_component_dimensions_from_duct_using_h5(
    data: Dict, path_to_h5file: str
) -> Dict:
    """
    For each edge in E_silencer and E_vfc:
      1) try to find the first duct edge upstream,
      2) if none found (hit root), try downstream,
      3) if still none found, raise ValueError.

    Branches (in-degree>1 upstream or out-degree>1 downstream) raise ValueError immediately.
    """
    duct_data = read_duct_data_from_h5(path_to_h5file)

    all_edges: List[Edge] = [tuple(e) for e in data["E"][None]]
    E_duct: Set[Edge] = set(tuple(e) for e in data["E_duct"][None])

    out = out_adjacency(all_edges)
    inn = in_adjacency(all_edges)

    print("TOOK Correct w.")
    input("WRONG h instead of w")

    def quantize_hw(
        h: float, w: float, max_min_dimensions: List[float]
    ) -> tuple[float, float]:
        max_height, min_height, max_width, min_width = max_min_dimensions
        h = min(max_height, max(min_height, np.ceil(h * 10) / 10))
        w = min(max_width, max(min_width, np.ceil(h * 10) / 10))
        return h, w

    def find_duct_or_fail(e: Edge) -> Edge:
        duct = first_duct_upstream(e, inn, E_duct)
        if duct is None:
            duct = first_duct_downstream(e, out, E_duct)
        if duct is None:
            raise ValueError(f"no duct found upstream or downstream for {e}")
        return duct

    if "silencer_height" not in data:
        data["silencer_height"] = {}
        data["silencer_width"] = {}
    for sil in (tuple(e) for e in data["E_silencer"][None]):
        duct_edge = find_duct_or_fail(sil)
        h, w = duct_data[duct_edge]["height"], duct_data[duct_edge]["width"]
        h, w = quantize_hw(h, w, [1, 0.1, 1.9, 0.3])
        data["silencer_height"][sil] = h
        data["silencer_width"][sil] = w

    if "vfc_height" not in data:
        data["vfc_height"] = {}
        data["vfc_width"] = {}
    for vfc in (tuple(e) for e in data["E_vfc"][None]):
        duct_edge = find_duct_or_fail(vfc)
        h, w = duct_data[duct_edge]["height"], duct_data[duct_edge]["width"]
        h, w = quantize_hw(h, w, [0.6, 0.1, 1, 0.2])
        data["vfc_height"][vfc] = h
        data["vfc_width"][vfc] = w

    return data


def prepare_load_case_yaml(data: Dict) -> Dict:
    """Convert raw scenario data into load-case YAML format."""
    load_case_data = {
        idx
        + 1: {
            "volume_flow": {
                key2: val["mean"] / 3600 for key2, val in scen["room"].items()
            }
        }
        for idx, (key, scen) in enumerate(data["scenario"].items())
    }

    time_shares = {idx + 1: v for idx, v in enumerate(data["time_share"].values())}

    return {
        "scenario": load_case_data,
        "Scenarios": {None: list(load_case_data.keys())},
        "time_share": time_shares,
    }
