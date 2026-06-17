from collections import defaultdict

import numpy as np
import pyomo.environ as pyo

from src.pyomo_models.element_functions import (
    add_level_addition_elements,
    level_add_multiple_multiindex,
    level_add_room,
)


def linearize_log10(m_scen):
    """
    the interval of linearisation is chosen to be that a fan supplies minimally 1/15th of the maximal volume flow (which is either the fan's max volume flow or the max volume flow in the scenario.
    Secondly, the number of segments is chosen so that the error of 20*log10(pressure) is just lower than 1.
    """

    model = m_scen.parent_block()

    m_scen.linearize_increment_v = pyo.Set(initialize=range(1, 11))

    m_scen.linearize_increment_S = pyo.Set(
        initialize=[(i, i + 1) for i in range(1, 10)]
    )

    m_scen.linearize_y_0 = pyo.Param(
        m_scen.linearize_increment_v,
        initialize={
            1: 0.001,
            2: 0.0021544346900318843,
            3: 0.004641588833612777,
            4: 0.01,
            5: 0.021544346900318832,
            6: 0.046415888336127774,
            7: 0.1,
            8: 0.21544346900318823,
            9: 0.46415888336127775,
            10: 1.0,
        },
        doc="parameters for linearising log10 from 0 to 1, x values",
    )

    m_scen.linearize_log10_y_0 = pyo.Param(
        m_scen.linearize_increment_v,
        initialize={
            1: -3.0,
            2: -2.6666666666666665,
            3: -2.3333333333333335,
            4: -2.0,
            5: -1.6666666666666667,
            6: -1.3333333333333335,
            7: -1.0,
            8: -0.666666666666667,
            9: -0.3333333333333335,
            10: 0.0,
        },
        doc="parameters for linearising log10 from 0 to 1, y values",
    )

    def linearize_log10_volume_flow(m_scen):

        m_scen.linearize_increment_z_q = pyo.Var(
            model.fan_set, list(m_scen.linearize_increment_v)[1:9], within=pyo.Binary
        )

        m_scen.linearize_increment_delta_q = pyo.Var(
            model.fan_set, m_scen.linearize_increment_S, within=pyo.Reals, bounds=(0, 1)
        )

        @m_scen.Constraint(model.fan_set, m_scen.linearize_increment_S)
        def lower_limit_delta(m_scen, i, j, p, d, n, k, l):
            if k > 1:
                return (
                    m_scen.linearize_increment_z_q[i, j, p, d, n, k]
                    >= m_scen.linearize_increment_delta_q[i, j, p, d, n, k, l]
                )
            return pyo.Constraint.Skip

        @m_scen.Constraint(model.fan_set, m_scen.linearize_increment_S)
        def upper_limit_delta(m_scen, i, j, p, d, n, k, l):
            if k < 9:
                return (
                    m_scen.linearize_increment_delta_q[i, j, p, d, n, k, l]
                    >= m_scen.linearize_increment_z_q[i, j, p, d, n, l]
                )
            return pyo.Constraint.Skip

        m_scen.log10_fan_volume_flow_intermediate = pyo.Var(
            model.fan_set,
            within=pyo.NonPositiveReals,
            bounds=(
                -3,
                0,
            ),
        )

        @m_scen.Constraint(model.fan_set)
        def linearize_define_q(m_scen, i, j, p, d, n):
            return m_scen.fan_volume_flow_intermediate[
                i, j, p, d, n
            ] / m_scen.fan_volume_flow_max_scenario[i, j, p, d] == m_scen.linearize_y_0[
                1
            ] + sum(
                (m_scen.linearize_y_0[l] - m_scen.linearize_y_0[k])
                * m_scen.linearize_increment_delta_q[i, j, p, d, n, k, l]
                for (k, l) in m_scen.linearize_increment_S
            )

        @m_scen.Constraint(model.fan_set)
        def linearize_define_log10_q(m_scen, i, j, p, d, n):
            return m_scen.log10_fan_volume_flow_intermediate[
                i, j, p, d, n
            ] == m_scen.linearize_log10_y_0[1] + sum(
                (m_scen.linearize_log10_y_0[l] - m_scen.linearize_log10_y_0[k])
                * m_scen.linearize_increment_delta_q[i, j, p, d, n, k, l]
                for (k, l) in m_scen.linearize_increment_S
            )

        return m_scen

    def linearize_log10_pressure_rise(m_scen):

        m_scen.linearize_increment_z_p = pyo.Var(
            model.fan_set, list(m_scen.linearize_increment_v)[1:9], within=pyo.Binary
        )

        m_scen.linearize_increment_delta_p = pyo.Var(
            model.fan_set, m_scen.linearize_increment_S, within=pyo.Reals, bounds=(0, 1)
        )

        @m_scen.Constraint(model.fan_set, m_scen.linearize_increment_S)
        def lower_limit_delta_p(m_scen, i, j, p, d, n, k, l):
            if k > 1:
                return (
                    m_scen.linearize_increment_z_p[i, j, p, d, n, k]
                    >= m_scen.linearize_increment_delta_p[i, j, p, d, n, k, l]
                )
            return pyo.Constraint.Skip

        @m_scen.Constraint(model.fan_set, m_scen.linearize_increment_S)
        def upper_limit_delta_p(m_scen, i, j, p, d, n, k, l):
            if k < 9:
                return (
                    m_scen.linearize_increment_delta_p[i, j, p, d, n, k, l]
                    >= m_scen.linearize_increment_z_p[i, j, p, d, n, l]
                )
            return pyo.Constraint.Skip

        m_scen.log10_fan_pressure_change = pyo.Var(
            model.fan_set,
            within=pyo.Reals,
            bounds=(-3, 0),
        )

        @m_scen.Constraint(model.fan_set)
        def linearize_define_p(m_scen, i, j, p, d, n):
            return m_scen.fan_pressure_change_dimless[
                i, j, p, d, n
            ] == m_scen.linearize_y_0[1] + sum(
                (m_scen.linearize_y_0[l] - m_scen.linearize_y_0[k])
                * m_scen.linearize_increment_delta_p[i, j, p, d, n, k, l]
                for (k, l) in m_scen.linearize_increment_S
            )

        @m_scen.Constraint(model.fan_set)
        def linearize_define_log10_p(m_scen, i, j, p, d, n):
            return m_scen.log10_fan_pressure_change[
                i, j, p, d, n
            ] == m_scen.linearize_log10_y_0[1] + sum(
                (m_scen.linearize_log10_y_0[l] - m_scen.linearize_log10_y_0[k])
                * m_scen.linearize_increment_delta_p[i, j, p, d, n, k, l]
                for (k, l) in m_scen.linearize_increment_S
            )

        return m_scen

    m_scen = linearize_log10_volume_flow(m_scen)
    m_scen = linearize_log10_pressure_rise(m_scen)
    return m_scen


def find_leafy_edges(model, switch):
    "Wrapper around initializer. Need for abstract model"

    def initializer_(m):
        """
        Returns the list of fan station or vfc edges that lead to a leaf
        through a path with no branches in a directed tree.

        Parameters:
            edges (list of tuple): All directed edges (u, v).
            fan_station_edges (set of tuple): Subset of edges (u, v).

        Returns:
            list of tuple: Valid fan station edges.
        """
        adj = defaultdict(list)
        for u, v in m.E:
            adj[u].append(v)

        def is_valid_fan_edge(v):
            visited = set()
            while v in adj and len(adj[v]) == 1:
                if v in visited:
                    return False  # cycle detection
                visited.add(v)
                v = adj[v][0]
                if len(adj[v]) > 1:
                    return False  # branch
            return len(adj[v]) == 0  # valid only if ends in leaf

        if switch == "fan_station":
            edge_searcher = m.E_fan_station
        elif switch == "vfc":
            edge_searcher = m.E_vfc
        elif switch == "silencer":
            edge_searcher = m.E_silencer
        else:
            raise ValueError("no such edge type")

        return [e for e in edge_searcher if is_valid_fan_edge(e[1])]

    return initializer_


def find_root(model):
    """
    Return a Pyomo initializer that finds the root of the graph.
    If no such edge exists, returns [].

    Assumptions:
    - m.E is a directed tree with exactly one root.
    - Edges are oriented away from the root.
    """

    def initializer_(m):
        # --- Build adjacency and indegree ---
        adj = defaultdict(list)
        indegree = defaultdict(int)
        nodes = set()

        for u, v in m.E:
            adj[u].append(v)
            indegree[v] += 1
            nodes.add(u)
            nodes.add(v)

        # --- Find the unique root (indegree == 0) ---
        roots = [n for n in nodes if indegree.get(n, 0) == 0]
        if len(roots) != 1:
            raise ValueError(f"Expected exactly one root, found {roots}")
        root = roots[0]
        return [root]

    return initializer_


def find_first_fan_edge_before_branch(model):
    """
    Return a Pyomo initializer that finds the first fan (or vfc) edge
    along the unique path starting from the root and stopping at the
    first branch (or leaf), and returns it as a singleton list.
    If no such edge exists, returns [].

    Assumptions:
    - m.E is a directed tree with exactly one root.
    - Edges are oriented away from the root.
    """

    def initializer_(m):
        # --- Build adjacency and indegree ---
        adj = defaultdict(list)
        indegree = defaultdict(int)
        nodes = set()

        for u, v in m.E:
            adj[u].append(v)
            indegree[v] += 1
            nodes.add(u)
            nodes.add(v)

        # --- Find the unique root (indegree == 0) ---
        roots = [n for n in nodes if indegree.get(n, 0) == 0]
        if len(roots) != 1:
            raise ValueError(f"Expected exactly one root, found {roots}")
        root = roots[0]

        fan_edges = m.E_fan_station

        # --- Walk from root down until first branch or leaf ---
        visited = set()
        node = root
        first_fan_edge = None

        while True:
            if node in visited:
                raise ValueError("Cycle detected in graph")
            visited.add(node)

            children = adj.get(node, [])

            # Leaf: stop
            if len(children) == 0:
                break

            # Branch: stop *before* going down any outgoing edge of it
            if len(children) > 1:
                break

            # Exactly one child: continue down the "trunk"
            child = children[0]
            edge = (node, child)

            # First fan edge on this trunk (before the branch)
            if first_fan_edge is None and edge in fan_edges:
                first_fan_edge = edge

            node = child

        # Pyomo initializer must return an iterable
        if first_fan_edge is None:
            return []
        else:
            return [first_fan_edge]

    return initializer_


def find_silencer_on_leaf_strand(model, leaf_node):
    """
    Return a Pyomo initializer that finds the unique silencer edge on the
    branch-free leaf strand ending at `leaf_node`.

    The "leaf strand" is the unique path from the leaf upward until the first
    branching node (or the root). That means:
    - the edge leaving the branching node toward this leaf strand is included
    - no other branches are allowed further down that strand

    Parameters:
        leaf_node: node id of the leaf for which the strand is inspected

    Returns:
        initializer_ for a Pyomo Set

    Behavior:
    - returns [silencer_edge] if exactly one silencer is found on that strand
    - returns [] if no silencer is found
    - raises ValueError if more than one silencer is found
    """

    # --- Build adjacency and parent map ---
    adj = defaultdict(list)
    parent = {}
    nodes = set()

    for u, v in model.E:
        adj[u].append(v)
        if v in parent:
            raise ValueError(f"Node {v} has more than one parent; graph is not a tree")
        parent[v] = u
        nodes.add(u)
        nodes.add(v)

    if leaf_node not in nodes:
        raise ValueError(f"Leaf node {leaf_node} is not part of the graph")

    # --- Check that leaf_node is really a leaf ---
    if len(adj.get(leaf_node, [])) != 0:
        raise ValueError(f"Node {leaf_node} is not a leaf")

    silencer_edges = set(model.E_silencer)

    # --- Walk upward from leaf to the first branch/root ---
    strand_edges = []
    node = leaf_node

    while node in parent:
        p = parent[node]
        edge = (p, node)
        strand_edges.append(edge)

        # Stop once we reached the first branching parent
        # (the edge leaving that branch toward the leaf is part of the strand)
        if len(adj[p]) > 1:
            break

        node = p

    # --- Find silencers on this strand ---
    silencers_on_strand = [e for e in strand_edges if e in silencer_edges]

    if len(silencers_on_strand) > 1:
        raise ValueError(
            f"Expected at most one silencer on leaf strand ending at {leaf_node}, "
            f"found {silencers_on_strand}"
        )

    if len(silencers_on_strand) == 0:
        return []

    return [silencers_on_strand[0]]


def model(
    planning_mode,
    duct_model=1,
    fan_model=1,
    branching_constraints=0,
    velocity_constraint=1,
    pressure_target_met=1,
    reduce_fan_curves=1,
    additional_investment_costs=0,
    all_scenarios_acoustics=0,
    variable_air_volume=1,
    capex_reduction=None,
):
    if planning_mode not in ["Topology", "Configuration"]:
        raise ValueError(
            "Please define a planning mode when loading the optimal planning model."
        )
    model = pyo.AbstractModel()

    model.Scenarios = pyo.Set(
        doc="Different load cases that are considered for operation"
    )

    model.V = pyo.Set(dimen=1, doc="Set of nodes")

    model.V_source = pyo.Set(within=model.V, doc="Set of source nodes")

    model.V_target = pyo.Set(
        within=model.V, doc="Set of target nodes, used for pressure targets"
    )

    model.E = pyo.Set(within=model.V * model.V, doc="Set of edges")

    model.E_fan_station = pyo.Set(within=model.E, doc="Set of fan station edges")

    model.E_vfc = pyo.Set(
        within=model.E - model.E_fan_station, doc="Set of volume flow controller edges"
    )

    model.E_fan_station_central = pyo.Set(
        within=model.E_fan_station,
        initialize=find_first_fan_edge_before_branch(model),
        doc="Set of the one central fan station",
    )

    model.E_fan_station_leaf = pyo.Set(
        within=model.E_fan_station,
        initialize=find_leafy_edges(model, "fan_station"),
        doc="Set of fan stations that are connected to an air diffuser without an intermediate branch",
    )

    model.E_vfc_leaf = pyo.Set(
        within=model.E_vfc,
        initialize=find_leafy_edges(model, "vfc"),
        doc="Set of VFCs that are connected to an air diffuser without an intermediate branch",
    )

    model.E_duct = pyo.Set(
        within=model.E - model.E_fan_station - model.E_vfc, doc="Set of duct edges"
    )
    model.E_duct_vertical = pyo.Set(
        within=model.E_duct,
        doc="Set of duct edges that are oriented vertically and thus shouldn't be limited when height is limited",
    )
    model.E_fixed = pyo.Set(
        within=model.E - model.E_fan_station - model.E_vfc - model.E_duct,
        doc="Set of fixed edges, i.e. a fixed zeta value according to dp = zeta * volume_flow^2",
    )

    model.E_silencer = pyo.Set(
        within=model.E
        - model.E_fan_station
        - model.E_vfc
        - model.E_duct
        - model.E_fixed,
        doc="Set of silencer edges which have a constant fixed zeta value",
    )

    model.E_empty = pyo.Set(
        initialize=model.E
        - model.E_fan_station
        - model.E_vfc
        - model.E_duct
        - model.E_fixed
        - model.E_silencer,
        doc="Set of empty nodes, where none of the above elements is present. This set always has to contain all edges that are not defined elsewhere!",
    )

    # Parameters

    if capex_reduction:
        model.capex_reduction = pyo.Param(
            initialize=capex_reduction,
            doc="CAPEX is reduced to capex_reduction %, this is applied to all CAPEX related costs.",
        )
    else:
        capex_reduction = 1

    model.time_share = pyo.Param(
        model.Scenarios, doc="Time share of each scenario. Sums up to 1"
    )

    model.max_pressure = pyo.Param(
        doc="Maximum pressure in the problem. Used for bigM constraints"
    )

    model.additional_cost_component_set = pyo.Set(
        initialize=["fan", "variable_vfc", "constant_vfc", "both"]
    )

    model.additional_measurement_costs = pyo.Param(
        model.additional_cost_component_set,
        doc="cost for additional measurement equipment for fans or vfcs.",
    )

    if planning_mode == "Configuration":

        if all_scenarios_acoustics:

            model.max_central_noise_scenarios = pyo.Set(
                initialize=model.Scenarios,
                doc="Set of Scenario(s) where central noise is set to be maximal.",
            )

            model.max_distributed_throttling_noise_scenarios = pyo.Set(
                dimen=1,
                initialize=None,
                within=model.Scenarios,
                doc="Set of Scenario(s) where distributed noise due to vfc is set to be maximal.",
            )

        else:
            model.max_central_noise_scenarios = pyo.Set(
                dimen=1,
                within=model.Scenarios,
                doc="Set of Scenario(s) where central noise is set to be maximal.",
            )

            model.max_distributed_throttling_noise_scenarios = pyo.Set(
                dimen=1,
                within=model.Scenarios,
                doc="Set of Scenario(s) where distributed noise due to vfc is set to be maximal.",
            )

            model.max_distributed_pressure_rise_noise_scenarios = pyo.Set(
                dimen=1,
                within=model.Scenarios,
                doc="Set of Scenario(s) where distributed noise due to fans is set to be maximal.",
            )

        model.max_distributed_noise_scenarios = pyo.Set(
            dimen=1,
            initialize=model.max_distributed_throttling_noise_scenarios,
            doc="Set of Scenario(s) where distributed noise is set to be maximal.",
        )

        model.max_noise_scenarios = pyo.Set(
            dimen=1,
            within=model.Scenarios,
            initialize=model.max_central_noise_scenarios
            | model.max_distributed_noise_scenarios,
            doc="Set of Scenarios that are relevant for acoustic considerations, for all other scenarios acoustics are not considered.",
        )

        model.max_throttling_in_max_noise_scenario = pyo.Param(
            model.E_vfc_leaf,
            doc="maximum throttling in leaf vfcs used for maximum distributed noise computation",
        )

        model.max_pressure_rise_in_max_noise_scenario = pyo.Param(
            model.E_fan_station_leaf,
            doc="maximum pressure rise in leaf fan station used for maximum distributed noise computation",
        )

        model.V_room = pyo.Set(
            dimen=1,
            within=model.V,
            doc="Set of target nodes, used for room models",
        )

        model.V_room_in_room_duct = pyo.Set(
            dimen=1,
            within=model.V_room,
            doc="Set of rooms with a ventilation duct going through it",
        )
        model.V_room_simple = pyo.Set(
            dimen=1,
            within=model.V_room - model.V_room_in_room_duct,
            doc="Set of rooms with an air diffuser where the distance to the air diffuser plays no role (Q/4/pi/r^2 << 4/A)",
        )
        model.V_room_normal = pyo.Set(
            dimen=1,
            within=model.V_room - model.V_room_in_room_duct - model.V_room_simple,
            doc="Set of rooms with an air diffuser where the distance to the air diffuser plays a role",
        )

        model.intervals = pyo.RangeSet(
            8, doc="8 intervals of the 8 different octave bands"
        )

        model.level_add_polyhedral_coeff_set = pyo.RangeSet(
            3, doc="set used for the coefficients of the polyhedral approximation"
        )

        model.level_add_polyhedral_approx_slope = pyo.Param(
            model.level_add_polyhedral_coeff_set,
            initialize={1: -0.415, 2: -0.219, 3: -0.066},
            doc="slope of the linear polyhedral approximation of the level rise",
        )

        model.level_add_polyhedral_approx_y_intercept = pyo.Param(
            model.level_add_polyhedral_coeff_set,
            initialize={1: 2.943, 2: 2.288, 3: 1.056},
            doc="y intercept of the linear polyhedral approximation of the level rise",
        )

        model.room_level_add_set = pyo.RangeSet(
            7, doc="set used for tournament style level addition"
        )

        model.room_area = pyo.Param(model.V_room, doc="Area of room v")

        model.room_height = pyo.Param(model.V_room, doc="Height of room v")

        model.room_reverberation_time = pyo.Param(
            model.V_room, doc="Reverberation time of room v"
        )

        model.number_of_air_diffusers = pyo.Param(
            model.V_room_simple | model.V_room_normal,
            doc="Number of air diffusers in the room",
        )

        model.min_distance_to_airdiffuser = pyo.Param(
            model.V_room_normal, doc="minimal distance to an air diffuser"
        )

        model.location_of_sound_source_opening = pyo.Param(
            model.V_room_normal,
            doc="location of the sound source opening acc. to VDI 2081",
        )

        model.angle_of_radiation = pyo.Param(
            model.V_room_normal, doc="angle of radiation acc. to VDI 2081"
        )

        model.area_of_outlet = pyo.Param(
            model.V_room_normal, doc="area of outlet acc. to VDI 2081"
        )

        model.in_room_duct_height = pyo.Param(
            model.V_room_in_room_duct,
            doc="height of in room duct in m acc. to VDI 2081",
        )

        model.in_room_duct_width = pyo.Param(
            model.V_room_in_room_duct, doc="width of in room duct in m acc. to VDI 2081"
        )

        model.in_room_duct_length = pyo.Param(
            model.V_room_in_room_duct,
            doc="length of in room duct in m acc. to VDI 2081",
        )

        model.in_room_duct_thickness = pyo.Param(
            model.V_room_in_room_duct,
            doc="thickness of in room duct in mm acc. to VDI 2081",
        )

        model.solid_angle_index = pyo.Param(
            model.V_room_in_room_duct,
            doc="solid angle index of in room duct acc. to VDI 2081",
        )

        def compute_power_to_pressure_conversion_term(model, v, f):
            f_m = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000])

            volume = model.room_height[v] * model.room_area[v]

            if v in model.V_room_simple:
                Lw_reduc = (
                    -10 * np.log10(volume)
                    + 10 * np.log10(model.room_reverberation_time[v])
                    + 10 * np.log10(model.number_of_air_diffusers[v])
                    + 14
                ) * np.ones(8)

            elif v in model.V_room_normal:
                if model.angle_of_radiation[v] == 0:
                    if model.location_of_sound_source_opening[v] == 1:
                        B1, B2, x0, p = 0.73, 7.62, 158.51, 1.29
                    elif model.location_of_sound_source_opening[v] == 2:
                        B1, B2, x0, p = 1.7, 7.88, 121.19, 1.28
                    elif model.location_of_sound_source_opening[v] == 3:
                        B1, B2, x0, p = 3.9, 8.28, 133.97, 1.27
                    elif model.location_of_sound_source_opening[v] == 4:
                        B1, B2, x0, p = 7.28, 9.42, 298.46, 0.37
                if model.angle_of_radiation[v] == 45:
                    if model.location_of_sound_source_opening[v] == 1:
                        B1, B2, x0, p = 0.84, 4.01, 213.89, 1.1
                    elif model.location_of_sound_source_opening[v] == 2:
                        B1, B2, x0, p = 1.9, 4.16, 221.72, 1.25
                    elif model.location_of_sound_source_opening[v] == 3:
                        B1, B2, x0, p = 3.78, 5.23, 3957.71, 1.38
                    elif model.location_of_sound_source_opening[v] == 4:
                        B1, B2, x0, p = 8.35, 4.42, 42.28, 1.75

                Q = B2 + (B1 - B2) / (
                    1 + (f_m * np.sqrt(model.area_of_outlet[v]) / x0) ** p
                )

                Lw_reduc = 10 * np.log10(
                    Q / (4 * np.pi * model.min_distance_to_airdiffuser[v] ** 2)
                    + 4
                    / 0.163
                    / volume
                    * model.room_reverberation_time[v]
                    * model.number_of_air_diffusers[v]
                )

            elif v in model.V_room_in_room_duct:

                B = np.where(f_m < 4000, 12 * np.log10(f_m), 43)
                h = model.in_room_duct_thickness[v]
                R_ia = 20 * np.log10(h) + B

                K_0 = model.solid_angle_index[v]
                A_2 = 0.163 * volume / model.room_reverberation_time[v]

                S_1 = model.in_room_duct_height[v] * model.in_room_duct_width[v]
                S_k = (
                    2
                    * model.in_room_duct_length[v]
                    * (model.in_room_duct_height[v] + model.in_room_duct_width[v])
                )

                Lw_reduc = -R_ia + 10 * np.log10(S_k / (S_1 * A_2)) + K_0 + 6
            else:
                raise ValueError("Value not within range")

            return Lw_reduc[f - 1]

        model.power_to_pressure_conversion_term = pyo.Param(
            model.V_room,
            model.intervals,
            within=pyo.Reals,
            initialize=compute_power_to_pressure_conversion_term,
            doc="room dependend sound power level to sound pressure level conversion factor. L_p = L_w + power_to_pressure_conversion_term",
        )

        model.max_sound_power_level = pyo.Param(
            doc="max (octave) sound power level occurring anywhere in the system."
        )

        model.max_sound_pressure_level_room = pyo.Param(
            model.V_room,
            doc="maximum allowed A weighted sound pressure level in a room",
        )

        model.duct_zeta = pyo.Param(model.E_duct, doc="duct zeta value of edge (i,j)")

        model.crosstalk_frequency_band = pyo.Set(
            doc="frequency band for crosstalking, i.e. five frequency octave bands",
            initialize={None: [2, 3, 4, 5, 6]},
        )

        model.crosstalk_nodes = pyo.Set(
            within=model.V_room,
            doc="crosstalk nodes of all rooms where crosstalking is accounted for",
        )

        model.crosstalk_node_pairs = pyo.Set(
            within=model.V_room * model.V_room,
            doc="room node pairs of neighbouring rooms where crosstalk is accounted for",
        )

        model.crosstalk_dampening_comparison_through_wall = pyo.Param(
            model.crosstalk_frequency_band,
            doc="dampening of the wall between two rooms, needed for comparing crosstalk_frequency_band",
        )

        model.crosstalk_dampening_duct = pyo.Param(
            model.crosstalk_frequency_band,
            model.crosstalk_node_pairs,
            doc="dampening of the duct between crosstalk rooms, already ",
        )

        model.crosstalk_dampening_source_room = pyo.Param(
            model.crosstalk_nodes,
            initialize=lambda m, v: 10 * np.log10(m.room_area[v] / 4),
            doc="dampening at the source room, used for crosstalk computation",
        )

        model.crosstalk_dampening_receiving_room = pyo.Param(
            model.crosstalk_nodes,
            initialize=lambda m, v: 10 * np.log10(1 / m.area_of_outlet[v]),
            doc="dampening at the receiving room, used for crosstalk computation",
        )

        model.partition_wall_area = pyo.Param(
            model.crosstalk_node_pairs,
            doc="Area of the partition wall between the two rooms",
        )

        @model.Param(
            model.crosstalk_frequency_band,
            model.crosstalk_node_pairs,
            doc="missing dampening according to crosstalk, if >= 0 then dampening has to be added by silencers",
        )
        def crosstalk_damping_margin(model, f, v1, v2):

            def damping_room(i, j):
                equivalent_sound_absorption_area = (
                    0.163
                    * model.room_area[j]
                    * model.room_height[j]
                    / model.room_reverberation_time[j]
                )

                return (
                    model.crosstalk_dampening_receiving_room[j]
                    + model.crosstalk_dampening_source_room[i]
                    + 10
                    * np.log10(partition_wall_area / equivalent_sound_absorption_area)
                    + crosstalk_damping_duct
                )

            partition_wall_area = model.partition_wall_area[v1, v2]
            crosstalk_damping_duct = model.crosstalk_dampening_duct[f, v1, v2]

            return (
                -min(damping_room(v1, v2), damping_room(v2, v1))
                + 10
                + model.crosstalk_dampening_comparison_through_wall[f]
            )

        # vfc
        model.variable_vfc_reg_coef_cost = pyo.Param(
            pyo.RangeSet(2), doc="coefficients of variable vfc's linear cost model"
        )
        model.constant_vfc_reg_coef_cost = pyo.Param(
            pyo.RangeSet(2), doc="coefficients of constant vfc's linear cost model"
        )

        model.vfc_reg_coef_cost = pyo.Param(
            pyo.RangeSet(2),
            within=pyo.Reals,
            doc="coefficients of linear vfc cost model",
            initialize=lambda m, i: (
                m.variable_vfc_reg_coef_cost[i]
                if variable_air_volume
                else m.constant_vfc_reg_coef_cost[i]
            ),
        )

        model.vfc_reg_coef_cost_acoustic_cladding = pyo.Param(
            pyo.RangeSet(2),
            within=pyo.Reals,
            doc="coefficients of silencer pressure loss model",
            initialize={1: 2080, 2: 190},
        )

        model.vfc_reg_coef_flow_noise = pyo.Param(
            model.intervals,
            pyo.RangeSet(6),
            within=pyo.Reals,
            doc="coefficients of linear regression model",
        )

        model.vfc_height = pyo.Param(model.E_vfc, doc="height of vfc (i,j)")
        model.vfc_width = pyo.Param(model.E_vfc, doc="width of vfc (i,j)")

        model.has_acoustic_cladding = pyo.Param(
            model.E_vfc,
            within=pyo.Binary,
            initialize=0,
            doc="defines whether the volume flow controller has an acoustic cladding that dams the sound through the walls.",
        )

        model.vfc_costs_expr = pyo.Expression(
            model.E_vfc,
            rule=lambda m, i, j: m.vfc_reg_coef_cost[1]
            * (m.vfc_height[i, j] * m.vfc_width[i, j])
            + m.vfc_reg_coef_cost[2]
            + m.has_acoustic_cladding[i, j]
            * (
                m.vfc_reg_coef_cost_acoustic_cladding[1]
                * (m.vfc_height[i, j] * m.vfc_width[i, j])
                + m.vfc_reg_coef_cost_acoustic_cladding[2]
            ),
            doc="cost for acoustic cladding of the VFC so that it can be placed close to a room",
        )

        # silencer

        model.silencer_reg_coef_pressure = pyo.Param(
            pyo.RangeSet(3),
            within=pyo.Reals,
            doc="coefficients of silencer pressure loss model",
        )

        model.silencer_reg_coef_cost = pyo.Param(
            pyo.RangeSet(4),
            within=pyo.Reals,
            doc="coefficients of linear silencer cost model",
        )

        model.silencer_reg_coef_dampening = pyo.Param(
            model.intervals,
            pyo.RangeSet(8),
            within=pyo.Reals,
            doc="coefficients of silencer dampening model",
        )

        model.silencer_reg_coef_flow_noise = pyo.Param(
            model.intervals,
            pyo.RangeSet(5),
            within=pyo.Reals,
            doc="coefficients of silencer flow noise model",
        )

        model.silencer_height = pyo.Param(
            model.E_silencer, doc="height of silencer (i,j)"
        )
        model.silencer_width = pyo.Param(
            model.E_silencer, doc="width of silencer (i,j)"
        )

        model.silencer_length_min = pyo.Param(
            model.E_silencer,
            within=pyo.PositiveReals,
            doc="minimal length of silencer (i,j)",
        )

        model.silencer_length_max = pyo.Param(
            model.E_silencer,
            within=pyo.PositiveReals,
            doc="maximal length of silencer (i,j)",
        )

        model.min_n_splitters = pyo.Param(
            model.E_silencer,
            within=pyo.PositiveIntegers,
            doc="min number of splitters",
        )

        model.max_n_splitters_raw = pyo.Param(
            model.E_silencer,
            within=pyo.PositiveIntegers,
            doc="maximal number of splitters",
        )

        def compute_tighter_silencer_bounds(model, i, j, k):

            possible_n_splitter = np.arange(
                model.min_n_splitters[i, j], model.max_n_splitters_raw[i, j]
            )

            possible_splitter_width = (
                model.silencer_width[i, j] / possible_n_splitter - 0.2
            )

            min_splitter_width = np.min(
                possible_splitter_width[possible_splitter_width > 0]
            )

            max_splitter_width = (
                model.silencer_width[i, j] / model.min_n_splitters[i, j] - 0.2
            )

            max_n_splitters = possible_n_splitter[
                np.argmin(possible_splitter_width[possible_splitter_width > 0])
            ]

            if k == "min_splitter_width":
                return np.round(min_splitter_width, 5)
            elif k == "max_splitter_width":
                return np.round(max_splitter_width, 5)
            elif k == "max_n_splitters":
                return np.round(max_n_splitters, 5)

            n = np.arange(model.min_n_splitters[i, j], max_n_splitters + 1)

            s = model.silencer_width[i, j] / n - 0.2

            A = model.silencer_height[i, j] * n * s

            a = model.silencer_reg_coef_pressure

            silencer_zeta_max = np.max(
                1
                / A**2
                * (a[1] + a[2] / s + a[3] * model.silencer_length_max[i, j] / s)
            )
            # silencer_zeta_max = np.round(silencer_zeta_max,5)
            silencer_zeta_min = np.min(
                1
                / A**2
                * (a[1] + a[2] / s + a[3] * model.silencer_length_min[i, j] / s)
            )
            # silencer_zeta_min = np.round(silencer_zeta_min,5)
            if k == "silencer_zeta_max":
                return silencer_zeta_max
            elif k == "silencer_zeta_min":
                return silencer_zeta_min

        model.min_splitter_width = pyo.Param(
            model.E_silencer,
            within=pyo.PositiveReals,
            initialize=lambda m, i, j: compute_tighter_silencer_bounds(
                m, i, j, "min_splitter_width"
            ),
            doc="minimal width of splitter elements of silencer (i,j)",
        )

        model.max_splitter_width = pyo.Param(
            model.E_silencer,
            within=pyo.PositiveReals,
            initialize=lambda m, i, j: compute_tighter_silencer_bounds(
                m, i, j, "max_splitter_width"
            ),
            doc="maximal width of splitter elements of silencer (i,j)",
        )

        model.max_n_splitters = pyo.Param(
            model.E_silencer,
            within=pyo.PositiveIntegers,
            initialize=lambda m, i, j: compute_tighter_silencer_bounds(
                m, i, j, "max_n_splitters"
            ),
            doc="real maximal number of splitters, limited due to silencer width",
        )

        model.silencer_zeta_max = pyo.Param(
            model.E_silencer,
            within=pyo.PositiveReals,
            initialize=lambda m, i, j: compute_tighter_silencer_bounds(
                m, i, j, "silencer_zeta_max"
            ),
            doc="max zeta value is multiplied by the volume flow to obtain the max pressure loss",
        )

        model.silencer_zeta_min = pyo.Param(
            model.E_silencer,
            within=pyo.PositiveReals,
            initialize=lambda m, i, j: compute_tighter_silencer_bounds(
                m, i, j, "silencer_zeta_min"
            ),
            doc="min zeta value is multiplied by the volume flow^2 to obtain the minimal pressure loss",
        )

        model.number_of_splitters = pyo.Var(
            model.E_silencer,
            within=pyo.PositiveIntegers,
            bounds=lambda m, i, j: (m.min_n_splitters[i, j], m.max_n_splitters[i, j]),
            doc="number of splitters (Kulissen) of the silencer",
        )

        model.silencer_costs_max = pyo.Expression(
            model.E_silencer,
            expr=lambda m, i, j: float(
                m.silencer_reg_coef_cost[1] * m.max_n_splitters[i, j]
                + m.silencer_reg_coef_cost[2]
                * m.silencer_length_max[i, j]
                * m.silencer_width[i, j]
                + m.silencer_reg_coef_cost[3]
                * m.silencer_width[i, j]
                * m.silencer_height[i, j]
                + m.silencer_reg_coef_cost[4]
            ),
            doc="maximum costs, calculated for this specific splitter silencer",
        )

        model.silencer_costs_min = pyo.Expression(
            model.E_silencer,
            expr=lambda m, i, j: float(
                m.silencer_reg_coef_cost[1] * m.min_n_splitters[i, j]
                + m.silencer_reg_coef_cost[2]
                * m.silencer_length_min[i, j]
                * m.silencer_width[i, j]
                + m.silencer_reg_coef_cost[3]
                * m.silencer_width[i, j]
                * m.silencer_height[i, j]
                + m.silencer_reg_coef_cost[4]
            ),
            doc="min costs, calculated for this specific splitter silencer",
        )

        model.silencer_dampening_max = pyo.Param(
            model.intervals,
            initialize={1: 7, 2: 16, 3: 29, 4: 50, 5: 50, 6: 50, 7: 44, 8: 39},
            doc="maximum dampening of all possible silencers.",
        )

        model.silencer_costs_intermediate = pyo.Var(
            model.E_silencer,
            within=pyo.NonNegativeReals,
            bounds=lambda m, i, j: (
                m.silencer_costs_min[i, j].expr(),
                m.silencer_costs_max[i, j].expr(),
            ),
        )

        model.silencer_costs = pyo.Var(
            model.E_silencer,
            within=pyo.NonNegativeReals,
            bounds=lambda m, i, j: (0, m.silencer_costs_max[i, j].expr()),
        )

        model.silencer_length = pyo.Var(
            model.E_silencer,
            within=pyo.NonNegativeReals,
            bounds=lambda m, i, j: (
                m.silencer_length_min[i, j],
                m.silencer_length_max[i, j],
            ),
            doc="length of the silencer",
        )

        model.splitter_width = pyo.Var(
            model.E_silencer,
            within=pyo.PositiveReals,
            bounds=lambda m, i, j: (
                m.min_splitter_width[i, j],
                m.max_splitter_width[i, j],
            ),
            doc="width of gap between splitters: s = B/n-T <=> B = n*(T+s)",
        )

        @model.Constraint(model.E_silencer)
        def silencer_costs_def(model, i, j):
            return (
                model.silencer_costs_intermediate[i, j]
                == model.silencer_reg_coef_cost[1] * model.number_of_splitters[i, j]
                + model.silencer_reg_coef_cost[2]
                * model.silencer_length[i, j]
                * model.silencer_width[i, j]
                + model.silencer_reg_coef_cost[3]
                * model.silencer_height[i, j]
                * model.silencer_width[i, j]
                + model.silencer_reg_coef_cost[4]
            )

        @model.Constraint(model.E_silencer)
        def splitter_width_definition(model, i, j):
            return (model.splitter_width[i, j] + 0.2) * model.number_of_splitters[
                i, j
            ] == model.silencer_width[i, j]
            # this is equal to  (s+T)*n = B > 0

        model.silencer_nonlinear_term_set = pyo.Set(
            initialize=range(1, 4),
            doc="set for indexing the quadratic polynomials representing higher order polynomials",
        )

        def lb(model, i, j, k):
            """lower bound of the quadratic polynomials, similar to Cormick"""
            if k == 1:

                return (
                    model.silencer_height[i, j]
                    * model.min_n_splitters[i, j]
                    * model.min_splitter_width[i, j]
                )
            if k == 2:
                return (
                    model.min_n_splitters[i, j]
                    * model.min_splitter_width[i, j]
                    * model.silencer_height[i, j]
                ) ** 2
            if k == 3:
                return (
                    model.min_n_splitters[i, j]
                    * model.min_splitter_width[i, j]
                    * model.silencer_height[i, j]
                ) ** 2 * model.min_splitter_width[i, j]

            raise IndexError

        def ub(model, i, j, k):
            try:
                if k == 1:
                    return (
                        model.silencer_height[i, j]
                        * model.max_n_splitters[i, j]
                        * model.max_splitter_width[i, j]
                    )
                if k == 2:
                    return (
                        model.max_n_splitters[i, j]
                        * model.max_splitter_width[i, j]
                        * model.silencer_height[i, j]
                    ) ** 2
                if k == 3:
                    return (
                        model.max_n_splitters[i, j]
                        * model.max_splitter_width[i, j]
                        * model.silencer_height[i, j]
                    ) ** 2 * model.max_splitter_width[i, j]

            except ZeroDivisionError:
                return None
            raise IndexError

        def bounder(model, i, j, k):
            return (lb(model, i, j, k), ub(model, i, j, k))

        model.silencer_nonlinear_term = pyo.Var(
            model.E_silencer,
            model.silencer_nonlinear_term_set,
            domain=pyo.NonNegativeReals,
            bounds=bounder,
        )

        @model.Constraint(model.E_silencer, model.silencer_nonlinear_term_set)
        def w_helper(model, i, j, k):
            if k == 1:  # Area1
                return (
                    model.silencer_nonlinear_term[i, j, 1]
                    == model.number_of_splitters[i, j]
                    * model.splitter_width[i, j]
                    * model.silencer_height[i, j]
                )
            if k == 2:  # Area^2
                return (
                    model.silencer_nonlinear_term[i, j, 2]
                    == model.silencer_nonlinear_term[i, j, 1] ** 2
                )
            if k == 3:  # Area^2 *s
                return (
                    model.silencer_nonlinear_term[i, j, 3]
                    == model.silencer_nonlinear_term[i, j, 2]
                    * model.splitter_width[i, j]
                )
            raise IndexError

        model.silencer_zeta = pyo.Var(
            model.E_silencer,
            doc="silencer zeta value per edge (i,j) w.r.t. volume flow^2",
            bounds=lambda m, i, j: (
                m.silencer_zeta_min[i, j],
                m.silencer_zeta_max[i, j],
            ),
        )

        @model.Constraint(
            model.E_silencer,
            doc="zeta value characteristic of silencer, w.r.t. volume_flow^2",
        )
        def performance_curve_zeta_definition(model, i, j):
            performance_curve = model.silencer_zeta[
                i, j
            ] * model.silencer_nonlinear_term[i, j, 3] == (
                model.silencer_reg_coef_pressure[1] * model.splitter_width[i, j]
                + model.silencer_reg_coef_pressure[2]
                + model.silencer_reg_coef_pressure[3] * model.silencer_length[i, j]
            )
            return performance_curve

    # Finances

    model.electric_energy_costs = pyo.Param(doc="Costs of electric energy in € / Wh")
    model.operating_years = pyo.Param(
        doc="Number of years the plant is designed to be operating"
    )
    model.operating_days_per_year = pyo.Param(
        doc="Number of days in a year the plant is operating"
    )
    model.operating_hours_per_day = pyo.Param(
        doc="Number of hours in a day the plant is operating"
    )

    model.component_names = pyo.Set(
        initialize=["fan", "vfc", "duct", "silencer", "measurement"],
        doc="Component names used for computing investment costs according to VDI 2067",
    )

    model.price_change_factor_service_maintenance = pyo.Param(
        within=pyo.PositiveReals,
        doc="Price change factor of the service and maintenance acc. to VDI 2067",
    )

    model.price_change_factor_electricity = pyo.Param(
        within=pyo.PositiveReals,
        doc="Price change factor of the energy costs acc. to VDI 2067",
    )

    model.maintenance_factor = pyo.Param(
        model.component_names,
        within=pyo.NonNegativeReals,
        doc="Factor for the cost of maintenance acc. to VDI 2067",
        initialize={
            "fan": 0.03,
            "vfc": 0.03,
            "duct": 0.02,
            "silencer": 0.01,
            "measurement": 0.02,
        },
    )

    model.service_factor = pyo.Param(
        model.component_names,
        within=pyo.NonNegativeReals,
        doc="Factor for the cost of service acc. to VDI 2067",
        initialize={
            "fan": 0.01,
            "vfc": 0.01,
            "duct": 0,
            "silencer": 0,
            "measurement": 0,
        },
    )

    model.deprecation_period = pyo.Param(
        model.component_names,
        within=pyo.PositiveIntegers,
        doc="Number of years of the deprecation period acc. to VDI 2067",
        initialize={
            "fan": 12,
            "vfc": 12,
            "duct": 30,
            "silencer": 20,
            "measurement": 10,
        },
    )

    model.interest_rate = pyo.Param(
        within=pyo.NonNegativeReals,
        doc="Interest factor of price change acc. to VDI 2067",
    )

    model.ind_purchase = pyo.Var(
        model.E_fan_station | model.E_vfc | model.E_silencer,
        within=pyo.Binary,
        doc="purchase indicator variable for fan station and VFC.",
    )

    if planning_mode == "Topology":

        model.constant_vfc_costs = pyo.Param(
            doc="Costs for a constantly operating volume flow controller"
        )

        model.variable_vfc_costs = pyo.Param(
            doc="Costs for a volume flow controller that operates according to demand"
        )

        model.vfc_costs = pyo.Param(
            doc="Costs of a single VFC. Assumed to be independend on height and width of the respective duct.",
            initialize=(
                model.variable_vfc_costs
                if variable_air_volume
                else model.constant_vfc_costs
            ),
        )

        model.silencer_zeta = pyo.Param(
            model.E_silencer, initialize=0, doc="Fixed zeta value per fixed edge (i,j)"
        )

        if duct_model == 1:
            model.duct_friction_hyperplanes_set = pyo.Set(
                doc="Set of duct friction hyperplanes. Used for outer polyhedral approximation of (h+w)/(h^3*w^3)"
            )

            model.duct_friction_hyperplanes_specific_pre_set = pyo.Set(
                model.E_duct,
                doc="Pre set of duct friction hyperplanes at a specific edge. Used for outer polyhedral approximation of (h+w)/(h^3*w^3)",
            )

            def duct_friction_hyperplanes_specific_creator(model):
                pairs = []
                for i, j in model.E_duct:
                    for x in model.duct_friction_hyperplanes_specific_pre_set[i, j]:
                        pairs.append((i, j, x))
                return pairs

            model.duct_friction_hyperplanes_specific_set = pyo.Set(
                dimen=3,
                initialize=duct_friction_hyperplanes_specific_creator,
                doc="Set of duct friction hyperplanes at a specific edge. Used for outer polyhedral approximation of (h+w)/(h^3*w^3)",
            )

            model.duct_area2_hyperplanes_set = pyo.Set(
                doc="Set of duct area^2 hyperplanes. Used for outer polyhedral approximation of 1/(h^2b^2)"
            )

            model.duct_area2_hyperplanes_specific_pre_set = pyo.Set(
                model.E_duct,
                doc="Pre set of duct area^2 hyperplanes at a specific edge. Used for outer polyhedral approximation of 1/(h^2b^2)",
            )

            def duct_area2_hyperplanes_specific_creator(model):
                pairs = []
                for i, j in model.E_duct:
                    for x in model.duct_area2_hyperplanes_specific_pre_set[i, j]:
                        pairs.append((i, j, x))
                return pairs

            model.duct_area2_hyperplanes_specific_set = pyo.Set(
                dimen=3,
                initialize=duct_area2_hyperplanes_specific_creator,
                doc="Set of duct area2 hyperplanes at a specific edge. Used Used for outer polyhedral approximation of 1/(h^2b^2)",
            )

            model.duct_resistance_coefficient = pyo.Param(
                doc="resistance coefficient lambda of Darcy-Weisbach equation"
            )

            model.duct_area_costs = pyo.Param(
                doc="Costs per square meter duct in € / m^2."
            )

            model.duct_width_min = pyo.Param(
                model.E_duct, doc="Minimum duct width in m"
            )
            model.duct_width_max = pyo.Param(
                model.E_duct, doc="Maximum duct width in m"
            )

            model.duct_height_min = pyo.Param(
                model.E_duct, doc="Minimum duct height in m"
            )
            model.duct_height_max = pyo.Param(
                model.E_duct, doc="Maximum duct height in m"
            )

            model.duct_length = pyo.Param(model.E_duct, doc="Duct length in m")

            model.n_duct_bendings = pyo.Param(
                model.E_duct,
                initialize=0,
                doc="Number of bendings in duct (i,j), used for pressure loss calculation.",
            )

            model.duct_t_branch_node = pyo.Set(
                within=model.V,
                doc="Set of nodes that are center of a T-branch, used for pressure loss calculation.",
            )
            model.duct_e_branch = pyo.Set(
                within=model.V * model.V * model.V,
                doc="Set of node triples (k,l,m) that are part of a branching where branch l goes straight and branch m bends.",
            )

            model.zeta_bending_val = pyo.Param(
                initialize=0.0675, doc="Fixed zeta value for all bendings."
            )
            model.zeta_t_branch_val = pyo.Param(
                initialize=0.094, doc="Fixed zeta value for T-branches added to edges."
            )
            model.zeta_e_branch_straight_val = pyo.Param(
                initialize=0.17,
                doc="Fixed zeta value of the straight branch of an e-branch",
            )
            model.zeta_e_branch_bend_val = pyo.Param(
                initialize=0.75,
                doc="Fixed zeta value of the bend branch of an e-branch",
            )

            model.duct_friction_slope_width = pyo.Param(
                model.duct_friction_hyperplanes_set,
                doc="Slope in width direction of the friction hyperplanes \
                    for outer polyhedral approximation",
            )

            model.duct_friction_slope_height = pyo.Param(
                model.duct_friction_hyperplanes_set,
                doc="Slope in height direction of the friction hyperplanes \
                    for outer polyhedral approximation",
            )

            model.duct_friction_intercept = pyo.Param(
                model.duct_friction_hyperplanes_set,
                doc="Intercept of the friction hyperplanes\
                    for outer polyhedral approximation",
            )

            model.duct_area2_slope_width = pyo.Param(
                model.duct_area2_hyperplanes_set,
                doc="Slope in width direction of the area^2 hyperplanes\
                    for outer polyhedral approximation",
            )

            model.duct_area2_slope_height = pyo.Param(
                model.duct_area2_hyperplanes_set,
                doc="Slope in height direction of the area^2 hyperplanes\
                    for outer polyhedral approximation",
            )

            model.duct_area2_intercept = pyo.Param(
                model.duct_area2_hyperplanes_set,
                doc="Intercept of the area^2 hyperplanes\
                    for outer polyhedral approximation",
            )

            model.duct_width = pyo.Var(
                model.E_duct,
                bounds=lambda model, i, j: (
                    model.duct_width_min[i, j],
                    model.duct_width_max[i, j],
                ),
                doc="Duct width of duct (i,j)",
            )

            model.duct_height = pyo.Var(
                model.E_duct,
                bounds=lambda model, i, j: (
                    model.duct_height_min[i, j],
                    model.duct_height_max[i, j],
                ),
                doc="Duct height of duct (i,j)",
            )

            model.fun_nonlinear_duct_hb_friction = pyo.Var(
                model.E_duct,
                within=pyo.NonNegativeReals,
                bounds=lambda model, i, j: (
                    (model.duct_width_max[i, j] + model.duct_height_max[i, j])
                    / (model.duct_width_max[i, j] * model.duct_height_max[i, j]) ** 3,
                    (model.duct_width_min[i, j] + model.duct_height_min[i, j])
                    / (model.duct_width_min[i, j] * model.duct_height_min[i, j]) ** 3,
                ),
                doc="Friction term (w+h)/w^3h^3 of duct (i,j)",
            )

            model.fun_duct_nonlinear_hb_area2 = pyo.Var(
                model.E_duct,
                within=pyo.NonNegativeReals,
                bounds=lambda model, i, j: (
                    1 / (model.duct_width_max[i, j] * model.duct_height_max[i, j]) ** 2,
                    1 / (model.duct_width_min[i, j] * model.duct_height_min[i, j]) ** 2,
                ),
                doc="Area^2 term 1/w^2h^2 of duct (i,j)",
            )

            @model.Expression(
                model.duct_friction_hyperplanes_specific_set,
                doc="Hyperplane expression approximating (h+b)/h^3/b^3",
            )
            def duct_friction_hyperplanes(model, i, j, t):
                return (
                    model.duct_friction_slope_width[t] * model.duct_width[i, j]
                    + model.duct_friction_slope_height[t] * model.duct_height[i, j]
                    + model.duct_friction_intercept[t]
                )

            @model.Expression(
                model.duct_area2_hyperplanes_specific_set,
                doc="Hyperplane expression approximating 1/h^2/b^2",
            )
            def duct_area2_hyperplanes(model, i, j, t):
                return (
                    model.duct_area2_slope_width[t] * model.duct_width[i, j]
                    + model.duct_area2_slope_height[t] * model.duct_height[i, j]
                    + model.duct_area2_intercept[t]
                )

            @model.Constraint(
                model.duct_friction_hyperplanes_specific_set,
                doc="Friction term must be larger than respective hyperplanes",
            )
            def duct_friction_outer_polyhedral_approx(model, i, j, t):
                return (
                    model.fun_nonlinear_duct_hb_friction[i, j]
                    >= model.duct_friction_hyperplanes[i, j, t]
                )

            @model.Constraint(
                model.duct_area2_hyperplanes_specific_set,
                doc="Area^2 term must be larger than respective hyperplanes",
            )
            def duct_area2_outer_polyhedral_approx(model, i, j, t):
                return (
                    model.fun_duct_nonlinear_hb_area2[i, j]
                    >= model.duct_area2_hyperplanes[i, j, t]
                )

            @model.Param(model.E_duct, doc="Zeta value of duct-t-branch = 0.7")
            def zeta_t_branch(model, i, j):
                if i in model.duct_t_branch_node:
                    return model.zeta_t_branch_val
                return 0

            @model.Param(
                model.E_duct,
                doc="zeta value of a E branch where one flow goes straight and one bends. (k,l) goes straight and the (k,m) bends.",
            )
            def zeta_e_branch(model, i, j):
                for k, l, m in model.duct_e_branch:
                    if k == i and l == j:
                        return model.zeta_e_branch_straight_val
                    if k == i and m == j:
                        return model.zeta_e_branch_bend_val
                return 0

            @model.Param(model.E_duct, doc="redundancy with n_duct_bendings")
            def zeta_bending(model, i, j):
                return model.n_duct_bendings[i, j] * model.zeta_bending_val

            @model.Constraint(model.E_duct, doc="limit height to width ratio")
            def limit_height_to_width_ratio1(model, i, j):
                return model.duct_width[i, j] >= 1 / 3 * model.duct_height[i, j]

            @model.Constraint(model.E_duct, doc="limit height to width ratio")
            def limit_height_to_width_ratio2(model, i, j):
                return model.duct_height[i, j] >= 1 / 3 * model.duct_width[i, j]

    # FAN MODEL
    if fan_model == 1:

        model.max_num_fans_per_fan_station = pyo.Param(
            model.E_fan_station,
            doc="Maximum number of fans that can be placed in a fan station",
        )

        model.leaf_component_decision = pyo.Var(
            within=pyo.Binary,
            doc="Decision variable that ensures that either in front of *all* rooms/zones a fan station or a VFC is placed. Additional fans or VFCs are not hindered.",
        )

        @model.Constraint(
            model.E_fan_station_leaf,
            doc="Decision whether fan stations are purchased in front of *all* rooms/zones",
        )
        def leaves_all_fan_stations_or_all_vfcs_purchased_a(model, i, j):
            return model.ind_purchase[i, j] >= model.leaf_component_decision

        @model.Constraint(
            model.E_vfc_leaf,
            doc="Decision whether VFCs are purchased in front of *all* rooms/zones",
        )
        def leaves_all_fan_stations_or_all_vfcs_purchased_b(model, i, j):
            return model.ind_purchase[i, j] >= (1 - model.leaf_component_decision)

        model.fan_product_line = pyo.Set(doc="Set of fan product lines")

        model.fan_diameter = pyo.Set(
            model.fan_product_line, doc="Set of fan diameters in a certain product line"
        )

        def valid_pd_pairs_init(model):
            pairs = []
            for p in model.fan_product_line:
                for d in model.fan_diameter[p]:
                    pairs.append((p, d))
            return pairs

        model.p_d_combination_set = pyo.Set(
            dimen=2,
            initialize=valid_pd_pairs_init,
            doc="Set of (product line, fan diameter) of fans",
        )

        if reduce_fan_curves:
            model.fan_hyperplanes_overestimation_specific_pre_set = pyo.Set(
                model.Scenarios,
                model.E_fan_station,
                model.p_d_combination_set,
                doc="Set of sets of scenarios, E_fan stations and distinct fans for overestimation of fan power loss",
            )

            def fan_hyperplanes_overestimation_specific_creator(model):
                pairs = []
                for s in model.Scenarios:
                    for i, j in model.E_fan_station:
                        for p, d in model.p_d_combination_set:
                            for (
                                x
                            ) in model.fan_hyperplanes_overestimation_specific_pre_set[
                                s, i, j, p, d
                            ]:
                                pairs.append((s, i, j, p, d, x))
                return pairs

            model.fan_hyperplanes_overestimation_specific_set = pyo.Set(
                dimen=6,
                initialize=fan_hyperplanes_overestimation_specific_creator,
                doc="Set of supporting hyperplanes per scenario per fan station per distinct fan (p,d) for overestimation of fan power loss",
            )

        model.fan_hyperplanes_overestimation_pre_set = pyo.Set(
            model.p_d_combination_set,
            doc="Set of sets of distinct fan for overestimation of fan power loss",
        )

        def fan_hyperplanes_overestimation_creator(model):
            pairs = []
            for p, d in model.p_d_combination_set:
                for x in model.fan_hyperplanes_overestimation_pre_set[p, d]:
                    pairs.append((p, d, x))
            return pairs

        model.fan_hyperplanes_overestimation_set = pyo.Set(
            dimen=3,
            initialize=fan_hyperplanes_overestimation_creator,
            doc="Set of supporting hyperplanes per distinct fan (p,d) for overestimation of fan power loss",
        )

        model.fan_n = pyo.Set(
            model.p_d_combination_set, doc="Set of number of distinct fans"
        )

        model.fan_set = pyo.Set(
            doc="Set of (*edge, product line, fan diameter, number) of fans",
        )

        def edge_fan_types_init(m):
            # tuple slicing drops the last element (n)
            return (t[:4] for t in m.fan_set)

        model.edge_fan_types = pyo.Set(
            dimen=4, initialize=edge_fan_types_init, doc="4-tupel of (i,j,p,d)"
        )

        if reduce_fan_curves:
            model.fan_hyperplanes_underestimation_specific_pre_set = pyo.Set(
                model.Scenarios,
                model.E_fan_station,
                model.p_d_combination_set,
                doc="Set of sets of distinct fans for underestimation of fan power loss",
            )

            def fan_hyperplanes_underestimation_specific_set(model):
                fan_hyp_set = []
                for s in model.Scenarios:
                    for i, j in model.E_fan_station:
                        for p, d in model.p_d_combination_set:
                            for (
                                t
                            ) in model.fan_hyperplanes_underestimation_specific_pre_set[
                                s, i, j, p, d
                            ]:
                                fan_hyp_set.append((s, i, j, p, d, t))
                return fan_hyp_set

            model.fan_hyperplanes_underestimation_specific_set = pyo.Set(
                initialize=fan_hyperplanes_underestimation_specific_set,
                doc="Set of supporting hyperplanes per scenario per fan station per distinct fan (p,d) for underestimation of fan power loss",
            )

        model.fan_hyperplanes_underestimation_pre_set = pyo.Set(
            model.p_d_combination_set,
            doc="Set of sets of distinct fans for underestimation of fan power loss",
        )

        def fan_hyperplanes_underestimation_set(model):
            fan_hyp_set = []
            for p, d in model.p_d_combination_set:
                for t in model.fan_hyperplanes_underestimation_pre_set[p, d]:
                    fan_hyp_set.append((p, d, t))
            return fan_hyp_set

        model.fan_hyperplanes_underestimation_set = pyo.Set(
            initialize=fan_hyperplanes_underestimation_set,
            doc="Set of supporting hyperplanes per distinct fan (p,d) for underestimation of fan power loss",
        )

        model.fan_pressure_coefficients = pyo.Param(
            model.p_d_combination_set,
            pyo.RangeSet(3),
            doc="Fan pressure coefficients for dp = a1*q^2 + a2*q*n + a3*n^2 - not used in model, only added for postprocessing",
        )

        model.fan_power_coefficients = pyo.Param(
            model.p_d_combination_set,
            pyo.RangeSet(5),
            doc="Fan power coefficients for pel = b1*q^3 + b2*q^2*n + b3*q*n^2 + b4*n^3 + b5 - not used in model, only added for postprocessing",
        )

        model.fan_hyperplanes_underestimation_slope_volume_flow = pyo.Param(
            model.fan_hyperplanes_underestimation_set,
            doc="Slope in volume flow direction of the fan hyperplanes \
                for outer polyhedral approximation",
        )

        model.fan_hyperplanes_underestimation_slope_pressure = pyo.Param(
            model.fan_hyperplanes_underestimation_set,
            doc="Slope in pressure direction of the fan hyperplanes \
                for outer polyhedral approximation",
        )

        model.fan_hyperplanes_underestimation_intercept = pyo.Param(
            model.fan_hyperplanes_underestimation_set,
            doc="Intercept of the fan curve approx\
                for outer polyhedral approximation",
        )

        model.fan_hyperplanes_overestimation_slope_volume_flow = pyo.Param(
            model.fan_hyperplanes_overestimation_set,
            doc="Slope of overestimating hyperplanes for fan power loss",
        )

        model.fan_hyperplanes_overestimation_intercept = pyo.Param(
            model.fan_hyperplanes_overestimation_set,
            doc="Intercept of overestimating hyperplanes for fan power loss",
        )

        model.fan_power_loss_max = pyo.Param(
            model.p_d_combination_set,
            doc="Maximal electric power consumption of distinct fan (p,d)",
        )

        def calculate_fan_power_loss_max_of_all_fans(model):
            return max(
                model.fan_power_loss_max[p, d] for (p, d) in model.p_d_combination_set
            )

        model.fan_power_loss_max_of_all_fans = pyo.Param(
            initialize=calculate_fan_power_loss_max_of_all_fans,
            doc="Maximal electric power of all fans",
        )

        model.fan_volume_flow_max = pyo.Param(
            model.p_d_combination_set, doc="Maximal volume flow of distinct fan (p,d)"
        )
        model.fan_pressure_max = pyo.Param(
            model.p_d_combination_set, doc="Maximal pressure rise of distinct fan (p,d)"
        )

        model.fan_rotational_speed_max = pyo.Param(
            model.p_d_combination_set,
            doc="Maximal rotational speed of distinct fan (p,d). Only used for postprocessing",
        )

        model.fan_costs = pyo.Param(
            model.p_d_combination_set, doc="Cost of distinct fan (p,d)"
        )

        model.fan_ind_purchase = pyo.Var(
            model.fan_set,
            within=pyo.Binary,
            doc="Purchase indicator for fan (p,d,n) in fan_station (i,j)",
        )

        @model.Constraint(
            model.E_fan_station, doc="Limit number of fans per fan station"
        )
        def only_n_fans_per_fan_station(model, i, j):
            return (
                sum(
                    model.fan_ind_purchase[i, j, p, d, n]
                    for (k, l, p, d, n) in model.fan_set
                    if (i, j) == (k, l)
                )
                <= model.max_num_fans_per_fan_station[i, j]
            )

        def fan_types_on_edge_init(m, i, j):
            return {
                (p, d) for (ii, jj, p, d) in m.edge_fan_types if ii == i and jj == j
            }

        model.fan_types_on_edge = pyo.Set(
            model.E_fan_station,
            dimen=2,
            initialize=fan_types_on_edge_init,
            doc="indexed set of (p,d) with index (i,j) in E",
        )

        def init_fan_station_single(m):
            return {
                (i, j)
                for (i, j) in m.E_fan_station
                if len(m.fan_types_on_edge[i, j]) == 1
            }

        model.E_fan_station_single = pyo.Set(
            dimen=2,
            initialize=init_fan_station_single,
            doc="set of fan station edges where only one different kind of fans (p,d) can be placed",
        )
        model.E_fan_station_multi = pyo.Set(
            dimen=2, initialize=model.E_fan_station - model.E_fan_station_single
        )
        if planning_mode == "Configuration":

            model.fan_flow_noise_coefficients = pyo.Param(
                model.p_d_combination_set,
                model.intervals,
                pyo.RangeSet(3),
                within=pyo.Reals,
                doc="coefficients of regression model: ",
            )

            level_increaser = {i: 10 * np.log10(i) for i in range(1, 11)}

            model.max_n_elements = pyo.RangeSet(10)

            model.identical_fan_level_increase = pyo.Param(
                model.max_n_elements,
                initialize=level_increaser,
                doc="level increase of n identical noise sources.",
            )

            model.assembly_specific_sound_power_level = pyo.Param(
                initialize=35,
                doc="assembly-specific fan sound power level of flow noise (from Table 7 VDI 2081)",
            )

    model.rho = pyo.Param(doc="Density of air. Used in pressure loss calculation")

    model.fixed_zeta = pyo.Param(
        model.E_fixed, doc="Fixed zeta value per fixed edge (i,j)"
    )

    @model.Block(model.Scenarios)
    def scenario(m_scen, s):
        model = m_scen.parent_block()

        m_scen.volume_flow = pyo.Param(model.E, doc="Volume flow along edge in m³/s")

        m_scen.pressure = pyo.Var(
            model.V,
            bounds=(-model.max_pressure / 3, model.max_pressure),
            doc="Pressure at node in Pa",
        )

        m_scen.pressure_change = pyo.Var(
            model.E,
            within=pyo.NonNegativeReals,
            bounds=(0, model.max_pressure),
            doc="Pressure change along edge in Pa",
        )

        m_scen.ind_active = pyo.Var(
            model.E_fan_station,
            within=pyo.Binary,
            doc="Activation indicator for fan station",
        )

        if fan_model == 1:
            m_scen.fan_power_loss = pyo.Var(
                model.fan_set,
                within=pyo.NonNegativeReals,
                bounds=lambda m, i, j, p, d, n: (0, model.fan_power_loss_max[p, d]),
                doc="Electrical power consumption of fan in W",
            )

            m_scen.fan_power_loss_intermediate = pyo.Var(
                model.fan_set,
                within=pyo.NonNegativeReals,
                bounds=lambda m, i, j, p, d, n: (0, model.fan_power_loss_max[p, d]),
                doc="Intermediate value of electrical power consumption of fan in W. Necessary for activation bigM constraints.",
            )

            m_scen.fan_volume_flow_max_scenario = pyo.Param(
                model.edge_fan_types,
                doc="Maximal volume flow of fan in specific load case on specific edge",
                initialize=lambda m, i, j, p, d: min(
                    m.volume_flow[i, j], model.fan_volume_flow_max[p, d]
                ),
            )

            m_scen.fan_volume_flow = pyo.Var(
                model.fan_set,
                within=pyo.NonNegativeReals,
                bounds=lambda m, i, j, p, d, n: (
                    0,
                    m_scen.fan_volume_flow_max_scenario[i, j, p, d],
                ),
                doc="Volume flow of fan in m³/h.",
            )

            m_scen.fan_volume_flow_intermediate = pyo.Var(
                model.fan_set,
                within=pyo.NonNegativeReals,
                bounds=lambda m, i, j, p, d, n: (
                    0,
                    m_scen.fan_volume_flow_max_scenario[i, j, p, d],
                ),
                doc="Intermediate value of volume flow of fan in m³/h. Necessary for activation bigM constraints.",
            )

            m_scen.fan_pressure_change_dimless = pyo.Var(
                model.fan_set,
                within=pyo.NonNegativeReals,
                bounds=(0.001, 1),
                doc="Pressure change of fan in dimensionless form. Dimensionless form has numerical stability reasons.",
            )

            m_scen.fan_ind_active = pyo.Var(model.fan_set, within=pyo.Binary)

            @m_scen.Constraint(
                model.fan_set, doc="Only purchased fans can be activated"
            )
            def only_purchased_fans_are_active(m_scen, i, j, p, d, n):
                return (
                    m_scen.fan_ind_active[i, j, p, d, n]
                    <= model.fan_ind_purchase[i, j, p, d, n]
                )

            @m_scen.Constraint(
                model.fan_set,
                doc="For a fan to be active, the fan station also has to be active.",
            )
            def fans_active_only_if_fan_station_active(m_scen, i, j, p, d, n):
                return m_scen.ind_active[i, j] >= m_scen.fan_ind_active[i, j, p, d, n]

            if planning_mode == "Configuration" and s in model.max_noise_scenarios:

                m_scen = linearize_log10(m_scen)

                @m_scen.Expression(
                    model.fan_set,
                    model.intervals,
                    doc="data for this expression is derived in fan_curves_Least_squares_underestimation_and_rot_speed_fit",
                )
                def fan_performance_curve_sound_power_level_flow_noise(
                    m_scen, i, j, p, d, n, f
                ):
                    # if s in model.max_distributed_throttling_noise_scenarios:
                    #     # fans do not add any noise to max distributed noise scenarios
                    #     return 0
                    fan_pressure_change = (
                        m_scen.fan_pressure_change_dimless[i, j, p, d, n]
                        * model.fan_pressure_max[p, d]
                    )
                    log10_fan_pressure_change = m_scen.log10_fan_pressure_change[
                        i, j, p, d, n
                    ] + np.log10(model.fan_pressure_max[p, d])
                    return (
                        10
                        * (
                            m_scen.log10_fan_volume_flow_intermediate[i, j, p, d, n]
                            + np.log10(m_scen.fan_volume_flow_max_scenario[i, j, p, d])
                        )
                        + 20 * log10_fan_pressure_change
                        + model.assembly_specific_sound_power_level
                        + model.fan_flow_noise_coefficients[p, d, f, 1]
                        * m_scen.fan_volume_flow_intermediate[i, j, p, d, n]
                        + model.fan_flow_noise_coefficients[p, d, f, 2]
                        * fan_pressure_change
                        + model.fan_flow_noise_coefficients[p, d, f, 3]
                    )

        if planning_mode == "Topology":

            if duct_model == 1:

                @m_scen.Expression(
                    model.E_duct,
                    doc="Zeta value of duct friction using the volume flow and not the velocity!",
                )
                def zeta_volume_flow_duct_friction(m_scen, i, j):
                    return (
                        model.rho
                        / 4
                        * model.duct_length[i, j]
                        * model.fun_nonlinear_duct_hb_friction[i, j]
                        * model.duct_resistance_coefficient
                    )

                @m_scen.Expression(
                    model.E_duct,
                    doc="Duct zeta as function of bendings, branches and friction. Equation is pressure_loss = zeta * volume_flow^2",
                )
                def duct_zeta_volume_flow_calc(m_scen, i, j):
                    return (
                        model.rho
                        / 2
                        * (
                            model.fun_duct_nonlinear_hb_area2[i, j]
                            * (
                                model.zeta_bending[i, j]
                                + model.zeta_e_branch[i, j]
                                + model.zeta_t_branch[i, j]
                            )
                        )
                        + m_scen.zeta_volume_flow_duct_friction[i, j]
                    )

                @m_scen.Constraint(
                    model.E_duct,
                    doc="Pressure change along duct is equal to zeta*volume_flow^2",
                )
                def pressure_loss_duct(m_scen, i, j):
                    return (
                        m_scen.pressure_change[i, j]
                        == m_scen.duct_zeta_volume_flow_calc[i, j]
                        * m_scen.volume_flow[i, j] ** 2
                    )

            elif duct_model == 0:

                @m_scen.Constraint(
                    model.E_duct,
                    doc="if duct_model is switched off, then pressure change is zero",
                )
                def pressure_loss_duct(m_scen, i, j):
                    return m_scen.pressure_change[i, j] == 0

        elif planning_mode == "Configuration" and s in model.max_noise_scenarios:
            m_scen.sound_power_level_at_source = pyo.Param(
                model.V_source,
                model.intervals,
                within=pyo.Reals,
                initialize=0,
                mutable=True,
                doc="pressure at target",
            )

            m_scen.sound_power_level = pyo.Var(
                model.V,
                model.intervals,
                within=pyo.Reals,
                bounds=(-60, model.max_sound_power_level),
                doc="sound power level at node v",
            )

            @m_scen.Constraint(model.V_source, model.intervals)
            def sound_power_level_at_source_is_known(m_scen, v, f):
                return (
                    m_scen.sound_power_level_at_source[v, f]
                    == m_scen.sound_power_level[v, f]
                )

            # duct model:

            m_scen.fixed_dampening = pyo.Param(
                model.E_fixed | model.E_duct,
                model.intervals,
                within=pyo.Reals,
                initialize=0,
                doc="dampening of sound power level of the component",
            )

            m_scen.fixed_flow_noise = pyo.Param(
                model.E_fixed | model.E_duct,
                model.intervals,
                within=pyo.Reals,
                initialize=-60,
                doc="flow noise (as sound power level) of the component",
            )

            # fan model:

            m_scen.sound_power_level_identical_fans = pyo.Var(
                model.edge_fan_types,
                model.intervals,
                within=pyo.NonNegativeReals,
                doc="resulting sound power level of all fans that are identical",
            )

            @m_scen.Constraint(model.intervals, model.fan_set)
            def fan_level_per_type_by_increase(fan_station, f, i, j, p, d, n):
                return (
                    m_scen.sound_power_level_identical_fans[i, j, p, d, f]
                    >= model.identical_fan_level_increase[n]
                    * m_scen.fan_ind_active[i, j, p, d, n]
                    + m_scen.fan_performance_curve_sound_power_level_flow_noise[
                        i, j, p, d, n, f
                    ]
                )

            # silencer model

            @m_scen.Constraint(
                model.E_silencer,
                doc="pressure loss is given by performance curve pressure if purchased, zero otherwise",
            )
            def pressure_big_M_constr1(m_scen, i, j):
                return (
                    m_scen.pressure_change[i, j]
                    <= m_scen.volume_flow[i, j] ** 2
                    * model.silencer_zeta_max[i, j]
                    * model.ind_purchase[i, j]
                )

            @m_scen.Constraint(
                model.E_silencer,
                doc="Necessary to not use bound here so that volume_flow can be changed dynamically within the model",
            )
            def pressure_loss_lower_bound(m_scen, i, j):
                return (
                    m_scen.pressure_change[i, j]
                    >= model.silencer_zeta_min[i, j]
                    * m_scen.volume_flow[i, j] ** 2
                    * model.ind_purchase[i, j]
                )

            m_scen = level_add_multiple_multiindex(
                model,
                m_scen,
            )

            m_scen.performance_curve_sound_power_level_flow_noise = pyo.Var(
                model.E,
                model.intervals,
                within=pyo.Reals,
                bounds=(-60, model.max_sound_power_level),
            )

            m_scen.performance_curve_sound_power_level_dampening = pyo.Var(
                model.E,
                model.intervals,
                within=pyo.Reals,
                bounds=(0, model.max_sound_power_level),
            )

            m_scen.performance_curve_sound_power_level_dampening_purchased = pyo.Var(
                model.E_silencer,
                model.intervals,
                within=pyo.Reals,
                bounds=(0, model.max_sound_power_level),
            )

            @m_scen.Constraint(model.E, model.intervals)
            def definition_performance_curve_sound_power_level_dampening(
                m_scen, i, j, f
            ):
                if (i, j) in model.E_duct | model.E_fixed:
                    return (
                        m_scen.performance_curve_sound_power_level_dampening[i, j, f]
                        == m_scen.fixed_dampening[i, j, f]
                    )
                elif (i, j) in model.E_silencer:
                    return m_scen.performance_curve_sound_power_level_dampening[
                        i, j, f
                    ] == (
                        model.silencer_reg_coef_dampening[f, 1]
                        * model.number_of_splitters[i, j] ** 2
                        + model.silencer_reg_coef_dampening[f, 2]
                        * model.silencer_width[i, j] ** 2
                        * model.number_of_splitters[i, j]
                        + model.silencer_reg_coef_dampening[f, 3]
                        * model.silencer_length[i, j]
                        * model.number_of_splitters[i, j]
                        + model.silencer_reg_coef_dampening[f, 4]
                        * model.silencer_length[i, j]
                        * model.silencer_width[i, j]
                        + model.silencer_reg_coef_dampening[f, 5]
                        * model.number_of_splitters[i, j]
                        + model.silencer_reg_coef_dampening[f, 6]
                        * model.silencer_length[i, j]
                        + model.silencer_reg_coef_dampening[f, 7]
                        * model.silencer_width[i, j]
                        + model.silencer_reg_coef_dampening[f, 8]
                    )
                return (
                    m_scen.performance_curve_sound_power_level_dampening[i, j, f] == 0
                )  # for empty, vfc, fan

            def vfc_pressure_loss(i, j):
                if s in model.max_distributed_throttling_noise_scenarios:
                    return model.max_throttling_in_max_noise_scenario[i, j]
                else:
                    return m_scen.pressure_change[i, j]

            @m_scen.Constraint(model.E, model.intervals)
            def definition_performance_curve_sound_power_level_flow_noise(
                m_scen, i, j, f
            ):
                if (i, j) in model.E_fan_station:
                    k = len(model.fan_types_on_edge[i, j])
                    if k == 1:  # acount for fan stations with a single type of fans
                        k = 2
                    return (
                        m_scen.performance_curve_sound_power_level_flow_noise[i, j, f]
                        == m_scen.sound_power_level_in_level_addition[i, j, k - 1, f]
                    )
                elif (i, j) in model.E_duct | model.E_fixed:
                    return (
                        m_scen.performance_curve_sound_power_level_flow_noise[i, j, f]
                        == m_scen.fixed_flow_noise[i, j, f]
                    )
                elif (i, j) in model.E_vfc:
                    return m_scen.performance_curve_sound_power_level_flow_noise[
                        i, j, f
                    ] == (
                        model.vfc_reg_coef_flow_noise[f, 1]
                        * m_scen.volume_flow[i, j] ** 2
                        / (model.vfc_height[i, j] * model.vfc_width[i, j]) ** 2
                        + model.vfc_reg_coef_flow_noise[f, 2]
                        * m_scen.volume_flow[i, j]
                        / (model.vfc_height[i, j] * model.vfc_width[i, j])
                        + model.vfc_reg_coef_flow_noise[f, 3]
                        * (model.vfc_height[i, j] * model.vfc_width[i, j])
                        + model.vfc_reg_coef_flow_noise[f, 4]
                        * (model.vfc_height[i, j] * model.vfc_width[i, j]) ** 0.5
                        + model.vfc_reg_coef_flow_noise[f, 5] * vfc_pressure_loss(i, j)
                        + model.vfc_reg_coef_flow_noise[f, 6]
                    )
                elif (i, j) in model.E_silencer:
                    return (
                        m_scen.performance_curve_sound_power_level_flow_noise[i, j, f]
                        * model.silencer_nonlinear_term[i, j, 2]
                        == model.silencer_reg_coef_flow_noise[f, 2]
                        * m_scen.volume_flow[i, j]
                        * model.silencer_nonlinear_term[i, j, 1]
                        + model.silencer_reg_coef_flow_noise[f, 3]
                        * model.silencer_height[i, j]
                        * model.silencer_width[i, j]
                        * model.silencer_nonlinear_term[i, j, 2]
                        + model.silencer_reg_coef_flow_noise[f, 4]
                        * (model.silencer_height[i, j] * model.silencer_width[i, j])
                        ** 2
                        * model.silencer_nonlinear_term[i, j, 2]
                        + model.silencer_reg_coef_flow_noise[f, 5]
                        * model.silencer_nonlinear_term[i, j, 2]
                    )
                return (
                    m_scen.performance_curve_sound_power_level_flow_noise[i, j, f]
                    == -10
                )  # for empty

            if s in model.max_central_noise_scenarios:

                @m_scen.Constraint(
                    model.crosstalk_frequency_band, model.crosstalk_node_pairs
                )
                def crosstalk(m_scen, f, i, j):

                    silencer_edge1 = find_silencer_on_leaf_strand(model, i)
                    silencer_edge2 = find_silencer_on_leaf_strand(model, j)

                    return (
                        m_scen.performance_curve_sound_power_level_dampening_purchased[
                            *silencer_edge1, f
                        ]
                        + m_scen.performance_curve_sound_power_level_dampening_purchased[
                            *silencer_edge2, f
                        ]
                        >= model.crosstalk_damping_margin[f, i, j]
                    )

            m_scen = add_level_addition_elements(model, m_scen)

            # connecting sound power level in, out and max+inc
            @m_scen.Constraint(
                model.E_fixed | model.E_duct,
                model.intervals,
                doc="sound power level out is sound power level in + increase",
            )
            def sound_power_level_connector_fixed_duct(m_scen, i, j, f):
                return (
                    m_scen.sound_power_level[j, f]
                    == m_scen.sound_power_level_out_intermediate[i, j, f]
                )

            @m_scen.Constraint(
                model.E_empty,
                model.intervals,
                doc="sound power level out is sound power level in + increase",
            )
            def sound_power_level_connector_empty(m_scen, i, j, f):
                return m_scen.sound_power_level[j, f] == m_scen.sound_power_level[i, f]

            # connecting sound power level in, out and max+inc
            @m_scen.Constraint(
                model.E_silencer | model.E_fan_station | model.E_vfc,
                model.intervals,
                doc="sound power level is given by performance inc if active, independent otherwise",
            )
            def sound_power_level_big_M_constr1(m_scen, i, j, f):
                if (i, j) in model.E_fan_station:
                    x = m_scen.ind_active[i, j]
                elif (i, j) in model.E_silencer | model.E_vfc:
                    x = model.ind_purchase[i, j]
                return m_scen.sound_power_level[j, f] - (
                    m_scen.sound_power_level_out_intermediate[i, j, f]
                ) <= model.max_sound_power_level * (1 - x)

            @m_scen.Constraint(
                model.E_silencer | model.E_fan_station | model.E_vfc,
                model.intervals,
                doc="sound power level is given by performance inc if active, independent otherwise",
            )
            def sound_power_level_big_M_constr2(m_scen, i, j, f):
                if (i, j) in model.E_fan_station:
                    x = m_scen.ind_active[i, j]
                elif (i, j) in model.E_silencer | model.E_vfc:
                    x = model.ind_purchase[i, j]
                return -m_scen.sound_power_level[j, f] + (
                    m_scen.sound_power_level_out_intermediate[i, j, f]
                ) <= model.max_sound_power_level * (1 - x)

            @m_scen.Constraint(
                model.E_silencer | model.E_fan_station | model.E_vfc,
                model.intervals,
                doc="sound power level is given by performance inc if active, independent otherwise",
            )
            def sound_power_level_big_M_constr3(m_scen, i, j, f):
                if (i, j) in model.E_fan_station:
                    x = m_scen.ind_active[i, j]
                elif (i, j) in model.E_silencer | model.E_vfc:
                    x = model.ind_purchase[i, j]
                return (
                    m_scen.sound_power_level[j, f] - m_scen.sound_power_level[i, f]
                    <= model.max_sound_power_level * x
                )

            @m_scen.Constraint(
                model.E_silencer | model.E_fan_station | model.E_vfc,
                model.intervals,
                doc="sound power level is given by performance inc if active, independent otherwise",
            )
            def sound_power_level_big_M_constr4(m_scen, i, j, f):
                if (i, j) in model.E_fan_station:
                    x = m_scen.ind_active[i, j]
                elif (i, j) in model.E_silencer | model.E_vfc:
                    x = model.ind_purchase[i, j]
                return (
                    -m_scen.sound_power_level[j, f] + m_scen.sound_power_level[i, f]
                    <= model.max_sound_power_level * x
                )

            @m_scen.Constraint(
                model.E_silencer,
                model.intervals,
                doc="silencer dampening only nonzero if silencer is purchased (needed for crosstalk)",
            )
            def silencer_dampening_big_M_constr1(m_scen, i, j, f):
                return m_scen.performance_curve_sound_power_level_dampening[
                    i, j, f
                ] - m_scen.performance_curve_sound_power_level_dampening_purchased[
                    i, j, f
                ] <= model.silencer_dampening_max[
                    f
                ] * (
                    1 - model.ind_purchase[i, j]
                )

            @m_scen.Constraint(
                model.E_silencer,
                model.intervals,
                doc="silencer dampening only nonzero if silencer is purchased (needed for crosstalk)",
            )
            def silencer_dampening_big_M_constr2(m_scen, i, j, f):
                return m_scen.performance_curve_sound_power_level_dampening[
                    i, j, f
                ] - m_scen.performance_curve_sound_power_level_dampening_purchased[
                    i, j, f
                ] <= model.silencer_dampening_max[
                    f
                ] * (
                    1 - model.ind_purchase[i, j]
                )

            @m_scen.Constraint(
                model.E_silencer,
                model.intervals,
                doc="silencer dampening only nonzero if silencer is purchased (needed for crosstalk)",
            )
            def silencer_dampening_big_M_constr3(m_scen, i, j, f):
                return (
                    m_scen.performance_curve_sound_power_level_dampening_purchased[
                        i, j, f
                    ]
                    <= model.silencer_dampening_max[f] * model.ind_purchase[i, j]
                )

            # room model

            m_scen.sound_pressure_level_room = pyo.Var(
                model.V_room,
                within=pyo.Reals,
                bounds=(-60, model.max_sound_power_level),
            )

            @m_scen.Expression(model.V_room, model.intervals)
            def sound_power_level_in_A_weighted(m_scen, v, f):
                A_weight = {
                    1: -25.2,
                    2: -15.6,
                    3: -8.4,
                    4: -3.1,
                    5: 0,
                    6: 1.2,
                    7: 0.9,
                    8: -1.1,
                }
                return (
                    m_scen.sound_power_level[v, f]
                    + A_weight[f]
                    + model.power_to_pressure_conversion_term[v, f]
                )

            m_scen = level_add_room(model, m_scen)

            @m_scen.Constraint(
                model.V_room,
                doc="sound power level out is sound power level in + increase",
            )
            def connect_input_with_output(m_scen, v):
                return (
                    m_scen.sound_pressure_level_room[v]
                    == m_scen.sound_power_level_in_level_addition_room[v, 7]
                )

            @m_scen.Constraint(
                model.V_room,
                doc="limit A weighted sound pressure level in the room by a predefined threshold in dB",
            )
            def limit_sound_pressure_level_in_room(m_scen, v):
                return (
                    m_scen.sound_pressure_level_room[v]
                    <= model.max_sound_pressure_level_room[v]
                )

        @m_scen.Constraint(
            model.E_fan_station, doc="Only purchased fan stations can be activated"
        )
        def only_purchased_fan_stations_are_active(m_scen, i, j):
            return m_scen.ind_active[i, j] <= model.ind_purchase[i, j]

        @m_scen.Constraint(
            model.V_source, doc="Pressure at source node is equal to zero"
        )
        def set_pressure_at_source_to_zero(m_scen, v):
            return m_scen.pressure[v] == 0

        if pressure_target_met == 1:

            @m_scen.Constraint(
                model.V_target,
                doc="Pressure at target and source node is equal to zero",
            )
            def set_pressure_at_ports_to_zero(m_scen, v):
                return m_scen.pressure[v] == 0

        elif pressure_target_met == 0:

            @m_scen.Constraint(
                model.V_target,
                doc="Pressure at source node is equal or larger than zero",
            )
            def set_pressure_at_targets_geq_zero(m_scen, v):
                return m_scen.pressure[v] >= 0

        @m_scen.Constraint(
            model.E_fan_station,
            doc="Pressure change of fan is zero if fan is not active",
        )
        def pressure_increase_fan(m_scen, i, j):
            return (
                m_scen.pressure_change[i, j]
                <= model.max_pressure * m_scen.ind_active[i, j]
            )

        @m_scen.Constraint(
            model.E_fixed,
            doc="Pressure change of a fixed component is zeta*volume_flow^2",
        )
        def pressure_loss_fix(m_scen, i, j):
            return (
                m_scen.pressure_change[i, j]
                == model.fixed_zeta[i, j] * m_scen.volume_flow[i, j] ** 2
            )

        @m_scen.Constraint(
            model.E_vfc, doc="Pressure loss of a VFC is zero if VFC is not active"
        )
        def pressure_loss_vfc(m_scen, i, j):
            return (
                m_scen.pressure_change[i, j]
                <= model.max_pressure * model.ind_purchase[i, j]
            )

        @m_scen.Constraint(
            model.E_empty,
            doc="pressure change along edge is zero for empty edges",
        )
        def pressure_change_empty_edges(m_scen, i, j):
            return m_scen.pressure_change[i, j] == 0

        if planning_mode == "Topology":

            @m_scen.Constraint(
                model.E_silencer,
                doc="Pressure change of a silencer is zeta*volume_flow^2",
            )
            def pressure_loss_silencer(m_scen, i, j):
                return (
                    m_scen.pressure_change[i, j]
                    == model.silencer_zeta[i, j] * m_scen.volume_flow[i, j] ** 2
                )

        elif planning_mode == "Configuration":

            @m_scen.Constraint(
                model.E_silencer,
                doc="Pressure change of a silencer component is zeta*volume_flow^2 if purchased",
            )
            def pressure_loss_silencer_bigM1(m_scen, i, j):
                return (
                    m_scen.pressure_change[i, j]
                    <= model.silencer_zeta_max[i, j] * m_scen.volume_flow[i, j] ** 2
                )

            @m_scen.Constraint(
                model.E_silencer,
                doc="Pressure change of a silencer component is zeta*volume_flow^2 if purchased",
            )
            def pressure_loss_silencer_bigM2(m_scen, i, j):
                return m_scen.pressure_change[i, j] - model.silencer_zeta[
                    i, j
                ] * m_scen.volume_flow[i, j] ** 2 <= model.silencer_zeta_max[
                    i, j
                ] * m_scen.volume_flow[
                    i, j
                ] ** 2 * (
                    1 - model.ind_purchase[i, j]
                )

            @m_scen.Constraint(
                model.E_silencer,
                doc="Pressure change of a silencer component is zeta*volume_flow^2 if purchased",
            )
            def pressure_loss_silencer_bigM3(m_scen, i, j):
                return -m_scen.pressure_change[i, j] + model.silencer_zeta[
                    i, j
                ] * m_scen.volume_flow[i, j] ** 2 <= model.silencer_zeta_max[
                    i, j
                ] * m_scen.volume_flow[
                    i, j
                ] ** 2 * (
                    1 - model.ind_purchase[i, j]
                )

            @m_scen.Constraint(
                model.E_duct,
                doc="Pressure change of a duct is zeta*volume_flow^2",
            )
            def pressure_loss_duct(m_scen, i, j):
                return (
                    m_scen.pressure_change[i, j]
                    == model.duct_zeta[i, j] * m_scen.volume_flow[i, j] ** 2
                )

        # pressure propagation

        @m_scen.Constraint(
            model.E_fan_station,
            doc="Pressure difference along a fan station edge is equal to the fan station's pressure rise",
        )
        def pressure_propagation_fan(m_scen, i, j):
            return (
                -m_scen.pressure[i] + m_scen.pressure[j] == m_scen.pressure_change[i, j]
            )

        @m_scen.Constraint(
            model.E_vfc
            | model.E_duct
            | model.E_fixed
            | model.E_empty
            | model.E_silencer,
            doc="Pressure difference along a VFC, duct, fixed component, silencer or empty edge is equal to the negative pressure_change of that edge",
        )
        def pressure_propagation_vfc_and_duct(m_scen, i, j):
            return (
                -m_scen.pressure[i] + m_scen.pressure[j]
                == -m_scen.pressure_change[i, j]
            )

        if fan_model == 1:

            # ALL FAN STATION CONSTRAINTS

            @m_scen.Expression(
                model.E_fan_station, doc="Hydraulic power of the fan station"
            )
            def electric_power_consumption_factor_fan_station(m_scen, i, j):
                return m_scen.volume_flow[i, j] * m_scen.pressure_change[i, j]

            @m_scen.Constraint(
                model.fan_set, doc="bigM connecting intermediate power consumption"
            )
            def power_loss_bigm_a(m_scen, i, j, p, d, n):
                return m_scen.fan_power_loss_intermediate[
                    i, j, p, d, n
                ] - m_scen.fan_power_loss[i, j, p, d, n] <= model.fan_power_loss_max[
                    p, d
                ] * (
                    1 - m_scen.fan_ind_active[i, j, p, d, n]
                )

            @m_scen.Constraint(
                model.fan_set, doc="bigM connecting intermediate power consumption"
            )
            def power_loss_bigm_b(m_scen, i, j, p, d, n):
                return m_scen.fan_power_loss_intermediate[
                    i, j, p, d, n
                ] - m_scen.fan_power_loss[i, j, p, d, n] >= -model.fan_power_loss_max[
                    p, d
                ] * (
                    1 - m_scen.fan_ind_active[i, j, p, d, n]
                )

            @m_scen.Expression(
                model.E_fan_station,
                doc="Power consumption of a fan station = fan's power loss + hydraulic power",
            )
            def electric_power_consumption_fan_station(m_scen, i, j):
                return (
                    sum(
                        m_scen.fan_power_loss[i, j, p, d, n]
                        for (k, l, p, d, n) in model.fan_set
                        if (i, j) == (k, l)
                    )
                    + m_scen.electric_power_consumption_factor_fan_station[i, j]
                )

            @m_scen.Constraint(
                model.fan_set, doc="Electric power of fan is zero if fan is not active"
            )
            def power_loss_bigm_c(m_scen, i, j, p, d, n):
                return (
                    m_scen.fan_power_loss[i, j, p, d, n]
                    <= model.fan_power_loss_max[p, d]
                    * m_scen.fan_ind_active[i, j, p, d, n]
                )

            @m_scen.Constraint(
                model.fan_set,
                doc="Pressure change of fan station and fan are equal if fan is active",
            )
            def pressure_change_connection_to_fan_station_bigm_a(m_scen, i, j, p, d, n):
                return m_scen.fan_pressure_change_dimless[
                    i, j, p, d, n
                ] * model.fan_pressure_max[p, d] - m_scen.pressure_change[
                    i, j
                ] <= model.max_pressure * (
                    1 - m_scen.fan_ind_active[i, j, p, d, n]
                )

            @m_scen.Constraint(
                model.fan_set,
                doc="Pressure change of fan station and fan are equal if fan is active",
            )
            def pressure_change_connection_to_fan_station_bigm_b(m_scen, i, j, p, d, n):
                return m_scen.fan_pressure_change_dimless[
                    i, j, p, d, n
                ] * model.fan_pressure_max[p, d] - m_scen.pressure_change[
                    i, j
                ] >= -model.max_pressure * (
                    1 - m_scen.fan_ind_active[i, j, p, d, n]
                )

            @m_scen.Constraint(
                model.E_fan_station,
                doc="Pressure change of fan station is zero if fan station is not active",
            )
            def pressure_change_connection_to_fan_station_bigm_c(m_scen, i, j):
                return (
                    m_scen.pressure_change[i, j]
                    <= model.max_pressure * m_scen.ind_active[i, j]
                )

            if reduce_fan_curves:

                @m_scen.Constraint(
                    model.fan_set,
                    model.fan_hyperplanes_underestimation_specific_set,
                    doc="Intermediate electrical power consumption >= underestimating hyperplanes",
                )
                def power_loss_lower_bound_fan(
                    m_scen, i, j, p, d, n, s_, i_, j_, p_, d_, t
                ):
                    if (s, i, j, p, d) == (s_, i_, j_, p_, d_):
                        return (
                            m_scen.fan_power_loss_intermediate[i, j, p, d, n]
                            >= model.fan_hyperplanes_underestimation_slope_pressure[
                                p, d, t
                            ]
                            * m_scen.fan_pressure_change_dimless[i, j, p, d, n]
                            * model.fan_pressure_max[p, d]
                            + model.fan_hyperplanes_underestimation_slope_volume_flow[
                                p, d, t
                            ]
                            * m_scen.fan_volume_flow_intermediate[i, j, p, d, n]
                            + model.fan_hyperplanes_underestimation_intercept[p, d, t]
                        )
                    return pyo.Constraint.Skip

                @m_scen.Constraint(
                    model.fan_set,
                    model.fan_hyperplanes_overestimation_specific_set,
                    doc="Intermediate electrical power consumption <= overestimating hyperplanes",
                )
                def power_loss_upper_bound_fan(
                    m_scen, i, j, p, d, n, s_, i_, j_, p_, d_, t
                ):
                    if (s, i, j, p, d) == (s_, i_, j_, p_, d_):
                        return (
                            m_scen.fan_power_loss_intermediate[i, j, p, d, n]
                            <= model.fan_hyperplanes_overestimation_slope_volume_flow[
                                p, d, t
                            ]
                            * m_scen.fan_volume_flow_intermediate[i, j, p, d, n]
                            + model.fan_hyperplanes_overestimation_intercept[p, d, t]
                        )
                    return pyo.Constraint.Skip

            else:

                @m_scen.Constraint(
                    model.fan_set,
                    model.fan_hyperplanes_underestimation_set,
                    doc="Intermediate electrical power consumption >= underestimating hyperplanes",
                )
                def power_loss_lower_bound_fan(m_scen, i, j, p, d, n, p_, d_, t):
                    if (p, d) == (p_, d_):
                        return (
                            m_scen.fan_power_loss_intermediate[i, j, p, d, n]
                            >= model.fan_hyperplanes_underestimation_slope_pressure[
                                p, d, t
                            ]
                            * m_scen.fan_pressure_change_dimless[i, j, p, d, n]
                            * model.fan_pressure_max[p, d]
                            + model.fan_hyperplanes_underestimation_slope_volume_flow[
                                p, d, t
                            ]
                            * m_scen.fan_volume_flow_intermediate[i, j, p, d, n]
                            + model.fan_hyperplanes_underestimation_intercept[p, d, t]
                        )
                    return pyo.Constraint.Skip

                @m_scen.Constraint(
                    model.fan_set,
                    model.fan_hyperplanes_overestimation_set,
                    doc="Intermediate electrical power consumption <= overestimating hyperplanes",
                )
                def power_loss_upper_bound_fan(m_scen, i, j, p, d, n, p_, d_, t):
                    if (p, d) == (p_, d_):
                        return (
                            m_scen.fan_power_loss_intermediate[i, j, p, d, n]
                            <= model.fan_hyperplanes_overestimation_slope_volume_flow[
                                p, d, t
                            ]
                            * m_scen.fan_volume_flow_intermediate[i, j, p, d, n]
                            + model.fan_hyperplanes_overestimation_intercept[p, d, t]
                        )
                    return pyo.Constraint.Skip

            @m_scen.Constraint(
                model.E_fan_station,
                doc="fan station volume flow = sum of fans' volume flows",
            )
            def volume_flow_connection_to_fan_station_bigm_a(m_scen, i, j):
                return (
                    sum(
                        m_scen.fan_volume_flow[i, j, p, d, n]
                        for (k, l, p, d, n) in model.fan_set
                        if (k, l) == (i, j)
                    )
                    == m_scen.volume_flow[i, j] * m_scen.ind_active[i, j]
                )

            @m_scen.Constraint(
                model.fan_set, doc="bigM connecting intermediate volume flow"
            )
            def volume_flow_connection_to_fan_station_bigm_b(m_scen, i, j, p, d, n):
                return m_scen.fan_volume_flow_intermediate[
                    i, j, p, d, n
                ] - m_scen.fan_volume_flow[i, j, p, d, n] <= m_scen.volume_flow[
                    i, j
                ] * (
                    1 - m_scen.fan_ind_active[i, j, p, d, n]
                )

            @m_scen.Constraint(
                model.fan_set, doc="bigM connecting intermediate volume flow"
            )
            def volume_flow_connection_to_fan_station_bigm_c(m_scen, i, j, p, d, n):
                return m_scen.fan_volume_flow_intermediate[
                    i, j, p, d, n
                ] - m_scen.fan_volume_flow[i, j, p, d, n] >= -m_scen.volume_flow[
                    i, j
                ] * (
                    1 - m_scen.fan_ind_active[i, j, p, d, n]
                )

            @m_scen.Constraint(
                model.fan_set, doc="Fan volume flow is zero if fan is not active"
            )
            def volume_flow_connection_to_fan_station_bigm_d(m_scen, i, j, p, d, n):
                return (
                    m_scen.fan_volume_flow[i, j, p, d, n]
                    <= m_scen.volume_flow[i, j] * m_scen.fan_ind_active[i, j, p, d, n]
                )

        elif fan_model == 0:

            @m_scen.Expression(
                model.E_fan_station,
                doc="If no fan model is used, the electrical energy consumption is P_hyd/0.6 - an efficiency of 0.6 is assumed",
            )
            def electric_power_consumption_fan_station(model, i, j):
                return m_scen.pressure_change[i, j] * m_scen.volume_flow[i, j] / 0.6

        m_scen.electric_power_consumption = pyo.Var(
            doc="Electric power consumption of all fan stations"
        )

        @m_scen.Constraint(
            doc="Electrical power consumption is sum of all fan stations'"
        )
        def def_electric_power_consumption(m_scen):
            return m_scen.electric_power_consumption == sum(
                m_scen.electric_power_consumption_fan_station[i, j]
                for (i, j) in model.E_fan_station
            )

    if duct_model == 1 and branching_constraints == 1:
        # the following for constraints are quite a conservative overestimation
        # for the relationship between the ducts of a branch
        # as such they are not used in the publication

        @model.Constraint(
            model.E_duct,
            doc="width of inbranch >= sqrt(0.6 * (q_in/q_out)) * width of bend outbranch which stems from w_A/w>=0.6",
        )
        def branch_limit_width_ratio_straight_branch(model, i, j):
            if (i, j) in [(k, l) for k, l, m in model.duct_e_branch]:
                edge_in = next((o, p) for o, p in model.E_duct if p == i)
                volume_flow_ratio = max(
                    model.scenario[s].volume_flow[i, j]
                    / model.scenario[s].volume_flow[edge_in]
                    for s in model.Scenarios
                )
                return (
                    model.duct_width[edge_in]
                    >= np.sqrt(0.6 / volume_flow_ratio) * model.duct_width[i, j]
                )
            return pyo.Constraint.Skip

        @model.Constraint(
            model.E_duct,
            doc="height of inbranch >= sqrt(0.6 * (q_in/q_out)) * height of bend outbranch which stems from w_A/w>=0.6",
        )
        def branch_limit_height_ratio_straight_branch(model, i, j):
            if (i, j) in [(k, l) for k, l, m in model.duct_e_branch]:
                edge_in = next((o, p) for o, p in model.E_duct if p == i)
                volume_flow_ratio = max(
                    model.scenario[s].volume_flow[i, j]
                    / model.scenario[s].volume_flow[edge_in]
                    for s in model.Scenarios
                )
                return (
                    model.duct_height[edge_in]
                    >= np.sqrt(0.6 / volume_flow_ratio) * model.duct_height[i, j]
                )
            return pyo.Constraint.Skip

        @model.Constraint(
            model.E_duct,
            doc="width of inbranch >= sqrt(0.6 * (q_in/q_out)) * width of bend outbranch which stems from w_A/w>=0.6",
        )
        def branch_limit_width_ratio_bending_branch(model, i, j):
            if (i, j) in [(k, m) for k, l, m in model.duct_e_branch]:
                edge_in = next((o, p) for o, p in model.E_duct if p == i)
                volume_flow_ratio = max(
                    model.scenario[s].volume_flow[i, j]
                    / model.scenario[s].volume_flow[edge_in]
                    for s in model.Scenarios
                )
                return (
                    model.duct_width[edge_in]
                    >= np.sqrt(0.6 / volume_flow_ratio) * model.duct_width[i, j]
                )
            return pyo.Constraint.Skip

        @model.Constraint(
            model.E_duct,
            doc="height of inbranch >= sqrt(0.6 * (q_in/q_out)) * height of bend outbranch which stems from w_A/w>=0.6",
        )
        def branch_limit_height_ratio_bending_branch(model, i, j):
            if (i, j) in [(k, m) for k, l, m in model.duct_e_branch]:
                edge_in = next((o, p) for o, p in model.E_duct if p == i)
                volume_flow_ratio = max(
                    model.scenario[s].volume_flow[i, j]
                    / model.scenario[s].volume_flow[edge_in]
                    for s in model.Scenarios
                )
                return (
                    model.duct_height[edge_in]
                    >= np.sqrt(0.6 / volume_flow_ratio) * model.duct_height[i, j]
                )
            return pyo.Constraint.Skip

    if planning_mode == "Topology" and duct_model == 1 and velocity_constraint == 1:

        model.inverse_hyperplanes_set = pyo.Set(
            doc="Set of duct inverse hyperplanes. Used for outer polyhedral approximation of 1/w"
        )

        model.inverse_slope_width = pyo.Param(
            model.inverse_hyperplanes_set,
            doc="Slope in width direction of the inverse hyperplanes \
                for outer polyhedral approximation",
        )
        model.inverse_intercept = pyo.Param(
            model.inverse_hyperplanes_set,
            doc="Intercept of the inverse hyperplanes \
                for outer polyhedral approximation",
        )

        model.inverse_width = pyo.Var(
            model.E_duct,
            bounds=lambda model, i, j: (
                1 / model.duct_width_max[i, j],
                1 / model.duct_width_min[i, j],
            ),
            doc="Variable set to 1/w using supporting hyperplanes",
        )

        model.max_velocity = pyo.Param(
            model.E_duct,
            initialize=5,
            mutable=True,
            doc="Maximum allowed velocity in duct (i,j)",
        )

        @model.Constraint(
            model.E_duct,
            model.inverse_hyperplanes_set,
            doc="Inverse width >= supporting hyperplanes",
        )
        def approx_inverse_width(model, i, j, t):
            return (
                model.inverse_width[i, j]
                >= model.inverse_intercept[t]
                + model.inverse_slope_width[t] * model.duct_width[i, j]
            )

        @model.Constraint(model.E_duct, doc="Limit duct area by max velocity")
        def velocity_limit(model, i, j):
            return (
                max(model.scenario[s].volume_flow[i, j] for s in model.Scenarios)
                * model.inverse_width[i, j]
            ) <= model.max_velocity[i, j] * model.duct_height[i, j]

    @model.Expression(
        model.component_names,
        doc="Compute annuity factors for three component types according to VDI 2067",
    )
    def component_annuity(model, comp_name):
        T = int(model.operating_years.value)
        T_N = model.deprecation_period[comp_name]

        Z = model.interest_rate
        R = model.price_change_factor_service_maintenance
        if abs(Z - R) < 1e-12:
            B_SM = T / Z
        else:
            B_SM = (1 - (R / Z) ** T) / (Z - R)
        annuity_factor = (Z - 1) / (1 - Z ** (-T))

        cost_factor = annuity_factor * (
            1
            + (model.service_factor[comp_name] + model.maintenance_factor[comp_name])
            * B_SM
        )

        div, mod = divmod(T, T_N)

        if mod == 0:
            return (
                1
                + sum((R / Z) ** (T_N * i) for i in range(1, div))
                - R ** (T_N * div) * mod / T_N * 1 / Z**T
            ) * cost_factor
        return (
            1
            + sum((R / Z) ** (T_N * i) for i in range(1, div + 1))
            - R ** (T_N * div) * mod / T_N * 1 / Z**T
        ) * cost_factor

    if planning_mode == "Topology" and duct_model == 1:

        @model.Expression(doc="Total used duct area in m²")
        def total_duct_used(model):
            return 2 * sum(
                (model.duct_width[i, j] + model.duct_height[i, j])
                * model.duct_length[i, j]
                for (i, j) in model.E_duct
            )

        @model.Expression(doc="Total duct costs in €")
        def total_duct_costs(model):
            return capex_reduction * (
                model.component_annuity["duct"]
                * model.operating_years
                * model.duct_area_costs
                * model.total_duct_used
            )

        def calculate_duct_losses(model, s, i, j):
            return (
                model.rho
                / 2
                * (
                    model.fun_duct_nonlinear_hb_area2[i, j]
                    * (
                        model.zeta_bending[i, j]
                        + model.zeta_e_branch[i, j]
                        + model.zeta_t_branch[i, j]
                    )
                    + model.fun_nonlinear_duct_hb_friction[i, j]
                    * model.duct_resistance_coefficient
                    * (model.duct_length[i, j] / 2)
                )
            )

    elif planning_mode == "Topology" and duct_model == 0:

        @model.Expression()
        def total_duct_costs(model, doc="If not duct model is used, the costs are 0"):
            return 0

    if fan_model == 1:

        @model.Constraint(model.fan_set, doc="Symmetry breaking constraint")
        def symmetry_breaking(model, i, j, p, d, n):
            if n > 1:
                return (
                    model.fan_ind_purchase[i, j, p, d, n]
                    <= model.fan_ind_purchase[i, j, p, d, n - 1]
                )
            return pyo.Constraint.Skip

        @model.Constraint(
            model.Scenarios,
            model.fan_set,
            doc="Identical fans have identical volume flows",
        )
        def identical_fans_operate_identically_a(model, s, i, j, p, d, n):
            if n > 1:
                return (
                    model.scenario[s].fan_volume_flow_intermediate[i, j, p, d, n]
                    == model.scenario[s].fan_volume_flow_intermediate[i, j, p, d, 1]
                )
            return pyo.Constraint.Skip

        @model.Constraint(
            model.Scenarios,
            model.fan_set,
            doc="Identical fans have identical power loss",
        )
        def identical_fans_operate_identically_b(model, s, i, j, p, d, n):
            if n > 1:
                return (
                    model.scenario[s].fan_power_loss_intermediate[i, j, p, d, n]
                    == model.scenario[s].fan_power_loss_intermediate[i, j, p, d, 1]
                )
            return pyo.Constraint.Skip

        @model.Constraint(
            model.Scenarios,
            model.fan_set,
            doc="Identical fans have identical power loss",
        )
        def identical_fans_operate_identically_c(model, s, i, j, p, d, n):
            if n > 1:
                return (
                    model.scenario[s].fan_pressure_change_dimless[i, j, p, d, n]
                    == model.scenario[s].fan_pressure_change_dimless[i, j, p, d, 1]
                )
            return pyo.Constraint.Skip

        @model.Expression(doc="Total fan costs in €")
        def total_fan_costs(model):
            return (
                model.component_annuity["fan"]
                * model.operating_years
                * capex_reduction
                * sum(
                    model.fan_costs[p, d] * model.fan_ind_purchase[i, j, p, d, n]
                    for (i, j, p, d, n) in model.fan_set
                )
            )

        @model.Constraint(
            model.E_fan_station,
            doc="A fan station is only purchased if at least one fan is purchased",
        )
        def fan_station_only_purchased_if_fan_purchased_a(model, i, j):
            return model.ind_purchase[i, j] <= sum(
                model.fan_ind_purchase[i, j, p, d, n]
                for (k, l, p, d, n) in model.fan_set
                if (k, l) == (i, j)
            )

        @model.Constraint(
            model.fan_set,
            doc="A fan station is only purchased if at least one fan is purchased",
        )
        def fan_station_only_purchased_if_fan_purchase_b(model, i, j, p, d, n):
            return model.ind_purchase[i, j] >= model.fan_ind_purchase[i, j, p, d, n]

        if planning_mode == "Topology":

            @model.Constraint(
                model.Scenarios, doc="Lower bound of electrical power consumption"
            )
            def limit_electric_power_subproblem(model, s):

                pel_hyd_fixed = sum(
                    model.scenario[s].volume_flow[edge] ** 3 * model.fixed_zeta[edge]
                    for edge in model.E_fixed
                )

                pel_hyd_silencer = sum(
                    model.scenario[s].volume_flow[edge] ** 3 * model.silencer_zeta[edge]
                    for edge in model.E_silencer
                )

                if duct_model == 1:
                    pel_hyd_duct = sum(
                        model.scenario[s].volume_flow[edge] ** 3
                        * calculate_duct_losses(model, s, *edge)
                        for edge in model.E_duct
                    )
                else:
                    pel_hyd_duct = 0

                return (
                    model.scenario[s].electric_power_consumption
                    >= (pel_hyd_duct + pel_hyd_fixed + pel_hyd_silencer) / 0.7
                )

        elif planning_mode == "Configuration":

            @model.Constraint(
                model.Scenarios, doc="Lower bound of electrical power consumption"
            )
            def limit_electric_power_subproblem(model, s):

                pel_hyd_fixed = sum(
                    model.scenario[s].volume_flow[edge] ** 3 * model.fixed_zeta[edge]
                    for edge in model.E_fixed
                )

                pel_hyd_silencer = sum(
                    model.scenario[s].volume_flow[edge] ** 3 * model.silencer_zeta[edge]
                    for edge in model.E_silencer
                )

                return (
                    model.scenario[s].electric_power_consumption
                    >= (pel_hyd_fixed + pel_hyd_silencer) / 0.7
                )

    elif fan_model == 0:

        @model.Expression(doc="If not fan model is used, fan costs are set to 1000 €")
        def total_fan_costs(model):
            return capex_reduction * (
                model.component_annuity["fan"]
                * model.operating_years
                * sum(1000 * model.ind_purchase[i, j] for (i, j) in model.E_fan_station)
            )

    @model.Expression(doc="Fan power consumption over all scenarios")
    def fan_power_consumption(model):
        return sum(
            model.time_share[s] * model.scenario[s].electric_power_consumption
            for s in model.Scenarios
        )

    if planning_mode == "Topology":

        @model.Expression(
            doc="Total duct volume in m³, only used in postprocessing - is quadratic constraint (removed during solve)"
        )
        def duct_volume(model):
            return sum(
                model.duct_length[e] * model.duct_height[e] * model.duct_width[e]
                for e in model.E_duct
            )

    @model.Expression(doc="Total fan energy costs in €")
    def fan_energy_costs(model):
        Z = model.interest_rate
        annuity_factor = (Z - 1) / (1 - Z ** (-model.operating_years))
        B_E = (
            1 - (model.price_change_factor_electricity / Z) ** model.operating_years
        ) / (Z - model.price_change_factor_electricity)

        return (
            annuity_factor
            * B_E
            * model.electric_energy_costs
            * model.operating_years
            * model.operating_days_per_year
            * model.operating_hours_per_day
            * model.fan_power_consumption
        )

    @model.Expression(doc="Total VFC invest costs in €")
    def total_vfc_costs(model):
        if planning_mode == "Topology":
            return capex_reduction * (
                model.component_annuity["vfc"]
                * model.operating_years
                * model.vfc_costs
                * sum(model.ind_purchase[i, j] for (i, j) in model.E_vfc)
            )
        elif planning_mode == "Configuration":
            return capex_reduction * (
                model.component_annuity["vfc"]
                * model.operating_years
                * sum(
                    model.ind_purchase[i, j] * model.vfc_costs_expr[i, j]
                    for (i, j) in model.E_vfc
                )
            )

    if planning_mode == "Configuration":

        @model.Constraint(model.E_silencer)
        def bigM_splitter_silencer_costs_a(model, i, j):
            return model.silencer_costs_intermediate[i, j] - model.silencer_costs[
                i, j
            ] <= model.silencer_costs_max[i, j] * (1 - model.ind_purchase[i, j])

        @model.Constraint(model.E_silencer)
        def bigM_splitter_silencer_costs_b(model, i, j):
            return -model.silencer_costs_intermediate[i, j] + model.silencer_costs[
                i, j
            ] <= model.silencer_costs_max[i, j] * (1 - model.ind_purchase[i, j])

        @model.Constraint(model.E_silencer)
        def bigM_splitter_silencer_costs_c(model, i, j):
            return (
                model.silencer_costs[i, j]
                <= model.silencer_costs_max[i, j] * model.ind_purchase[i, j]
            )

        @model.Expression(doc="Total VFC invest costs in €")
        def total_silencer_costs(model):
            return capex_reduction * (
                model.component_annuity["silencer"]
                * model.operating_years
                * sum(model.silencer_costs[i, j] for i, j in model.E_silencer)
            )

    ## additional equipment costs
    if additional_investment_costs:
        model.binary_vfc_fan = pyo.Var(
            model.E_fan_station_leaf,
            doc="variable that is equal to y_VFC * y_Fan, needed for computation of costs",
        )

        @model.Constraint(
            model.E_fan_station_leaf,
            doc="logical connection binary_vfc_fan A",
        )
        def logical_connection_binary_vfc_fan_a(model, i, j):
            return model.binary_vfc_fan[i, j] <= model.ind_purchase[i, j]

        @model.Constraint(
            model.E_fan_station_leaf,
            model.E_vfc_leaf,
            doc="logical connection binary_vfc_fan A",
        )
        def logical_connection_binary_vfc_fan_b(model, i, j, k, l):
            if (l == i) or (j == k):  # fan placed behind or before vfc
                return model.binary_vfc_fan[i, j] <= model.ind_purchase[k, l]
            return pyo.Constraint.Skip

        @model.Constraint(
            model.E_fan_station_leaf,
            model.E_vfc_leaf,
            doc="logical connection binary_vfc_fan A",
        )
        def logical_connection_binary_vfc_fan_c(model, i, j, k, l):
            if (l == i) or (j == k):  # fan placed behind or before vfc
                return (
                    model.binary_vfc_fan[i, j]
                    >= model.ind_purchase[i, j] + model.ind_purchase[k, l] - 1
                )
            return pyo.Constraint.Skip

        @model.Expression(doc="Additional equipment's investment cost")
        def equipment_investment_cost(model):
            additional_vfc_cost = (
                model.additional_measurement_costs["variable_vfc"]
                if variable_air_volume
                else model.additional_measurement_costs["constant_vfc"]
            )

            return (
                model.component_annuity["measurement"]
                * model.operating_years
                * capex_reduction
                * (
                    additional_vfc_cost
                    * sum(model.ind_purchase[i, j] for (i, j) in model.E_vfc_leaf)
                    + model.additional_measurement_costs["fan"]
                    * sum(
                        model.ind_purchase[i, j]
                        for (i, j) in model.E_fan_station - model.E_fan_station_central
                    )
                    + model.additional_measurement_costs["both"]
                    * sum(
                        model.binary_vfc_fan[i, j]
                        for (i, j) in model.E_fan_station_leaf
                    )
                )
            )

        @model.Expression(doc="Total investment costs in €")
        def total_invest_costs(model):
            if planning_mode == "Topology":
                return (
                    model.total_duct_costs
                    + model.total_fan_costs
                    + model.total_vfc_costs
                    + model.equipment_investment_cost
                )
            else:
                return (
                    model.total_fan_costs
                    + model.total_vfc_costs
                    + model.total_silencer_costs
                    + model.equipment_investment_cost
                )

    else:

        @model.Expression(doc="Total investment costs in €")
        def total_invest_costs(model):
            if planning_mode == "Topology":
                return (
                    model.total_duct_costs
                    + model.total_fan_costs
                    + model.total_vfc_costs
                )
            else:
                return (
                    model.total_fan_costs
                    + model.total_silencer_costs
                    + model.total_vfc_costs
                )

    ## end of additional equipment costs

    @model.Objective(
        doc="Minimal life-cycle costs consisting of energy and investment costs in €"
    )
    def min_lcc(model):
        costs = model.fan_energy_costs + model.total_invest_costs
        if planning_mode == "Topology":
            return costs
        # return costs + 1e-6*sum(model.scenario[s].sound_power_level[i,f] for s in model.Scenarios for (i) in model.V for f in model.intervals) + 1e-4*sum(model.scenario[s].sound_pressure_level_room[v] for s in model.Scenarios for v in model.V_room)
        #
        return (
            costs
            + 1e-6
            * sum(
                model.scenario[s].sound_power_level[i, f]
                for s in model.max_noise_scenarios
                for (i) in model.V
                for f in model.intervals
            )
            + 1e-4
            * sum(
                model.scenario[s].sound_pressure_level_room[v]
                for s in model.max_noise_scenarios
                for v in model.V_room
            )
        )

    return model
