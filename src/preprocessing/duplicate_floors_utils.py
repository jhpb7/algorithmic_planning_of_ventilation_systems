from data.vfc_sil_data.create_sil_vfc_data import specific_silencer_data


IGNORE_COMPONENTS = [
    "rho",
    "max_pressure",
    "electric_energy_costs",
    "vfc_costs",
    "operating_years",
    "operating_days_per_year",
    "operating_hours_per_day",
    "price_change_factor_electricity",
    "price_change_factor_service_maintenance",
    "interest_rate",
    "duct_area_costs",
    "duct_resistance_coefficient",
    "Scenarios",
    "time_share",
    "duct_friction_slope_height",
    "duct_friction_hyperplanes_set",
    "duct_friction_intercept",
    "duct_friction_slope_width",
    "duct_area2_slope_height",
    "duct_area2_hyperplanes_set",
    "duct_area2_intercept",
    "duct_area2_slope_width",
    "inverse_hyperplanes_set",
    "inverse_intercept",
    "inverse_slope_width",
    "fan_hyperplanes_underestimation_pre_set",
    "fan_hyperplanes_underestimation_intercept",
    "fan_hyperplanes_underestimation_slope_volume_flow",
    "fan_hyperplanes_underestimation_slope_pressure",
    "fan_hyperplanes_overestimation_pre_set",
    "fan_hyperplanes_overestimation_intercept",
    "fan_hyperplanes_overestimation_slope_volume_flow",
    "fan_product_line",
    "fan_diameter",
    "fan_costs",
    "fan_power_loss_max",
    "fan_volume_flow_max",
    "fan_pressure_max",
    "fan_pressure_coefficients",
    "fan_power_coefficients",
    "max_sound_power_level",
    "silencer_reg_coef_cost",
    "silencer_reg_coef_pressure",
    "silencer_reg_coef_flow_noise",
    "silencer_reg_coef_dampening",
    "variable_vfc_reg_coef_cost",
    "constant_vfc_reg_coef_cost",
    "constant_vfc_costs",
    "variable_vfc_costs",
    "vfc_reg_coef_flow_noise",
    "max_volume_flow_scenario",
    "min_volume_flow_scenario",
    "acoustically_relevant_scenarios",
    "additional_measurement_costs",
    "fixed_zeta_central_AHU",
    "crosstalk_dampening_comparison_through_wall",
]


def create_multi_floor(floor_dict, n_floors):
    if None in floor_dict.keys():
        if isinstance(floor_dict[None], list) and floor_dict[None]:
            if isinstance(floor_dict[None][0], str):
                return {
                    None: [
                        str(n) + x for n in range(n_floors) for x in floor_dict[None]
                    ]
                }
            return {
                None: [
                    tuple(str(n) + x[i] for i in range(len(x)))
                    for n in range(n_floors)
                    for x in floor_dict[None]
                ]
            }
        return floor_dict
    if isinstance(list(floor_dict.keys())[0], tuple):

        return {
            (str(n) + key[0], str(n) + key[1]): value
            for n in range(n_floors)
            for key, value in floor_dict.items()
        }
    return {
        str(n) + key: value
        for n in range(n_floors)
        for key, value in floor_dict.items()
    }


def get_successor_node(build_data, node):
    successor_nodes = [e[1] for e in build_data["E"][None] if e[0] == node]
    if len(successor_nodes) != 1:
        raise ValueError(f"Expected 1 successor nodes but found {len(successor_nodes)}")
    return successor_nodes[0]


def add_floor_connection(build_data, n_floors, floor_entrance_node):
    for n in range(n_floors, 0, -1):
        if n == n_floors:
            build_data["V"][None].extend(["Z0", "Z1", "Z2", "Z3"])
            build_data["V_source"][None].append("Z0")
            e = ("Z3", str(n_floors - 1) + floor_entrance_node)
            e_fs = ("Z0", "Z1")
            e_fixed = ("Z1", "Z2")
            e_silencer = ("Z2", "Z3")
            build_data["E"][None].extend([e_fs, e_fixed, e_silencer])
            build_data["E_fan_station"][None].append(e_fs)
            build_data["E_fixed"][None].append(e_fixed)
            build_data["E_silencer"][None].append(e_silencer)
            zeta = build_data["fixed_zeta_central_AHU"][None]  # 34
            build_data["fixed_zeta"][e_fixed] = zeta
            e_s = [e]

            build_data = specific_silencer_data(build_data, [e_silencer], 1.0, 1.0)

        else:
            e1 = (
                str(n) + floor_entrance_node,
                str(n) + floor_entrance_node + "D",
            )  # duct edge
            e2 = (
                str(n) + floor_entrance_node + "D",
                str(n) + floor_entrance_node + "F",
            )  # fan edge
            e3 = (
                str(n) + floor_entrance_node + "F",
                str(n) + floor_entrance_node + "S",
            )  # silencer edge
            e4 = (
                str(n) + floor_entrance_node + "S",
                str(n - 1) + floor_entrance_node,
            )  # duct edge
            e_s = [e1, e4]

            successor_to_floor_entrance_node = get_successor_node(
                build_data, str(n) + floor_entrance_node
            )

            build_data["duct_e_branch"][None].append(
                tuple(
                    [
                        str(n) + floor_entrance_node,
                        str(n) + floor_entrance_node + "D",
                        successor_to_floor_entrance_node,
                    ]
                )
            )
            build_data["E"][None].extend([e2, e3])
            build_data["E_fan_station"][None].append(e2)
            build_data["E_silencer"][None].append(e3)
            build_data["V"][None].append(str(n) + floor_entrance_node + "D")
            build_data["V"][None].append(str(n) + floor_entrance_node + "F")
            build_data["V"][None].append(str(n) + floor_entrance_node + "S")

            build_data = specific_silencer_data(build_data, [e3], width=2, height=1.5)

        for e in e_s:
            if "Z3" == e[0]:
                build_data["duct_length"][e] = 3.8
            else:
                build_data["duct_length"][e] = 3.8 / 2
            build_data["E"][None].append(e)
            build_data["E_duct"][None].append(e)
            build_data["E_duct_vertical"][None].append(e)
            build_data["n_duct_bendings"][e] = 0
            build_data["duct_width_min"][e] = 0.2
            build_data["duct_height_min"][e] = 0.2
            build_data["duct_width_max"][e] = 2
            build_data["duct_height_max"][e] = 2
            build_data["duct_area2_hyperplanes_specific_pre_set"][e] = build_data[
                "duct_area2_hyperplanes_set"
            ]
            build_data["duct_friction_hyperplanes_specific_pre_set"][e] = build_data[
                "duct_friction_hyperplanes_set"
            ]

    return build_data


def duplicate_floors_add_connectors(floor_data, n_floors, floor_entrance_node="A0"):
    build_data = dict()

    for key, value in floor_data.items():
        if key in IGNORE_COMPONENTS:
            build_data[key] = floor_data[key]
        elif key == "scenario":
            build_data["scenario"] = dict()
            for s, volume_flows in floor_data["scenario"].items():
                build_data["scenario"][s] = {"volume_flow": dict()}
                scen = build_data["scenario"][s]["volume_flow"]
                for n in range(n_floors):
                    for edge, value in volume_flows["volume_flow"].items():
                        scen[(str(n) + edge[0], str(n) + edge[1])] = value
                floor_volume_flow = max(volume_flows["volume_flow"].values())

                scen[("Z0", "Z1")] = floor_volume_flow * n_floors
                scen[("Z1", "Z2")] = floor_volume_flow * n_floors
                scen[("Z2", "Z3")] = floor_volume_flow * n_floors
                scen[("Z3", str(n_floors - 1) + floor_entrance_node)] = (
                    floor_volume_flow * n_floors
                )
                for n in range(n_floors - 1):
                    scen[
                        str(n + 1) + floor_entrance_node,
                        str(n + 1) + floor_entrance_node + "D",
                    ] = floor_volume_flow * (1 + n)
                    scen[
                        str(n + 1) + floor_entrance_node + "D",
                        str(n + 1) + floor_entrance_node + "F",
                    ] = floor_volume_flow * (1 + n)
                    scen[
                        str(n + 1) + floor_entrance_node + "F",
                        str(n + 1) + floor_entrance_node + "S",
                    ] = floor_volume_flow * (1 + n)
                    scen[
                        str(n + 1) + floor_entrance_node + "S",
                        str(n) + floor_entrance_node,
                    ] = floor_volume_flow * (1 + n)

        elif key == "fan_set":
            build_data["fan_set"] = {
                None: [
                    (
                        [str(idx) + node_from, str(idx) + node_to, fan, diameter, i]
                        if not (node_from, node_to)
                        in [
                            ("Z0", "Z1"),
                            *[
                                (
                                    str(x) + floor_entrance_node + "D",
                                    str(x) + floor_entrance_node + "F",
                                )
                                for x in range(1, n_floors)
                            ],
                        ]
                        else [node_from, node_to, fan, diameter, i]
                    )
                    for idx in range(n_floors)
                    for (node_from, node_to, fan, diameter, i) in floor_data[key][None]
                ]
            }
        elif key == "max_num_fans_per_fan_station":
            build_data["max_num_fans_per_fan_station"] = {
                (
                    (str(idx) + node_from, str(idx) + node_to)
                    if not (node_from, node_to)
                    in [
                        ("Z0", "Z1"),
                        *[
                            (
                                str(x) + floor_entrance_node + "D",
                                str(x) + floor_entrance_node + "F",
                            )
                            for x in range(1, n_floors)
                        ],
                    ]
                    else (node_from, node_to)
                ): max_num
                for idx in range(n_floors)
                for (node_from, node_to), max_num in floor_data[key].items()
            }

        else:
            # print(key)
            build_data[key] = create_multi_floor(value, n_floors)

    build_data = add_floor_connection(build_data, n_floors, floor_entrance_node)
    build_data["n_floors"] = n_floors
    return build_data
