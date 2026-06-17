from pyomo2h5 import save_yaml
from data.vfc_sil_data.create_sil_vfc_data import (
    silencer_reg_coef_data,
    specific_silencer_data,
    vfc_reg_coef_data,
)


def get_floor_data():
    data = {
        "V": {
            None: [
                "A0",
                "A1",
                "A2",
                "A3",
                "A4",
                "A5",
                "B0",
                "B1",
                "B2",
                "B3",
                "C0",
                "C1",
                "C2",
                "C3",
                "C4",
                "D0",
                "D1",
                "D2",
                "D3",
                "D4",
                "E0",
                "E1",
                "E2",
                "E3",
                "E4",
                "E5",
                "F0",
                "F1",
                "F2",
                "F3",
                "F4",
                "G0",
                "G1",
                "G2",
                "G3",
                "G4",
                "H0",
                "H1",
                "H2",
                "H3",
                "H4",
                "I0",
                "I1",
                "I2",
                "I3",
                "I4",
            ]
        },
        "V_source": {None: []},
        "V_target": {None: ["C4", "D4", "F4", "G4", "H4", "I4"]},
        "V_room": {None: ["C4", "D4", "F4", "G4", "H4", "I4"]},
        "E": {
            None: [
                ("A0", "A1"),
                ("A1", "A2"),
                ("A2", "A3"),
                ("A3", "A4"),
                ("A4", "A5"),
                ("A5", "B0"),
                ("B0", "B1"),
                ("B1", "B2"),
                ("B2", "B3"),
                ("B3", "C0"),
                ("C0", "C1"),
                ("C1", "C2"),
                ("C2", "C3"),
                ("C3", "C4"),
                ("B3", "D0"),
                ("D0", "D1"),
                ("D1", "D2"),
                ("D2", "D3"),
                ("D3", "D4"),
                ("A5", "E0"),
                ("E0", "E1"),
                ("E1", "E2"),
                ("E2", "E3"),
                ("E3", "E4"),
                ("E4", "E5"),
                ("E3", "F0"),
                ("F0", "F1"),
                ("F1", "F2"),
                ("F2", "F3"),
                ("F3", "F4"),
                ("E4", "G0"),
                ("G0", "G1"),
                ("G1", "G2"),
                ("G2", "G3"),
                ("G3", "G4"),
                ("E5", "H0"),
                ("H0", "H1"),
                ("H1", "H2"),
                ("H2", "H3"),
                ("H3", "H4"),
                ("E5", "I0"),
                ("I0", "I1"),
                ("I1", "I2"),
                ("I2", "I3"),
                ("I3", "I4"),
            ]
        },
        "E_duct": {
            None: [
                ("A0", "A1"),
                ("A4", "A5"),
                ("A5", "B0"),
                ("A5", "E0"),
                ("B2", "B3"),
                ("E2", "E3"),
                ("E3", "E4"),
                ("E4", "E5"),
                ("B3", "C0"),
                ("B3", "D0"),
                ("E3", "F0"),
                ("E4", "G0"),
                ("E5", "H0"),
                ("E5", "I0"),
            ]
        },
        "E_duct_vertical": {None: []},
        "E_fan_station": {
            None: [
                ("A2", "A3"),
                ("B0", "B1"),
                ("C0", "C1"),
                ("D0", "D1"),
                ("E0", "E1"),
                ("F0", "F1"),
                ("G0", "G1"),
                ("H0", "H1"),
                ("I0", "I1"),
            ]
        },  #
        "E_vfc": {
            None: [
                ("C1", "C2"),
                ("D1", "D2"),
                ("F1", "F2"),
                ("G1", "G2"),
                ("H1", "H2"),
                ("I1", "I2"),
            ]
        },  #
        "E_silencer": {
            None: [
                ("B1", "B2"),
                ("E1", "E2"),
                ("A3", "A4"),
                ("C2", "C3"),
                ("D2", "D3"),
                ("F2", "F3"),
                ("G2", "G3"),
                ("H2", "H3"),
                ("I2", "I3"),
            ]
        },
        "E_fixed": {
            None: [
                ("A1", "A2"),
                ("C3", "C4"),
                ("D3", "D4"),
                ("F3", "F4"),
                ("G3", "G4"),
                ("H3", "H4"),
                ("I3", "I4"),
            ]
        },
        "rho": {None: 1.2},
        "max_pressure": {None: 6000},
        "fixed_zeta_link": {
            ("A1", "A2"): "fire_damper.yaml",
            ("C3", "C4"): "office3.yaml",
            ("D3", "D4"): "recreation_room.yaml",
            ("F3", "F4"): "offices_seminar.yaml",
            ("G3", "G4"): "office3.yaml",
            ("H3", "H4"): "office3.yaml",
            ("I3", "I4"): "offices_seminar.yaml",
        },
        "fixed_zeta_central_AHU": {None: 6.42},
        "electric_energy_costs": {None: 0.1453 / 1000},  # Industriestrompreis in €/Wh
        "constant_vfc_costs": {None: 600},
        "variable_vfc_costs": {None: 1100},
        "operating_years": {None: 15},
        "operating_days_per_year": {None: 250},
        "operating_hours_per_day": {None: 14},
        "price_change_factor_electricity": {None: 1.052},
        "price_change_factor_service_maintenance": {None: 1.03},
        "interest_rate": {None: 1.03},
        "additional_measurement_costs": {
            "fan": 1500,
            "constant_vfc": 200,
            "variable_vfc": 450,
            "both": -1000,
        },
        "max_sound_power_level": {None: 150},
    }

    # %% duct sizing

    duct_sizing_data = {
        "duct_width_min": {(i, j): 0.2 for (i, j) in data["E_duct"][None]},
        "duct_width_max": {(i, j): 2 for (i, j) in data["E_duct"][None]},
        "duct_height_min": {(i, j): 0.2 for (i, j) in data["E_duct"][None]},
        "duct_height_max": {(i, j): 2 for (i, j) in data["E_duct"][None]},
        "duct_length": {
            ("A0", "A1"): 0.5,
            ("A4", "A5"): 0.5,
            ("A5", "B0"): 1,
            ("A5", "E0"): 1,
            ("B2", "B3"): 1,  # fans and silencers are 1 m altogether
            ("E2", "E3"): 1,
            ("E3", "E4"): 4.3,
            ("E4", "E5"): 4.34,
            ("B3", "C0"): 3.66,
            ("B3", "D0"): 6,
            ("E3", "F0"): 10,  # to the rooms center
            ("E4", "G0"): 3.66,  # to the rooms center
            ("E5", "H0"): 3.66,  # to the rooms center
            ("E5", "I0"): 12.5,
        },
    }

    duct_data = {
        # "lambda_duct": {None: 0.02},
        "n_duct_bendings": {
            ("E4", "E5"): 1,
            ("B3", "D0"): 1,
            ("E3", "F0"): 2,
            ("E5", "I0"): 1,
        },
        "duct_e_branch": {  # k,l,m: l goes straight, m bends
            None: [
                ("B3", "C0", "D0"),
                ("E3", "F0", "E4"),
                ("E4", "E5", "G0"),
                ("E5", "I0", "H0"),
            ]
        },
        "duct_t_branch_node": {
            None: [
                "A5",
            ]
        },
        "duct_area_costs": {None: 50},
        "duct_resistance_coefficient": {None: 0.02},
    }

    duct_data.update(duct_sizing_data)

    data.update(duct_data)

    data.update(silencer_reg_coef_data())
    data = specific_silencer_data(data, data["E_silencer"][None])

    data.update(vfc_reg_coef_data())
    # data = specific_vfc_data(data, data["E_vfc"][None],width=0.2,height=0.1)

    room_data = {
        "max_sound_pressure_level_room": {
            "C4": 45,
            "D4": 35,
            "F4": 35,
            "G4": 45,
            "H4": 45,
            "I4": 35,
        },
        "V_room_normal": {None: ["C4", "D4", "F4", "G4", "H4", "I4"]},
        "room_area": {
            "C4": 5.33 * 14.93,
            "D4": 8 * 5.33 / 2,
            "F4": 5.33**2,
            "G4": 9.6 * 5.33,
            "H4": 3.73 * (8 + 5.33),
            "I4": 5.33 * 3.73,
        },
        "room_height": {
            "C4": 3,
            "D4": 3,
            "F4": 3,
            "G4": 3,
            "H4": 3,
            "I4": 3,
        },
        "room_reverberation_time": {
            "C4": 0.5,
            "D4": 1.0,
            "F4": 0.5,
            "G4": 0.5,
            "H4": 0.5,
            "I4": 0.5,
        },
        "number_of_air_diffusers": {
            "C4": 2,
            "D4": 1.0,
            "F4": 1,
            "G4": 2,
            "H4": 2,
            "I4": 1,
        },
        "min_distance_to_airdiffuser": {
            "C4": 2,
            "D4": 1.7,
            "F4": 2,
            "G4": 2,
            "H4": 2,
            "I4": 1.7,
        },
        "location_of_sound_source_opening": {
            "C4": 1,
            "D4": 1,
            "F4": 1,
            "G4": 1,
            "H4": 1,
            "I4": 1,
        },
        "angle_of_radiation": {
            "C4": 0,
            "D4": 0,
            "F4": 0,
            "G4": 0,
            "H4": 0,
            "I4": 0,
        },
        "area_of_outlet": {
            "C4": 0.14 * 0.14,
            "D4": 0.14 * 0.14,
            "F4": 0.14 * 0.14,
            "G4": 0.2 * 0.2,
            "H4": 0.2 * 0.2,
            "I4": 0.14 * 0.14,
        },
        "partition_wall_area": {
            ("C4", "D4"): 5.33 / 2 * 3,
            ("F4", "G4"): 5.33 * 3,
            ("G4", "H4"): 5.33 * 3,
            ("H4", "I4"): 3.73 * 3,
        },
        "crosstalk_dampening_comparison_through_wall": {
            2: 30,
            3: 41,
            4: 45,
            5: 48,
            6: 47,  # gipskarton auf stahlprofil
        },
        "crosstalk_nodes": {None: ["C4", "D4", "F4", "G4", "H4", "I4"]},
        "crosstalk_node_pairs": {
            None: [("C4", "D4"), ("F4", "G4"), ("G4", "H4"), ("H4", "I4")]
        },
        "crosstalk_link": {
            ("C4", "D4"): "crosstalk_CD.yaml",
            ("F4", "G4"): "crosstalk.yaml",
            ("G4", "H4"): "crosstalk.yaml",
            ("H4", "I4"): "crosstalk.yaml",
        },
    }

    data.update(room_data)

    # which scenario to use for max noise using pressure change
    data["max_volume_flow_scenario"] = 8
    data["min_volume_flow_scenario"] = 1
    data["acoustically_relevant_scenarios"] = [8]

    return data


if __name__ == "__main__":

    floor_data = get_floor_data()

    save_yaml("data/network_data/OFF.yml", floor_data)
