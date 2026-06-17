from typing import Dict, Iterable, Optional, Tuple


def specific_silencer_data(
    data_dict: Dict,
    E_silencer: Iterable[Tuple],
    width: Optional[float] = None,
    height: Optional[float] = None,
    min_n_splitters: int = 1,
    max_n_splitters: int = 11,
    silencer_length_max: float = 1.5,
    silencer_length_min: float = 0.5,
) -> Dict:
    """Add silencer dimensions and bounds to the data dictionary."""
    silencer_data = {
        "min_n_splitters": {(i, j): min_n_splitters for i, j in E_silencer},
        "max_n_splitters_raw": {(i, j): max_n_splitters for i, j in E_silencer},
        "silencer_length_max": {(i, j): silencer_length_max for i, j in E_silencer},
        "silencer_length_min": {(i, j): silencer_length_min for i, j in E_silencer},
        "silencer_height": {(i, j): height for i, j in E_silencer},
        "silencer_width": {(i, j): width for i, j in E_silencer},
    }

    for key in silencer_data.keys():
        if key in data_dict:
            data_dict[key].update(silencer_data[key])
        else:
            data_dict[key] = silencer_data[key]

    return data_dict


def silencer_reg_coef_data() -> Dict:
    """Return regression coefficients for silencers."""
    silencer_reg_coef_data = {
        "silencer_reg_coef_cost": {
            1: 268.84701,
            2: 1188.80552,
            3: -265.95679,
            4: -84.44579,
        },
        "silencer_reg_coef_pressure": {
            1: 0.06904,
            2: 0.01991,
            3: 0.00903,
        },
        "silencer_reg_coef_flow_noise": {
            (1, 1): -0.0,
            (1, 2): 0.00119,
            (1, 3): -3.8407,
            (1, 4): 12.64305,
            (1, 5): 9.20462,
            (2, 1): -0.0,
            (2, 2): 0.00119,
            (2, 3): -3.89285,
            (2, 4): 12.67729,
            (2, 5): 4.87445,
            (3, 1): -0.0,
            (3, 2): 0.00115,
            (3, 3): -4.07966,
            (3, 4): 13.17477,
            (3, 5): 1.1905,
            (4, 1): -0.0,
            (4, 2): 0.00112,
            (4, 3): -4.07761,
            (4, 4): 13.11626,
            (4, 5): -2.01805,
            (5, 1): -0.0,
            (5, 2): 0.00111,
            (5, 3): -3.86444,
            (5, 4): 12.80267,
            (5, 5): -5.00744,
            (6, 1): -0.0,
            (6, 2): 0.00108,
            (6, 3): -4.01165,
            (6, 4): 13.0546,
            (6, 5): -7.69825,
            (7, 1): -0.0,
            (7, 2): 0.0011,
            (7, 3): -3.88397,
            (7, 4): 12.87891,
            (7, 5): -10.99401,
            (8, 1): -0.0,
            (8, 2): 0.00112,
            (8, 3): -3.69122,
            (8, 4): 12.4232,
            (8, 5): -14.49875,
        },
        "silencer_reg_coef_dampening": {
            (1, 1): -0.19505,
            (1, 2): 0.08821,
            (1, 3): 1.0625,
            (1, 4): -3.42188,
            (1, 5): 1.80448,
            (1, 6): 2.5,
            (1, 7): -1.93867,
            (1, 8): -1.06263,
            (2, 1): -0.31752,
            (2, 2): -0.042,
            (2, 3): 2.9875,
            (2, 4): -7.96563,
            (2, 5): 3.77201,
            (2, 6): 4.7,
            (2, 7): -4.37341,
            (2, 8): -2.17472,
            (3, 1): -0.5303,
            (3, 2): 0.11173,
            (3, 3): 2.2875,
            (3, 4): -6.54063,
            (3, 5): 6.82603,
            (3, 6): 10.9,
            (3, 7): -10.21416,
            (3, 8): -2.85852,
            (4, 1): -1.03326,
            (4, 2): 0.29822,
            (4, 3): 5.0125,
            (4, 4): -13.78438,
            (4, 5): 12.15426,
            (4, 6): 20.3,
            (4, 7): -18.09029,
            (4, 8): -3.13258,
            (5, 1): -1.88372,
            (5, 2): 0.99126,
            (5, 3): 0.225,
            (5, 4): -1.11875,
            (5, 5): 26.21772,
            (5, 6): 19.4,
            (5, 7): -48.79298,
            (5, 8): -0.75232,
            (6, 1): -1.56766,
            (6, 2): 0.31334,
            (6, 3): 5.35,
            (6, 4): -14.7125,
            (6, 5): 21.48014,
            (6, 6): 14.4,
            (6, 7): -34.66173,
            (6, 8): -3.22284,
            (7, 1): -0.61644,
            (7, 2): -0.53679,
            (7, 3): 7.075,
            (7, 4): -19.45625,
            (7, 5): 10.26761,
            (7, 6): 8.8,
            (7, 7): -11.83264,
            (7, 8): -2.91561,
            (8, 1): -0.32687,
            (8, 2): -0.53007,
            (8, 3): 3.6875,
            (8, 4): -10.14063,
            (8, 5): 7.91933,
            (8, 6): 4.5,
            (8, 7): -10.25292,
            (8, 8): -1.20602,
        },
    }
    return silencer_reg_coef_data


def specific_vfc_data(
    data_dict: Dict,
    E_vfc: Iterable[Tuple],
    width: float,
    height: float,
) -> Dict:
    """Add VFC dimensions to the data dictionary."""
    vfc_data = {
        "vfc_height": {(i, j): height for i, j in E_vfc},
        "vfc_width": {(i, j): width for i, j in E_vfc},
    }

    for key in vfc_data.keys():
        if key in data_dict:
            data_dict[key].update(vfc_data[key])
        else:
            data_dict[key] = vfc_data[key]

    return data_dict


def vfc_reg_coef_data() -> Dict:
    """Return regression coefficients for volume flow controllers."""
    vfc_reg_coef_data = {
        "variable_vfc_reg_coef_cost": {
            1: 1100,
            2: 1100,
        },
        "constant_vfc_reg_coef_cost": {1: 1100, 2: 600},
        "vfc_reg_coef_flow_noise": {
            (1, 1): -0.12784,
            (1, 2): 3.76991,
            (1, 3): -34.44698,
            (1, 4): 52.16539,
            (1, 5): 0.02259,
            (1, 6): 30.54927,
            (2, 1): -0.14832,
            (2, 2): 4.43775,
            (2, 3): -36.07968,
            (2, 4): 55.42897,
            (2, 5): 0.02336,
            (2, 6): 23.51695,
            (3, 1): -0.13337,
            (3, 2): 3.96179,
            (3, 3): -52.42564,
            (3, 4): 83.17849,
            (3, 5): 0.03804,
            (3, 6): 7.98912,
            (4, 1): -0.09123,
            (4, 2): 2.582,
            (4, 3): -30.93785,
            (4, 4): 46.50587,
            (4, 5): 0.04037,
            (4, 6): 20.61149,
            (5, 1): -0.06334,
            (5, 2): 1.65548,
            (5, 3): -20.48581,
            (5, 4): 29.03846,
            (5, 5): 0.03603,
            (5, 6): 32.91954,
            (6, 1): -0.03856,
            (6, 2): 0.83666,
            (6, 3): -27.56702,
            (6, 4): 41.06506,
            (6, 5): 0.03887,
            (6, 6): 35.15944,
            (7, 1): -0.03996,
            (7, 2): 0.88318,
            (7, 3): -20.05177,
            (7, 4): 28.74215,
            (7, 5): 0.04791,
            (7, 6): 32.53312,
            (8, 1): -0.05056,
            (8, 2): 1.23905,
            (8, 3): -21.12551,
            (8, 4): 30.03966,
            (8, 5): 0.05274,
            (8, 6): 26.45979,
        },
    }
    return vfc_reg_coef_data
