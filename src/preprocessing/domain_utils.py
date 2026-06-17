from collections import defaultdict
from typing import Any, Dict, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from .general_utils import get_point_distance

# --- Fan utils ---


def dimensionalise_coeffs(
    a: dict, b: dict, d: float, max_values: Dict
) -> Tuple[dict, dict]:
    """Scale polynomial coefficients with fan diameter and max values."""
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


def get_max_values(fan_data: Dict) -> Dict:
    """Extract max q, dp, pel from fan data limits."""
    limits = fan_data["limits"]
    return {
        "q": limits["volume_flow"]["max_dim"],
        "dp": limits["pressure"]["max_dim"],
        "pel": limits["power"]["max_dim"],
    }


def prepare_fan_on_edges(data: Dict) -> Dict:
    """Format fan-on-edges data into Pyomo-style sets and parameters."""
    result = []
    for (node_from, node_to), values in data["edges"].items():
        for p_d, weight in values.items():
            if p_d != "total":
                fan, diameter = p_d
                for i in range(1, weight + 1):
                    result.append([node_from, node_to, fan, diameter, i])
    max_num_fans_per_fs = {edge: vals["total"] for edge, vals in data["edges"].items()}
    return {
        "fan_set": {None: result},
        "max_num_fans_per_fan_station": max_num_fans_per_fs,
    }


def level_add(oktav_spl):
    """
    Pegel Addition beliebiger Element einer Liste oder eines 1D arrays
    """
    return 10 * np.log10(np.sum([10 ** (0.1 * (x)) for x in oktav_spl]))


def compute_flow_noise_coefficients(a, max_values, n_max):
    # q = np.logspace(-2, 0, 100)
    # dp = np.logspace(-2, 0, 10)  # np.linspace(1e-3,1,10)

    q = np.linspace(0.01 * max_values["q"], max_values["q"], 100)
    dp = np.linspace(0.01 * max_values["dp"], max_values["dp"], 100)

    f_m = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000])

    nlam = (
        lambda q, dp: (
            -a[2] * q + np.sqrt(q**2 * (a[2] ** 2 - 4 * a[1] * a[3]) + 4 * a[3] * dp)
        )
        / 2
        / a[3]
    )
    Stlam = lambda n: f_m * 60 / (np.pi * n * n_max)
    # Lw = lambda Lws, q, dp: Lws + 10 * np.log10(q) + 20 * np.log10(dp)

    rows_all = []

    for fi in range(len(f_m)):  # expecting 8
        for dpi in dp:
            for qi in q:
                n = nlam(qi, dpi)
                if n < 0 or n > 1:
                    continue

                St = Stlam(n)

                dLwokt = -5 - 5 * (np.log10(St) + 0.4) ** 2
                lwokt_ges = level_add(dLwokt)

                lw = dLwokt[fi] - lwokt_ges  # scalar

                rows_all.append({"dp": dpi, "q": qi, "f": fi, "lw": lw})

    df = pd.DataFrame(rows_all)

    models = {}
    r2s = {}
    df["lw_pred"] = np.nan

    coeffs = {}

    for fi in range(8):
        g = df[df["f"] == fi].copy()

        X = np.c_[g["q"].to_numpy(), g["dp"].to_numpy(), np.ones(len(g))]
        y = g["lw"].to_numpy()

        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)

        y_pred = model.predict(X)

        models[fi] = model
        r2s[fi] = r2_score(y, y_pred)

        df.loc[g.index, "lw_pred"] = y_pred

        print(f"fi={fi}  coef [q, dp, 1]={model.coef_}  R2={r2s[fi]:.6f}")

        coeffs.update({(fi + 1, idx + 1): val for idx, val in enumerate(model.coef_)})
    return coeffs


def prepare_fan_yaml(
    data: Dict,
    fan_set: dict,
    max_pressure_loss_in_problem: float | None = None,
    fan_edge_volume_flow: dict[str, dict] | None = None,
    max_volume_flow_in_problem: float | None = None,
) -> Dict:
    """Prepare fan YAML dictionary for Pyomo model input.

    Args:
        data (Dict): Nested dict of fan data with under/overestimation hyperplanes, costs, etc.
        max_pressure_loss_in_problem (float, optional): Threshold for filtering pressure points.
            Defaults to 1e6.
        fan_edge_volume_flow (dict, optional): Mapping scenario -> edge -> max volume flow.
            Defaults to {}.
        max_volume_flow_in_problem (float, optional): Threshold for filtering volume flow points.
            Defaults to 1e6.

    Returns:
        Dict: Filtered dictionary with fan hyperplanes, coefficients, and costs.
    """
    max_pressure_loss_in_problem = max_pressure_loss_in_problem or 1e6
    max_volume_flow_in_problem = max_volume_flow_in_problem or 1e6
    fan_edge_volume_flow = fan_edge_volume_flow or {}

    fan_set_without_n = [(i, j, p, d) for (i, j, p, d, _) in fan_set]

    unique_ps = sorted({p for (p, _) in data.keys()})
    p_to_ds = defaultdict(list)
    for product_line, diameter in data.keys():
        p_to_ds[product_line].append(diameter)

    filtered_data = {
        "fan_hyperplanes_underestimation_pre_set": {},
        "fan_hyperplanes_underestimation_specific_pre_set": {},
        "fan_hyperplanes_underestimation_intercept": {},
        "fan_hyperplanes_underestimation_slope_volume_flow": {},
        "fan_hyperplanes_underestimation_slope_pressure": {},
        "fan_hyperplanes_overestimation_pre_set": {},
        "fan_hyperplanes_overestimation_specific_pre_set": {},
        "fan_hyperplanes_overestimation_intercept": {},
        "fan_hyperplanes_overestimation_slope_volume_flow": {},
        "fan_product_line": {None: unique_ps},
        "fan_diameter": p_to_ds,
        "fan_costs": {(p, d): data[p, d]["fan_costs"] for (p, d) in data.keys()},
        "fan_power_loss_max": {
            (p, d): data[p, d]["fan_power_loss_max"] for (p, d) in data.keys()
        },
        "fan_volume_flow_max": {
            (p, d): data[p, d]["fan_volume_flow_max"] for (p, d) in data.keys()
        },
        "fan_pressure_max": {
            (p, d): data[p, d]["fan_pressure_max"] for (p, d) in data.keys()
        },
        "fan_pressure_coefficients": {
            (p, d, idx): val
            for (p, d) in data.keys()
            for idx, val in data[p, d]["fan_pressure_coefficients"].items()
        },
        "fan_power_coefficients": {
            (p, d, idx): val
            for (p, d) in data.keys()
            for idx, val in data[p, d]["fan_power_coefficients"].items()
        },
    }

    if "fan_flow_noise_coefficients" in [
        keys for vals in data.values() for keys in vals.keys()
    ]:
        filtered_data["fan_rotational_speed_max"] = {
            (p, d): data[p, d]["fan_rotational_speed_max"] for (p, d) in data.keys()
        }
        filtered_data["fan_flow_noise_coefficients"] = {
            (p, d, *idx): val
            for (p, d) in data.keys()
            for idx, val in data[p, d]["fan_flow_noise_coefficients"].items()
        }

    for (p, d), values in data.items():
        intercepts = values["underestimation"]["intercept"]
        grad_vf = values["underestimation"]["grad_volume_flow"]
        grad_pr = values["underestimation"]["grad_pressure"]
        over_intercepts = values["overestimation"]["intercept"]
        over_grad_vf = values["overestimation"]["grad_volume_flow"]

        volume_flows_lb = values["underestimation"]["point_volume_flow"]
        pressures_lb = values["underestimation"]["point_pressure"]
        volume_flows_ub = values["overestimation"]["point_volume_flow"]

        dq = get_point_distance(volume_flows_lb)
        dp = get_point_distance(pressures_lb)
        dq_ub = get_point_distance(volume_flows_ub)

        # valid indices for underestimation
        valid_indices = [
            i
            for i, (v, pr) in enumerate(zip(volume_flows_lb, pressures_lb))
            if v < max_volume_flow_in_problem + dq
            and pr < max_pressure_loss_in_problem + dp
        ]

        # underestimation sets (1-based indexing for Pyomo)
        filtered_data["fan_hyperplanes_underestimation_pre_set"][(p, d)] = [
            i + 1 for i in valid_indices
        ]

        for i in valid_indices:
            filtered_data["fan_hyperplanes_underestimation_intercept"][
                (p, d, i + 1)
            ] = intercepts[i]
            filtered_data["fan_hyperplanes_underestimation_slope_volume_flow"][
                (p, d, i + 1)
            ] = grad_vf[i]
            filtered_data["fan_hyperplanes_underestimation_slope_pressure"][
                (p, d, i + 1)
            ] = grad_pr[i]

        # valid indices for overestimation
        valid_indices_ub = [
            i
            for i, v in enumerate(volume_flows_ub)
            if v < max_volume_flow_in_problem + dq
        ]
        filtered_data["fan_hyperplanes_overestimation_pre_set"][(p, d)] = [
            i + 1 for i in valid_indices_ub
        ]
        for i in valid_indices_ub:
            filtered_data["fan_hyperplanes_overestimation_slope_volume_flow"][
                (p, d, i + 1)
            ] = over_grad_vf[i]
            filtered_data["fan_hyperplanes_overestimation_intercept"][(p, d, i + 1)] = (
                over_intercepts[i]
            )

        # Apply filtering based on thresholds per scenario & edge
        for s, edges in fan_edge_volume_flow.items():
            for e, max_q in edges.items():
                # only add if in fan_set
                if (e[0], e[1], p, d) in fan_set_without_n:
                    e_in, e_out = e

                    valid_indices_specific = [
                        i
                        for (i, v, pr) in zip(
                            valid_indices, volume_flows_lb, pressures_lb
                        )
                        if v < max_q + dq and pr < max_pressure_loss_in_problem + dp
                    ]
                    filtered_data["fan_hyperplanes_underestimation_specific_pre_set"][
                        (s, e_in, e_out, p, d)
                    ] = [i + 1 for i in valid_indices_specific]
                    # print(len(filtered_data["fan_hyperplanes_underestimation_specific_pre_set"]))
                    valid_indices_specific_ub = [
                        i for i, v in enumerate(volume_flows_ub) if v < max_q + dq_ub
                    ]
                    filtered_data["fan_hyperplanes_overestimation_specific_pre_set"][
                        (s, e_in, e_out, p, d)
                    ] = [i + 1 for i in valid_indices_specific_ub]

    return filtered_data


# --- Duct utils ---


def get_duct_max_dimensions(data: Dict) -> Tuple[Dict, float, float]:
    """Return per-duct max dimensions and overall max width/height."""
    duct_max = {
        (i, j): (data["duct_width_max"][i, j], data["duct_height_max"][i, j])
        for i, j in data["duct_width_max"].keys()
    }
    return (
        duct_max,
        max(data["duct_width_max"].values()),
        max(data["duct_height_max"].values()),
    )


def filter_hyperplanes(
    data_dict: Dict, width_thresh: float, height_thresh: float, duct_edge_data: Dict
) -> Dict:
    """Filter hyperplanes dictionary based on duct width/height thresholds."""
    filtered = defaultdict(dict)
    for prefix in ["duct_friction", "duct_area2", "inverse"]:
        indices = data_dict[f"{prefix}_hyperplanes_set"]
        width_points = data_dict[f"{prefix}_point_width"]

        if prefix != "inverse":
            height_points = data_dict[f"{prefix}_point_height"]
            valid = [
                i
                for i in indices
                if width_points[i] < width_thresh and height_points[i] < height_thresh
            ]
            filtered[f"{prefix}_slope_height"] = {
                i: data_dict[f"{prefix}_slope_height"][i] for i in valid
            }
        else:
            valid = [i for i in indices if width_points[i] < width_thresh]

        filtered[f"{prefix}_hyperplanes_set"] = valid
        filtered[f"{prefix}_intercept"] = {
            i: data_dict[f"{prefix}_intercept"][i] for i in valid
        }
        filtered[f"{prefix}_slope_width"] = {
            i: data_dict[f"{prefix}_slope_width"][i] for i in valid
        }

        for (i, j), (max_w, max_h) in duct_edge_data.items():
            if prefix != "inverse":
                valid_specific = [
                    idx
                    for idx in indices
                    if width_points[idx] < max_w and height_points[idx] < max_h
                ]
                filtered[f"{prefix}_hyperplanes_specific_pre_set"][
                    (i, j)
                ] = valid_specific

    return filtered
