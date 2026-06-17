import h5py
import os
import numpy as np
import ast
from matplotlib.font_manager import FontProperties
import hashlib


def contains_string(file_path, target_string):
    with h5py.File(file_path, "r") as f:

        def recursive_search(h5_obj):
            if isinstance(h5_obj, h5py.Dataset):
                # If dataset contains ASCII strings
                if h5_obj.dtype.kind in {
                    "S",
                    "O",
                }:  # S = bytes, O = object (e.g. variable length strings)
                    data = h5_obj[()]
                    if isinstance(data, bytes):
                        return target_string in data.decode()
                    elif isinstance(data, str):
                        return target_string in data
                    elif isinstance(data, (list, tuple, np.ndarray)):
                        # iterate over each item
                        return any(
                            (
                                target_string in d.decode()
                                if isinstance(d, bytes)
                                else target_string in d
                            )
                            for d in data
                        )
            elif isinstance(h5_obj, h5py.Group):
                for key in h5_obj:
                    if recursive_search(h5_obj[key]):
                        return True
            return False

        return recursive_search(f)


def normalize_edge_tuple(x):
    # bytes -> string
    if isinstance(x, bytes):
        x = x.decode()

    # remove whitespace
    x = x.strip()

    # Case: "('Z3', '15A0')"
    try:
        parsed = ast.literal_eval(x)
        if isinstance(parsed, tuple):
            return tuple(str(i).strip() for i in parsed)
    except Exception:
        pass

    # Case: "(Z3, 15A0)"
    if x.startswith("(") and x.endswith(")"):
        x = x[1:-1]

    return tuple(part.strip().strip("'").strip('"') for part in x.split(","))


def sort_list_of_dicts(data, by, descending=False):
    """
    Sort each dict in a list of dicts by the values in key `by`.
    Each dict is sorted independently.
    """
    return [sort_dict_of_arrays(d, by, descending) for d in data]


def sort_dict_of_arrays(results_dict, by, descending=False):
    """
    Sort all 1D arrays in a dict according to d[by].

    Parameters
    ----------
    d : dict
        Dict whose values are 1D numpy arrays of the same length.
    by : str
        Key of the array to sort by.
    descending : bool
        If True, sort in descending order.

    Returns
    -------
    dict
        New dict with all arrays sorted consistently.
    """
    idx = np.argsort(results_dict[by], kind="stable")
    if descending:
        idx = idx[::-1]

    return {
        k: (
            v[idx]
            if isinstance(v, np.ndarray)
            and v.ndim == 1
            and len(v) == len(results_dict[by])
            else v
        )
        for k, v in results_dict.items()
    }


def load_component_costs_from_connected_optimisation(file_path, which_component="duct"):
    if which_component == "duct":
        cost_name = "total_duct_costs"
    elif which_component == "silencer":
        cost_name = "total_silencer_costs"
    else:
        raise NameError(
            f"Component name is {which_component} but should be 'duct' or 'silencer'"
        )

    with h5py.File(file_path, "r") as h5_file:
        return h5_file["Optimisation Components/Expression/" + cost_name][0]["value"]


def process_dicts(dict_from_h5, sortby="exact_lcc"):
    # if dict_from_h5:

    # Get all keys (like 'invest_costs', etc.) from one of the entries
    value_keys = list(next(iter(dict_from_h5.values())).keys())

    # For each key, collect all values across the dicts
    collected = {
        key: [entry[key] for entry in dict_from_h5.values()] for key in value_keys
    }

    # Sort by the first metric (e.g., 'invest_costs')
    sort_key = sortby if sortby in value_keys else value_keys[2]
    sorted_indices = np.argsort(collected[sort_key])

    # Build a sorted output dictionary
    sorted_dict = {
        key: np.array(values)[sorted_indices] for key, values in collected.items()
    }
    return sorted_dict


def create_standard_list(filename_list, planning_mode, use_connected_opt=True):
    standard_dict_list = []
    for x in filename_list:
        h5_data = extract_h5_data(
            x, planning_mode=planning_mode, use_connected_opt=use_connected_opt
        )
        standard_dict = process_dicts(h5_data)
        standard_dict_list.append(standard_dict)
    return standard_dict_list


def create_standard_list_lab(filename_list):
    standard_dict_list = []
    for x in filename_list:
        # print(x)
        h5_data = extract_h5_data_lab(x)
        standard_dict = process_dicts(h5_data, "power_consumption")
        standard_dict_list.append(standard_dict)
    return standard_dict_list


def merge_standard_list(standard_dict_list):
    merged_standard_dict = standard_dict_list[0]
    for i in standard_dict_list[1:]:
        merged_standard_dict = {
            key: np.array([*val1, *val2])
            for (key, val1), val2 in zip(merged_standard_dict.items(), i.values())
        }
    return merged_standard_dict


def extract_h5_data(folder_path, planning_mode, use_connected_opt, excluded_folders=()):

    data_dict = {}

    # Iterate through all files in the folder
    # for file_name in os.listdir(folder_path):
    for root, dirs, files in os.walk(folder_path):
        dirs[:] = [d for d in dirs if d not in excluded_folders]
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if file_path.endswith(".h5"):  # Ensure it's an HDF5 file
                # print(file_name)
                try:
                    with h5py.File(file_path, "r") as h5_file:
                        expression_group = h5_file["Optimisation Components"][
                            "Expression"
                        ]
                        invest_costs = 0
                        power_consumption = 0
                        exact_power_consumption, exact_lcc = 0, 0
                        duct_costs, vfc_costs, fan_costs, sil_costs = 0, 0, 0, 0
                        duct_volume = 0
                        optimality_gap = None
                        computation_time = None
                        equipment_investment_cost = None
                        max_noise_value = 0
                        additional_cost = 0

                        if "Optimization Type" in h5_file:

                            if (
                                h5_file["Optimization Type/Planning mode"][()].decode()
                                == planning_mode
                            ):

                                if use_connected_opt:

                                    file_name_connected_opt = h5_file[
                                        "Optimization Type/Connected Optimisation"
                                    ][()].decode()
                                    file_name_connected_opt = (
                                        file_name_connected_opt + ".h5"
                                        if not file_name_connected_opt[-3:] == ".h5"
                                        else file_name_connected_opt
                                    )
                                    file_path_connected_optimisation = os.path.join(
                                        root, file_name_connected_opt
                                    )
                                    if planning_mode == "Configuration":
                                        duct_costs = load_component_costs_from_connected_optimisation(
                                            file_path_connected_optimisation, "duct"
                                        )
                                        additional_cost = duct_costs
                                    elif planning_mode == "Topology":
                                        sil_costs = load_component_costs_from_connected_optimisation(
                                            file_path_connected_optimisation, "silencer"
                                        )
                                        additional_cost = sil_costs
                                elif use_connected_opt:
                                    raise ValueError("No connected optimization found.")

                                # Access the subdatasets under 'expression'

                                # Extract invest_costs if available
                                if "total_invest_costs" in expression_group:
                                    dataset = expression_group["total_invest_costs"]
                                    invest_costs = dataset[0]["value"] + additional_cost

                                # Extract operating_costs if available
                                if "fan_power_consumption" in expression_group:
                                    dataset = expression_group["fan_power_consumption"]
                                    power_consumption = dataset[0]["value"]

                                if "fan_energy_costs" in expression_group:
                                    dataset = expression_group["fan_energy_costs"]
                                    fan_energy_costs = dataset[0]["value"]

                                if (
                                    "Exact Power Consumption"
                                    in h5_file["Postprocessing"]
                                ):
                                    dataset = h5_file["Postprocessing"][
                                        "Exact Power Consumption"
                                    ]
                                    exact_power_consumption = dataset[()]

                                if "Wallclock time" in h5_file["Solver"]:
                                    dataset = h5_file["Solver"]["Wallclock time"]
                                    computation_time = dataset[()]

                                if (
                                    "Exact Life-Cycle Costs"
                                    in h5_file["Postprocessing"]
                                ):
                                    dataset = h5_file["Postprocessing"][
                                        "Exact Life-Cycle Costs"
                                    ]
                                    exact_lcc = dataset[()] + additional_cost

                                if "total_silencer_costs" in expression_group:
                                    dataset = expression_group["total_silencer_costs"]
                                    sil_costs = dataset[0]["value"]

                                if "total_duct_costs" in expression_group:
                                    dataset = expression_group["total_duct_costs"]
                                    duct_costs = dataset[0]["value"]

                                if "total_vfc_costs" in expression_group:
                                    dataset = expression_group["total_vfc_costs"]
                                    vfc_costs = dataset[0]["value"]

                                if "total_fan_costs" in expression_group:
                                    dataset = expression_group["total_fan_costs"]
                                    fan_costs = dataset[0]["value"]

                                if "duct_volume" in expression_group:
                                    dataset = expression_group["duct_volume"]
                                    duct_volume = dataset[0]["value"]

                                if "equipment_investment_cost" in expression_group:
                                    dataset = expression_group[
                                        "equipment_investment_cost"
                                    ]
                                    equipment_investment_cost = dataset[0]["value"]

                                if (
                                    "fan_ind_purchase"
                                    in h5_file["Optimisation Components"]["Variable"]
                                ):
                                    dataset = h5_file["Optimisation Components"][
                                        "Variable"
                                    ]["fan_ind_purchase"][:]
                                    fan_ind_purchase = dict(
                                        zip(
                                            [
                                                x.decode("utf-8")
                                                for x in dataset["fan_set"]
                                            ],
                                            dataset["value"],
                                        )
                                    )

                                if "Duct Variations" in h5_file:
                                    dv = h5_file["Duct Variations"]
                                    max_allowed_hori_velo = dv[
                                        "Maximum duct velocity in horizontal ducts"
                                    ][()]
                                    max_allowed_vert_velo = dv[
                                        "Maximum duct velocity in vertical ducts"
                                    ][()]
                                    max_allowed_height = dv[
                                        "Maximum duct height in horizontal ducts"
                                    ][()]

                                    mean_vertical_velocity = dv[
                                        "duct_velocities/vertical/mean"
                                    ][()]
                                    mean_horizontal_velocity = dv[
                                        "duct_velocities/horizontal/mean"
                                    ][()]

                                elif (
                                    "duct_height"
                                    in h5_file["Optimisation Components/Variable"]
                                ):
                                    E_duct_vertical = h5_file[
                                        "Optimisation Components/Set/E_duct_vertical"
                                    ][:]
                                    E_duct_vertical = [
                                        normalize_edge_tuple(x) for x in E_duct_vertical
                                    ]
                                    duct_height = h5_file[
                                        "Optimisation Components/Variable/duct_height"
                                    ][:]
                                    duct_height_val = duct_height["value"][:]
                                    duct_height_edge = duct_height["E_duct"][:]

                                    relevant_duct_heights = [
                                        val
                                        for edge, val in zip(
                                            duct_height_edge, duct_height_val
                                        )
                                        if normalize_edge_tuple(edge)
                                        not in E_duct_vertical
                                    ]
                                    max_allowed_height = max(relevant_duct_heights)

                                    max_allowed_hori_velo = 0
                                    max_allowed_vert_velo = 0
                                    mean_horizontal_velocity = 0
                                    mean_vertical_velocity = 0
                                else:
                                    max_allowed_height = 0
                                    max_allowed_hori_velo = 0
                                    max_allowed_vert_velo = 0
                                    mean_horizontal_velocity = 0
                                    mean_vertical_velocity = 0
                                if (
                                    "max_noise_scenarios"
                                    in h5_file["Optimisation Components"]["Set"]
                                ):
                                    max_noise_scenarios = h5_file[
                                        "Optimisation Components"
                                    ]["Set"]["max_noise_scenarios"][()]
                                    max_noise_value = 0
                                    for s in max_noise_scenarios:
                                        curr_max_noise_value = max(
                                            h5_file[
                                                "Optimisation Components/Variable/Scenario"
                                            ][s]["sound_pressure_level_room"]["value"]
                                        )
                                        if curr_max_noise_value > max_noise_value:
                                            max_noise_value = curr_max_noise_value

                                if "Problem Definition" in h5_file:
                                    lb = h5_file["Problem Definition"]["Lower bound"][
                                        ()
                                    ]
                                    ub = h5_file["Problem Definition"]["Upper bound"][
                                        ()
                                    ]
                                    postprocess_gap = (ub - lb) / lb

                                if "Solver" in h5_file and "Log" in h5_file["Solver"]:
                                    log_data = h5_file["Solver"]["Log"][()]

                                    # Handle string types and arrays
                                    found = False
                                    if isinstance(log_data, bytes):
                                        found = (
                                            "Optimal solution found"
                                            in log_data.decode()
                                        )
                                    elif isinstance(log_data, str):
                                        found = "Optimal solution found" in log_data
                                    elif isinstance(
                                        log_data, (list, tuple, np.ndarray)
                                    ):
                                        found = any(
                                            (
                                                "Optimal solution found" in d.decode()
                                                if isinstance(d, bytes)
                                                else "Optimal solution found" in d
                                            )
                                            for d in log_data
                                        )

                                    if found:
                                        optimality_gap = 0
                                    else:
                                        optimality_gap = h5_file["Solver"][
                                            "Convergence"
                                        ][-1]["Gap in %"]

                                # Store values in dictionary
                                data_dict[file_name] = {
                                    "invest_costs": invest_costs,
                                    "power_consumption": power_consumption,
                                    "exact_power_consumption": exact_power_consumption,
                                    "duct_costs": duct_costs,
                                    "vfc_costs": vfc_costs,
                                    "fan_costs": fan_costs,
                                    "sil_costs": sil_costs,
                                    "duct_volume": duct_volume,
                                    "fan_ind_purchase": fan_ind_purchase,
                                    "fan_energy_costs": fan_energy_costs,
                                    "postprocess_gap": postprocess_gap,
                                    "optimality_gap": optimality_gap,
                                    "exact_lcc": exact_lcc,
                                    "computation_time": computation_time,
                                    "equipment_investment_cost": equipment_investment_cost,
                                    "max_noise_value": max_noise_value,
                                    "filename": file_name,
                                    "max_horizontal_velocity": max_allowed_hori_velo,
                                    "max_vertical_velocity": max_allowed_vert_velo,
                                    "max_height": max_allowed_height,
                                    "mean_horizontal_velocity": mean_horizontal_velocity,
                                    "mean_vertical_velocity": mean_vertical_velocity,
                                }

                            else:
                                continue

                except Exception as e:
                    print(f"Error processing {folder_path}/{file_name}: {e}")
    if data_dict == {}:
        data_dict = {
            "name": {
                "invest_costs": 0,
                "power_consumption": 0,
                "exact_power_consumption": 0,
                "duct_costs": 0,
                "vfc_costs": 0,
                "fan_costs": 0,
                "sil_costs": 0,
                "duct_volume": 0,
                "fan_ind_purchase": 0,
                "fan_energy_costs": 0,
                "postprocess_gap": 0,
                "optimality_gap": 0,
                "exact_lcc": 0,
                "computation_time": 0,
                "equipment_investment_cost": 0,
                "max_noise_value": 0,
                "filename": None,
            }
        }
    return data_dict


def extract_h5_data_fan_acoustics(folder_path, planning_mode):

    data_dict = {}

    # Iterate through all files in the folder
    # for file_name in os.listdir(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if file_path.endswith(".h5"):  # Ensure it's an HDF5 file
                # print(file_name)
                try:
                    with h5py.File(file_path, "r") as h5_file:

                        if "Optimization Type" in h5_file:

                            if (
                                h5_file["Optimization Type/Planning mode"][()].decode()
                                == planning_mode
                            ):

                                if (
                                    "Exact Life-Cycle Costs"
                                    in h5_file["Postprocessing"]
                                ):
                                    dataset = h5_file["Postprocessing"][
                                        "Exact Life-Cycle Costs"
                                    ]
                                    exact_lcc = dataset[()]

                                if (
                                    "Postprocessing/Acoustics: Fan Sound Power Level"
                                    in h5_file
                                ):
                                    octave_frequencies = h5_file[
                                        "Postprocessing/Acoustics: Fan Sound Power Level"
                                    ]["FrequencyIndex"]

                                    spl_approx = h5_file[
                                        "Postprocessing/Acoustics: Fan Sound Power Level"
                                    ]["SPL_Approx"]

                                    spl_exact = h5_file[
                                        "Postprocessing/Acoustics: Fan Sound Power Level"
                                    ]["SPL_Exact"]

                                # Store values in dictionary
                                data_dict[file_name] = {
                                    "spl_approx": {
                                        fi: spl
                                        for fi, spl in zip(
                                            octave_frequencies, spl_approx
                                        )
                                    },
                                    "spl_exact": {
                                        fi: spl
                                        for fi, spl in zip(
                                            octave_frequencies, spl_exact
                                        )
                                    },
                                    "spl_diff": {
                                        fi: spl_e - spl_a
                                        for fi, spl_e, spl_a in zip(
                                            octave_frequencies, spl_exact, spl_approx
                                        )
                                    },
                                    "exact_lcc": exact_lcc,
                                    "filename": file_name,
                                }

                            else:
                                continue

                except Exception as e:
                    print(f"Error processing {folder_path}/{file_name}: {e}")
    if data_dict == {}:
        data_dict = {
            "name": {
                "spl_approx": {0: 0},
                "exact_lcc": 0,
                "filename": 0,
                "spl_exact": {0: 0},
                "spl_diff": {0: 0},
            }
        }
    return data_dict


def extract_h5_data_lab(folder_path, excluded_folders=()):

    data_dict = {}

    # same as h5_data but for lab. needed as lab data is actually outdated in terms of h5 files.

    # Iterate through all files in the folder
    # for file_name in os.listdir(folder_path):
    for root, dirs, files in os.walk(folder_path):
        dirs[:] = [d for d in dirs if d not in excluded_folders]
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if file_path.endswith(".h5"):  # Ensure it's an HDF5 file
                # print(file_name)
                try:
                    with h5py.File(file_path, "r") as h5_file:
                        expression_group = h5_file["Optimisation Components"][
                            "Expression"
                        ]
                        # expression_group = h5_file["Expression"]
                        invest_costs = 0
                        power_consumption = 0
                        computation_time = None
                        postprocess_gap = 0
                        operating_costs = 0
                        total_silencer_costs = 0
                        total_VFC_costs = 0
                        total_fan_costs = 0
                        additional_equipment_costs = 0

                        # Extract invest_costs if available
                        if "invest_costs" in expression_group:
                            dataset = expression_group["invest_costs"]
                            invest_costs = dataset[0]["value"]
                            total_VFC_costs = expression_group["total_VFC_costs"][()]
                            total_fan_costs = expression_group["total_fan_costs"][()]
                            additional_equipment_costs = expression_group[
                                "additional_equipment_costs"
                            ][()]

                        if "total_silencer_costs" in expression_group:
                            total_silencer_costs = expression_group[
                                "total_silencer_costs"
                            ][()]

                        # Extract operating_costs if available
                        if "power_consumption" in expression_group:
                            dataset = expression_group["power_consumption"]
                            power_consumption = dataset[0]["value"]

                        if "operating_costs" in expression_group:
                            dataset = expression_group["operating_costs"]
                            operating_costs = dataset[0]["value"]

                        if "Wallclock time" in h5_file["Solver Output"]:
                            dataset = h5_file["Solver Output"]["Wallclock time"]
                            computation_time = dataset[()]

                        if "Problem Definition" in h5_file:
                            lb = h5_file["Problem Definition"]["Lower bound"][()]
                            ub = h5_file["Problem Definition"]["Upper bound"][()]
                            postprocess_gap = (ub - lb) / lb

                        # Store values in dictionary
                        data_dict[file_name] = {
                            "invest_costs": invest_costs,
                            "power_consumption": power_consumption,
                            "operating_costs": operating_costs,
                            "postprocess_gap": postprocess_gap,
                            "exact_lcc": invest_costs + operating_costs,
                            "computation_time": computation_time,
                            "filename": file_name,
                            "total_silencer_costs": total_silencer_costs,
                            "total_VFC_costs": total_VFC_costs,
                            "total_fan_costs": total_fan_costs,
                            "additional_equipment_costs": additional_equipment_costs,
                        }

                except Exception as e:
                    print(f"Error processing {folder_path}/{file_name}: {e}")
    return data_dict


def add_legend_right(ax, legend_entries):

    fontsize = 9
    arial_font = FontProperties(family="Arial", style="italic", size=fontsize)

    handles, labels = ax.get_legend_handles_labels()
    custom_order = np.arange(len(legend_entries) - 1, -1, -1)

    handles = [handles[i] for i in custom_order]
    labels = [labels[i] for i in custom_order]

    # upper legend
    leg1 = ax.legend(
        handles[:1],
        labels[:1],
        loc="center left",
        bbox_to_anchor=(1.0, 0.64),
        frameon=False,
        alignment="left",
        handletextpad=0.8,
        borderpad=0.3,
        labelspacing=0.5,
        fontsize=fontsize,
        prop=arial_font,
        # prop={'weight': 'bold'}
    )
    ax.add_artist(leg1)

    # lower legend
    leg2 = ax.legend(
        handles[1:],
        labels[1:],
        loc="center left",
        bbox_to_anchor=(1.0, 0.33),
        title="kapitalbezogene\nKosten:",
        frameon=False,
        alignment="left",
        handletextpad=0.8,
        borderpad=0.3,
        labelspacing=0.3,
        fontsize=fontsize,
        title_fontproperties={"family": "Arial", "style": "italic", "size": fontsize},
        prop=arial_font,
    )
    return ax


def add_legend_below(ax, legend_entries):
    fontsize = 10

    handles, labels = ax.get_legend_handles_labels()
    custom_order = np.arange(len(legend_entries) - 1, -1, -1)

    handles = [handles[i] for i in custom_order]
    labels = [labels[i] for i in custom_order]

    y = -0.3  # vertical position below the axes

    # first label on its own
    leg1 = ax.legend(
        handles[:1],
        labels[:1],
        loc="upper left",
        bbox_to_anchor=(0.00, y),
        frameon=False,
        fontsize=fontsize,
        handletextpad=0.6,
        borderaxespad=0.0,
    )
    ax.add_artist(leg1)

    # remaining entries in one row -> wider, less high
    leg2 = ax.legend(
        handles[1:],
        labels[1:],
        loc="upper left",
        bbox_to_anchor=(0.38, y - fontsize / 100, 0.52, 0.10),
        mode="expand",
        ncols=2,
        frameon=False,
        fontsize=fontsize,
        handlelength=1.2,
        handletextpad=0.5,
        columnspacing=1.2,
        borderaxespad=0.0,
        alignment="left",
        title="kapitalbezogene Kosten:",
        title_fontproperties={"size": fontsize},
    )

    return ax


def save_data_raw(
    fig,
    serializer,
    out_directory,
    outname,
    filedata,
    caption,
    general_information=None,
    id_result_subfolder=None,
):

    def figure_id(figure_name: str, length: int = 8) -> str:
        return hashlib.sha1(figure_name.encode("utf-8")).hexdigest()[:length]

    if id_result_subfolder:
        out_directory_id = out_directory + id_result_subfolder
    else:
        out_directory_id = out_directory
    fig.savefig(out_directory + outname + ".svg")
    fig.savefig(out_directory + outname + ".pdf")
    fig.savefig(out_directory + outname + ".png")
    fig.savefig(out_directory_id + figure_id(outname) + ".svg")
    fig.savefig(out_directory_id + figure_id(outname) + ".pdf")
    fig.savefig(out_directory_id + figure_id(outname) + ".png")
    if general_information:
        serializer.add_custom_metadata_figure(general_information)

    caption += r"\figureid{" + figure_id(outname) + "}"
    serializer.add_custom_metadata_figure(
        {
            "based on hdf5-files": filedata,
            "Caption in Figure": caption,
            "figure_id": figure_id(outname),
        }
    )

    serializer.write_json_file(out_directory_id + figure_id(outname) + ".json")
    texfile = out_directory + outname + ".caption.tex"
    with open(texfile, "w", encoding="utf-8") as out:
        out.write(caption)
