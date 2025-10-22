"""
Module for reading and writing Moment Tensor Potential (MTP) files.

This module provides functionality to parse and write MTP files in the standard format. It handles both trained and untrained potentials.
"""

from typing import Dict, List, Union, Optional
import regex as re
from collections import OrderedDict


def parse_mtp_file(
    filepath: str,
) -> Optional[Dict[str, Union[str, int, float, List, bool, Dict]]]:
    """
    Parse a trained or untrained MTP potential file.

    This function reads and parses an MTP file. It does not process active learning information.

    Parameters
    ----------
    filepath : str
        Path to the MTP data file to parse

    Returns
    -------
    dict or None
        Dictionary containing the parsed MTP data with following key components:
        - Standard MTP parameters (version, species_count, etc.)
        - Tree structure (alpha_index_basic, alpha_index_times, etc.)
        - Trained coefficients if present (scaling, coefficients)
        - is_trained flag indicating presence of trained parameters
        Returns None if file cannot be opened or parsed
    """

    def _parse_scalar(s: str) -> Union[int, float, str]:
        try:
            return int(s)
        except ValueError:
            try:
                return float(s)
            except ValueError:
                return s

    def _parse_value(value_str):
        value_str = value_str.strip()
        if not value_str:
            return ""
        if value_str.startswith("{{") and value_str.endswith("}}"):
            inner_lists = re.findall(r"\{.*?\}", value_str)
            return [_parse_value(lst) for lst in inner_lists]
        if value_str.startswith("{") and value_str.endswith("}"):
            items_str = value_str.strip("{}")
            if not items_str:
                return []
            return [_parse_scalar(item.strip()) for item in items_str.split(",")]

        return _parse_scalar(value_str)

    try:
        with open(filepath, "r", errors="ignore") as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"Error: The file at '{filepath}' was not found.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading the file: {e}")
        return None

    parsed_dict = OrderedDict()
    is_trained = False
    trained_keys = {"scaling", "radial_coeffs", "species_coeffs", "moment_coeffs"}

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line or line.lower() == "mtp":
            i += 1
            continue

        if line.startswith("#MVS_v1.1"):
            break

        if "=" in line:
            key, value_str = line.split("=", 1)
            key = key.strip()
            parsed_dict[key] = _parse_value(value_str)
            if key in trained_keys:
                is_trained = True
            i += 1
        elif line == "radial_coeffs":
            is_trained = True
            radial_coeffs_data = OrderedDict()
            i += 1
            while i < len(lines) and (
                lines[i].strip().startswith("{")
                or re.match(r"\d+-\d+", lines[i].strip())
            ):
                block_line = lines[i].strip()
                if re.match(r"\d+-\d+", block_line):
                    current_key = block_line
                    radial_coeffs_data[current_key] = []
                    i += 1
                    while i < len(lines) and lines[i].strip().startswith("{"):
                        coeffs_line = lines[i].strip()
                        radial_coeffs_data[current_key].append(
                            _parse_value(coeffs_line)
                        )
                        i += 1
                else:
                    i += 1
            parsed_dict["radial_coeffs"] = radial_coeffs_data
        else:
            i += 1

    parsed_dict["is_trained"] = is_trained
    return parsed_dict


def write_mtp_file(
    mtp_dict: Dict[str, Union[str, int, float, List, bool, Dict]],
    filepath: str,
    write_trained: bool = True,
) -> None:
    """
    Write an MTP dictionary to a file in standard format.

    This function writes an MTP potential specification to a file, with optional exclusion of trained parameters.

    Parameters
    ----------
    mtp_dict : dict
        Dictionary containing MTP data with standard keys:
        - Standard parameters (version, species_count, etc.)
        - Tree structure (alpha_index_basic, alpha_index_times, etc.)
        - Optional trained parameters (scaling, coefficients)
    filepath : str
        Path where the MTP file should be written
    write_trained : bool, optional
        If False, excludes trained parameters (scaling, coefficients)
        Default is True
    """
    # This key order strictly follows the format of a trained MTP file example.
    KEY_ORDER = [
        "version",
        "potential_name",
        "scaling",
        "species_count",
        "potential_tag",
        "radial_basis_type",
        "min_dist",
        "max_dist",
        "radial_basis_size",
        "radial_funcs_count",
        "radial_coeffs",
        "alpha_moments_count",
        "alpha_index_basic_count",
        "alpha_index_basic",
        "alpha_index_times_count",
        "alpha_index_times",
        "alpha_scalar_moments",
        "alpha_moment_mapping",
        "species_coeffs",
        "moment_coeffs",
    ]

    TRAINED_KEYS = {"scaling", "radial_coeffs", "species_coeffs", "moment_coeffs"}

    data_to_write = mtp_dict.copy()
    if not write_trained:
        for key in TRAINED_KEYS:
            if key in data_to_write:
                del data_to_write[key]

    def _format_value(value):
        if isinstance(value, list):
            if not value:
                return "{}"
            if value and isinstance(value[0], list):
                return f"{{{', '.join(_format_value(v) for v in value)}}}"
            else:
                formatted_items = [
                    f"{v:.15e}" if isinstance(v, float) else str(v) for v in value
                ]
                return f"{{{', '.join(formatted_items)}}}"
        if isinstance(value, float):
            return f"{value:.15e}"
        return str(value)

    try:
        with open(filepath, "w") as f:
            f.write("MTP\n")
            written_keys = set()
            for key in KEY_ORDER:
                if key in data_to_write:
                    value = data_to_write[key]
                    written_keys.add(key)
                    if key == "radial_coeffs" and isinstance(value, dict):
                        f.write("radial_coeffs\n")
                        for block_key, block_lists in value.items():
                            f.write(f"\t{block_key}\n")
                            for sublist in block_lists:
                                f.write(f"\t\t{_format_value(sublist)}\n")
                    else:
                        f.write(f"{key} = {_format_value(value)}\n")

        print(f"Successfully wrote MTP data to '{filepath}'")
    except IOError as e:
        print(f"An error occurred while writing to file: {e}")
