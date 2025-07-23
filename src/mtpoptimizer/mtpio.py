import regex as re


def parse_mtp_file(filepath: str):
    """
    Reads and parses an MTP potential file from a given path.

    Args:
        filepath (str): The path to the MTP data file.

    Returns:
        dict: A dictionary containing the parsed data.
        None: If the file cannot be found or another error occurs during reading.
    """

    def _parse_value(value_str):
        value_str = value_str.strip()
        if not value_str:
            return ""
        if value_str.startswith("{{") and value_str.endswith("}}"):
            inner_lists = re.findall(r"\{.*?\}", value_str)
            return [_parse_value(lst) for lst in inner_lists]
        if value_str.startswith("{") and value_str.endswith("}"):
            items = value_str.strip("{}").split(",")
            return [int(item.strip()) for item in items]
        try:
            return int(value_str) if "." not in value_str else float(value_str)
        except ValueError:
            return value_str

    try:
        with open(filepath, "r") as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"Error: The file at '{filepath}' was not found.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading the file: {e}")
        return None

    parsed_dict = {}
    for line in lines:
        line = line.strip()

        if not line or line.lower() == "mtp" or "=" not in line:
            continue

        key, value_str = line.split("=", 1)
        key = key.strip()

        parsed_dict[key] = _parse_value(value_str)

    return parsed_dict


def write_mtp_file(mtp_dict: dict, filepath: str):
    """
    Writes an MTP dictionary to a file in the standard MTP format.

    Args:
        mtp_dict (dict): A dictionary containing the MTP data.
        filepath (str): The path to the output file.
    """
    # Define the standard order of keys for a readable MTP file
    KEY_ORDER = [
        "version",
        "potential_name",
        "species_count",
        "potential_tag",
        "radial_basis_type",
        "min_dist",
        "max_dist",
        "radial_basis_size",
        "radial_funcs_count",
        "radial_funcs",
        "alpha_moments_count",
        "alpha_index_basic_count",
        "alpha_index_basic",
        "alpha_index_times_count",
        "alpha_index_times",
        "alpha_scalar_moments",
        "alpha_moment_mapping",
        "alpha_coeffs",
    ]

    def _format_value(value):
        """Converts a Python object to its MTP string representation."""
        if isinstance(value, list):
            if not value:
                return "{}"
            if isinstance(value[0], list):
                inner_strings = [
                    f"{{{', '.join(map(str, sublist))}}}" for sublist in value
                ]
                return f"{{{', '.join(inner_strings)}}}"
            else:
                return f"{{{', '.join(map(str, value))}}}"

        return str(value)

    written_keys = set()

    try:
        with open(filepath, "w") as f:
            f.write("MTP\n")

            for key in KEY_ORDER:
                if key in mtp_dict:
                    value = mtp_dict[key]

                    formatted_value = _format_value(value)
                    f.write(f"{key} = {formatted_value}\n")
                    written_keys.add(key)

            # for key, value in mtp_dict.items():
            #     if key not in written_keys:
            #         formatted_value = _format_value(value)
            #         f.write(f"{key} = {formatted_value}\n")

        print(f"Successfully wrote MTP data to '{filepath}'")

    except IOError as e:
        print(f"An error occurred while writing to file: {e}")


if __name__ == "__main__":
    mtp6 = parse_mtp_file("6.almtp")
    write_mtp_file(mtp6, "tmp.almtp")
