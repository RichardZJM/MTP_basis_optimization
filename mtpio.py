import regex as re


# ===== PARSERS =====
def parse_mtp_file(filepath):
    """
    Reads and parses an MTP potential file from a given path.

    This function handles key-value pairs, nested lists like {..} and {{..}},
    and automatically converts numeric values to int/float.

    Args:
        filepath (str): The path to the MTP data file.

    Returns:
        dict: A dictionary containing the parsed data.
        None: If the file cannot be found or another error occurs during reading.
    """

    # Helper function to process the value string
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
            return value_str  # It's a string

    # --- Main function logic ---
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

        # Skip header, comments, or empty lines
        if not line or line.lower() == "mtp" or "=" not in line:
            continue

        key, value_str = line.split("=", 1)
        key = key.strip()

        # Parse the value and add it to the dictionary
        parsed_dict[key] = _parse_value(value_str)

    return parsed_dict


def write_mtp_file(mtp_dict, filepath):
    """
    Writes an MTP dictionary to a file in the standard MTP format.

    This function handles different value types and ensures a consistent
    key order in the output file.

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
            # Check if it's a list of lists (e.g., alpha_index_basic)
            if isinstance(value[0], list):
                # Format each sublist as "{i1, i2, ...}"
                inner_strings = [
                    f"{{{', '.join(map(str, sublist))}}}" for sublist in value
                ]
                # Join the sublist strings and wrap in "{{...}}"
                return f"{{{', '.join(inner_strings)}}}"
            else:
                # It's a simple list (e.g., alpha_moment_mapping)
                return f"{{{', '.join(map(str, value))}}}"

        # For numbers, strings, etc., just convert to string
        return str(value)

    # Use a set for efficient lookup of keys already written
    written_keys = set()

    try:
        with open(filepath, "w") as f:
            f.write("MTP\n")

            # Write keys in the standard order
            for key in KEY_ORDER:
                if key in mtp_dict:
                    value = mtp_dict[key]
                    # The original format uses 'alpha_scalar_moments' for the count
                    # but the dict might have 'alpha_scalar_moments_count'
                    key_to_write = (
                        "alpha_scalar_moments"
                        if key == "alpha_scalar_moments_count"
                        else key
                    )

                    formatted_value = _format_value(value)
                    f.write(f"{key_to_write} = {formatted_value}\n")
                    written_keys.add(key)

            # Write any remaining keys not in the standard order
            for key, value in mtp_dict.items():
                if key not in written_keys:
                    formatted_value = _format_value(value)
                    f.write(f"{key} = {formatted_value}\n")

        print(f"Successfully wrote MTP data to '{filepath}'")

    except IOError as e:
        print(f"An error occurred while writing to file: {e}")


if __name__ == "__main__":
    mtp6 = parse_mtp_file("6.almtp")
    write_mtp_file(mtp6, "tmp.almtp")
