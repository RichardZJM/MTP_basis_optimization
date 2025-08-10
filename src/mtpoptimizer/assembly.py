import numpy as np
from collections import OrderedDict


def assemble_new_tree(original_mtp, mask, theta: np.ndarray = None):
    """
    Assembles a new, pruned MTP tree from an original tree and a mask.

    Args:
        original_mtp (dict): A dictionary containing initial MTP data.
        mask (np.ndarray): A boolean mask for the scalar moments to keep.
        theta (np.ndarray, optional): A vector of new coefficients. If provided,
            a trained potential is written. The first `species_count` elements
            are species coefficients, and the rest are moment coefficients.

    Returns:
        dict: A dictionary containing the pruned MTP data.
    """
    # Build a parent lookup for efficient upwards traversal of the tree
    parents_lookup = [[] for _ in range(original_mtp["alpha_moments_count"])]
    for p1, p2, k, child in original_mtp["alpha_index_times"]:
        parents_lookup[child].append((p1, p2))

    # Identify all nodes that must be kept using a BFS traversal
    nodes_to_keep = np.zeros(original_mtp["alpha_moments_count"], dtype=bool)
    queue = []
    for i, keep in enumerate(mask):
        if keep:
            node_idx = original_mtp["alpha_moment_mapping"][i]
            if not nodes_to_keep[node_idx]:
                nodes_to_keep[node_idx] = True
                queue.append(node_idx)

    head = 0
    while head < len(queue):
        child_idx = queue[head]
        head += 1
        for p1, p2 in parents_lookup[child_idx]:
            if not nodes_to_keep[p1]:
                nodes_to_keep[p1] = True
                queue.append(p1)
            if not nodes_to_keep[p2]:
                nodes_to_keep[p2] = True
                queue.append(p2)

    # Create a mapping from old node indices to new, compact indices
    old_to_new_node_map = {}
    new_idx_counter = 0
    for old_idx in range(original_mtp["alpha_moments_count"]):
        if nodes_to_keep[old_idx]:
            old_to_new_node_map[old_idx] = new_idx_counter
            new_idx_counter += 1

    # Identify which radial functions (mus) are still in use and remap them
    intermediate_basic_indices = []
    used_mus = set()
    for old_idx, basic_info in enumerate(original_mtp["alpha_index_basic"]):
        if nodes_to_keep[old_idx]:
            intermediate_basic_indices.append(basic_info)
            used_mus.add(basic_info[0])

    sorted_used_mus = sorted(list(used_mus))
    old_mu_to_new_mu_map = {
        old_mu: new_mu for new_mu, old_mu in enumerate(sorted_used_mus)
    }

    # Rebuild the basic indices and graph edges with the new mappings
    new_basic_indices = []
    for old_mu, l, n, k in intermediate_basic_indices:
        new_mu = old_mu_to_new_mu_map.get(old_mu, old_mu)
        new_basic_indices.append([new_mu, l, n, k])

    new_times_indices = []
    for p1_old, p2_old, k, child_old in original_mtp["alpha_index_times"]:
        if nodes_to_keep[child_old]:
            new_times_indices.append(
                [
                    old_to_new_node_map[p1_old],
                    old_to_new_node_map[p2_old],
                    k,
                    old_to_new_node_map[child_old],
                ]
            )

    new_scalar_indices = []
    for i, keep in enumerate(mask):
        if keep:
            old_node_idx = original_mtp["alpha_moment_mapping"][i]
            new_scalar_indices.append(old_to_new_node_map[old_node_idx])

    # Assemble the final MTP dictionary
    new_mtp = OrderedDict()
    # Copy non-structural, non-trained parameters first
    for key in [
        "version",
        "potential_name",
        "potential_tag",
        "radial_basis_type",
        "min_dist",
        "max_dist",
        "species_count",
    ]:
        if key in original_mtp:
            new_mtp[key] = original_mtp[key]

    new_mtp["radial_basis_size"] = original_mtp["radial_basis_size"]
    new_mtp["radial_funcs_count"] = len(used_mus)

    # Add structural components
    new_mtp["alpha_moments_count"] = len(old_to_new_node_map)
    new_mtp["alpha_index_basic_count"] = len(new_basic_indices)
    new_mtp["alpha_index_basic"] = new_basic_indices
    new_mtp["alpha_index_times_count"] = len(new_times_indices)
    new_mtp["alpha_index_times"] = new_times_indices
    new_mtp["alpha_scalar_moments"] = len(new_scalar_indices)
    new_mtp["alpha_moment_mapping"] = new_scalar_indices

    # If theta is provided, build a trained potential
    if theta is not None:
        num_species = original_mtp["species_count"]
        expected_size = num_species + np.sum(mask)
        if len(theta) != expected_size:
            raise ValueError(
                f"Theta vector has wrong size. Expected {expected_size}, got {len(theta)}"
            )

        new_mtp["is_trained"] = True
        if "scaling" in original_mtp:
            new_mtp["scaling"] = original_mtp["scaling"]

        # Prune radial coefficients based on used radial functions (mus)
        if "radial_coeffs" in original_mtp:
            new_radial_coeffs = OrderedDict()
            for key, coeff_lists in original_mtp["radial_coeffs"].items():
                pruned_coeffs = [coeff_lists[old_mu] for old_mu in sorted_used_mus]
                new_radial_coeffs[key] = pruned_coeffs
            new_mtp["radial_coeffs"] = new_radial_coeffs

        # Split theta into species and moment coefficients
        new_mtp["species_coeffs"] = list(theta[:num_species])
        new_mtp["moment_coeffs"] = list(theta[num_species:])
    else:
        # Otherwise, the potential is untrained
        new_mtp["is_trained"] = False

    return new_mtp
