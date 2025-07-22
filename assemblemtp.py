import numpy as np
from mtpio import parse_mtp_file, write_mtp_file


def assemble_new_tree(original_mtp, mask):
    """
    Assembles a new, pruned MTP tree that preserves the relative order of all nodes.

    This function rebuilds the tree by:
    1. Identifying all necessary parent moment descriptors.
    2. Creating an order-preserving map from old indices to new, minimal indices.
       (e.g., keeping [0, 1, 4, 5] maps them to [0, 1, 2, 3]).
    3. Identifying all necessary radial basis functions (by their 'mu' index).
    4. Re-indexing the 'mu' indices in an order-preserving way.
    5. Constructing a new, valid MTP dictionary.

    Args:
        original_mtp (dict): The dictionary of the original, unpruned MTP.
        mask (array-like): A boolean array where `True` indicates that the
                           corresponding scalar moment should be kept.

    Returns:
        dict: A new MTP dictionary representing the pruned and re-indexed tree.
    """
    # 1. Identify all moment descriptor nodes to keep by traversing up the graph
    parents_lookup = [[] for _ in range(original_mtp["alpha_moments_count"])]
    for p1, p2, k, child in original_mtp["alpha_index_times"]:
        parents_lookup[child].append((p1, p2))

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

    # 2. Create an ORDER-PRESERVING re-indexing map for moment descriptors
    old_to_new_node_map = {}
    new_idx_counter = 0
    # Iterate through all original nodes. If a node is kept, assign it the next
    # available new index. This inherently preserves the relative order.
    for old_idx in range(original_mtp["alpha_moments_count"]):
        if nodes_to_keep[old_idx]:
            old_to_new_node_map[old_idx] = new_idx_counter
            new_idx_counter += 1

    # 3. Rebuild basic indices (intermediate step) and identify used `mu` values
    intermediate_basic_indices = []
    used_mus = set()
    # Iterate through original basic indices to preserve their order when collecting
    for old_idx, basic_info in enumerate(original_mtp["alpha_index_basic"]):
        if nodes_to_keep[old_idx]:
            intermediate_basic_indices.append(basic_info)
            used_mus.add(basic_info[0])

    # 4. Create an ORDER-PRESERVING re-indexing map for `mu`
    new_mtp = original_mtp.copy()
    old_mu_to_new_mu_map = {}
    if "radial_basis_size" in original_mtp and original_mtp["radial_basis_size"] > 0:
        # Sort the used mu indices to ensure their relative order is maintained
        sorted_used_mus = sorted(list(used_mus))
        for new_mu, old_mu in enumerate(sorted_used_mus):
            old_mu_to_new_mu_map[old_mu] = new_mu

        new_mtp["radial_funcs_count"] = len(used_mus)

    # 5. Build final index lists using the order-preserving re-indexing maps
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
    original_coeffs = np.array(original_mtp.get("alpha_coeffs", []))
    for i, keep in enumerate(mask):
        if keep:
            old_node_idx = original_mtp["alpha_moment_mapping"][i]
            new_scalar_indices.append(old_to_new_node_map[old_node_idx])

    # 6. Assemble the final MTP dictionary
    new_mtp["alpha_moments_count"] = len(old_to_new_node_map)
    new_mtp["alpha_scalar_moments"] = len(new_scalar_indices)
    new_mtp["alpha_index_basic_count"] = len(new_basic_indices)
    new_mtp["alpha_index_times_count"] = len(new_times_indices)

    new_mtp["alpha_index_basic"] = new_basic_indices
    new_mtp["alpha_index_times"] = new_times_indices
    new_mtp["alpha_moment_mapping"] = new_scalar_indices

    return new_mtp


if __name__ == "__main__":
    mtp18 = parse_mtp_file("18.almtp")
    mask = np.ones(mtp18["alpha_scalar_moments"]).astype(bool)
    mask[3:160] = False  # 12 is a very contracted scalar
    new_mtp = assemble_new_tree(mtp18, mask)

    write_mtp_file(new_mtp, "tmp.almtp")
