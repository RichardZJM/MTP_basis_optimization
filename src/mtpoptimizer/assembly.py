import numpy as np


def assemble_new_tree(original_mtp, mask):
    """
    Assembles a new, pruned MTP tree. Uses minimum node indexing; preserves ordering.

    Args:
        mtp_file (dict): A dictionary containing inital MTP data.
        mask (np.ndarray): mask of pruned scalar vectors.

    Returns:
        dict: A dictionary containing the pruned MTP data.
    """
    # Create parent tree
    parents_lookup = [[] for _ in range(original_mtp["alpha_moments_count"])]
    for p1, p2, k, child in original_mtp["alpha_index_times"]:
        parents_lookup[child].append((p1, p2))

    # Add initial scalar moments to keep
    nodes_to_keep = np.zeros(original_mtp["alpha_moments_count"], dtype=bool)
    queue = []
    for i, keep in enumerate(mask):
        if keep:
            node_idx = original_mtp["alpha_moment_mapping"][i]
            if not nodes_to_keep[node_idx]:
                nodes_to_keep[node_idx] = True
                queue.append(node_idx)

    # Traverse the tree to find nodes to keep(BFS)
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

    # Create a map to reindex the nodes based on those removed
    old_to_new_node_map = {}
    new_idx_counter = 0

    # Iterate through all original nodes, preserve minimum node numbering
    for old_idx in range(original_mtp["alpha_moments_count"]):
        if nodes_to_keep[old_idx]:
            old_to_new_node_map[old_idx] = new_idx_counter
            new_idx_counter += 1

    # Iterate original nodes and identify used remaining radial funcs
    intermediate_basic_indices = []
    used_mus = set()
    for old_idx, basic_info in enumerate(original_mtp["alpha_index_basic"]):
        if nodes_to_keep[old_idx]:
            intermediate_basic_indices.append(basic_info)
            used_mus.add(basic_info[0])
    new_mtp = original_mtp.copy()
    new_mtp["radial_funcs_count"] = len(used_mus)

    # Now reindex the radial funcs, preseving minimum mu numbering
    old_mu_to_new_mu_map = {}
    sorted_used_mus = sorted(list(used_mus))
    for new_mu, old_mu in enumerate(sorted_used_mus):
        old_mu_to_new_mu_map[old_mu] = new_mu

    # Finish  rebuilding the basic nodes
    new_basic_indices = []
    for old_mu, l, n, k in intermediate_basic_indices:
        new_mu = old_mu_to_new_mu_map.get(old_mu, old_mu)
        new_basic_indices.append([new_mu, l, n, k])

    # Finish rebuilding the graph edges.
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

    # Reindex the old scalars
    new_scalar_indices = []
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
