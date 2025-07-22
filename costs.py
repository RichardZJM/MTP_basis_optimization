import re
import numpy as np

from mtpio import parse_mtp_file

mtp = parse_mtp_file("18.almtp")

print(mtp.keys())

basic_indices = mtp["alpha_index_basic"]
times_indices = mtp["alpha_index_times"]
scalar_indices = mtp["alpha_moment_mapping"]


# ===== Cache the root values =====
root_mus = np.zeros(20)
root_ranks = np.zeros(20)

for i, ele in enumerate(basic_indices):
    root_mus[ele[0]] += 1
    root_ranks[max(ele[1:3])] += 1

root_mus = root_mus[root_mus != 0]
root_ranks = root_ranks[root_ranks != 0]

# ===== Prepare the graph =====
parents = [[] for _ in range(mtp["alpha_moments_count"])]
nchildren = [0] * mtp["alpha_moments_count"]

for i, ele in enumerate(times_indices):
    parents[ele[3]].append((ele[0], ele[1]))
    nchildren[ele[0]] += 1
    nchildren[ele[1]] += 1

# ===== Constants =====
neigh_count = 25
radial_basis_size = 8


def cost_heuristic(max_rank, radial_func_count, alpha_basic, alpha_times):
    precompute = 4 * max_rank
    radial_vals = 4 * radial_func_count * radial_basis_size
    basics = 33 * alpha_basic
    times = 9 * alpha_times
    return neigh_count * (precompute + radial_vals + basics) + times


def prune_tree(mask):
    nbasic = mtp["alpha_index_basic_count"]
    ntimes = mtp["alpha_index_times_count"]
    max_ranks = root_ranks.copy()
    max_mus = root_mus.copy()
    local_nchildren = nchildren.copy()

    num_nodes = mtp["alpha_moments_count"]

    preserve_nodes = np.zeros(num_nodes, dtype=bool)
    for i, should_keep in enumerate(mask):
        if should_keep:
            node_to_preserve = scalar_indices[i]
            preserve_nodes[node_to_preserve] = True

    processed_for_removal = np.zeros(num_nodes, dtype=bool)

    def process_node(i):
        nonlocal nbasic, ntimes

        if processed_for_removal[i]:
            return

        processed_for_removal[i] = True

        if not parents[i]:
            nbasic -= 1
            ele = basic_indices[i]
            max_ranks[max(ele[1:3])] -= 1
            max_mus[ele[0]] -= 1
            return

        for parent1, parent2 in parents[i]:
            ntimes -= 1

            local_nchildren[parent1] -= 1
            local_nchildren[parent2] -= 1

            if local_nchildren[parent1] == 0 and not preserve_nodes[parent1]:
                process_node(parent1)

            if local_nchildren[parent2] == 0 and not preserve_nodes[parent2]:
                process_node(parent2)

    for i, ele in enumerate(mask):
        if ele == False:
            base = scalar_indices[i]
            process_node(base)

    max_rank = len(root_ranks)
    for ele in max_ranks:
        if ele == 0:
            max_rank -= 1

    radial_func_count = len(root_mus)
    for ele in max_mus:
        if ele == 0:
            radial_func_count -= 1

    # print(nbasic, ntimes, max_ranks, max_mus)
    # print(cost_heuristic(max_rank, radial_func_count, nbasic, ntimes))
    return cost_heuristic(max_rank, radial_func_count, nbasic, ntimes)


if __name__ == "__main__":
    mask = np.ones(len(scalar_indices)).astype(bool)
    print(prune_tree(mask=mask))

    mask[12] = False  # 12 is a very contracted scalar
    # mask[1:163] = False
    print(mtp["alpha_index_basic_count"], mtp["alpha_index_times_count"])
    print(prune_tree(mask=mask))
