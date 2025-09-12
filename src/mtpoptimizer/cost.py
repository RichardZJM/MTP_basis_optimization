import numpy as np
import numba


@numba.njit(cache=True)
def _calculate_jitted(
    mask,
    nbasic_orig,
    ntimes_orig,
    num_moments,
    root_ranks,
    root_mus,
    scalar_indices,
    basic_indices,
    parents_data,
    parents_idx,
    neigh_count,
    radial_basis_size,
    base_cost,
):

    # Copies of mutable state
    nbasic = nbasic_orig
    ntimes = ntimes_orig
    max_ranks = root_ranks.copy()
    max_mus = root_mus.copy()

    # Mark preserved moments by backpropagating from kept outputs
    to_preserve = np.zeros(num_moments, dtype=np.bool_)
    queue = np.empty(num_moments, dtype=np.int32)
    qh = 0
    qt = 0

    # Seed queue with moments explicitly selected by mask
    for i in range(len(mask)):
        if mask[i]:
            m = scalar_indices[i]
            if not to_preserve[m]:
                to_preserve[m] = True
                queue[qt] = m
                qt += 1

    # Backpropagate: for each preserved child, preserve its parents (p1, p2)
    while qh < qt:
        child = queue[qh]
        qh += 1

        # If child has parents, iterate them and mark their parents
        start = parents_idx[child]
        end = parents_idx[child + 1]
        # If start == end -> basic node, nothing to propagate
        for j in range(start, end):
            p1 = parents_data[j, 0]
            p2 = parents_data[j, 1]

            if not to_preserve[p1]:
                to_preserve[p1] = True
                queue[qt] = p1
                qt += 1

            if not to_preserve[p2]:
                to_preserve[p2] = True
                queue[qt] = p2
                qt += 1

    # Recompute nbasic, ntimes, and update root counters by accounting removed basics
    # ntimes_remaining = number of parent relations whose child is preserved
    ntimes_remaining = 0
    nbasic_remaining = 0

    for i in range(num_moments):
        start = parents_idx[i]
        end = parents_idx[i + 1]
        if to_preserve[i]:
            # Count retained time relations for this preserved child
            ntimes_remaining += end - start
            # If no parents -> basic, count it
            if start == end:
                nbasic_remaining += 1
        else:
            # If this basic was removed, decrement root counters accordingly
            if start == end:
                ele = basic_indices[i]
                # Decrement the stored root counts (mirror original behaviour)
                max_ranks[max(ele[1], ele[2], ele[3])] -= 1
                max_mus[ele[0]] -= 1

    # Replace working counters with remaining counts
    nbasic = nbasic_remaining
    ntimes = ntimes_remaining

    # ===== Cost Heuristic =====
    max_rank_val = np.count_nonzero(max_ranks)
    radial_func_count_val = np.count_nonzero(max_mus)

    precompute = 4 * max_rank_val
    radial_vals = 4 * radial_func_count_val * radial_basis_size
    basics = 39 * nbasic
    times = 9 * ntimes

    return (neigh_count * (24 + precompute + radial_vals + basics) + times) / base_cost


class MTPCostCalculator:
    """
    Calculates a computational cost heuristic for a pruned MTP tree.
    """

    def __init__(self, mtp_data: dict, neigh_count: int, radial_basis_size: int):
        self.neigh_count = neigh_count
        self.radial_basis_size = radial_basis_size

        self.basic_indices = np.array(mtp_data["alpha_index_basic"], dtype=np.int32)
        self.times_indices = np.array(mtp_data["alpha_index_times"], dtype=np.int32)
        self.scalar_indices = np.array(mtp_data["alpha_moment_mapping"], dtype=np.int32)
        self.num_moments = mtp_data["alpha_moments_count"]
        self.nbasic_orig = mtp_data["alpha_index_basic_count"]
        self.ntimes_orig = mtp_data["alpha_index_times_count"]

        self._prepare_graph()

        self.base_cost = 1
        self.base_cost = self.calculate(np.ones_like(self.scalar_indices).astype(bool))

    def _prepare_graph(self):
        """Pre-computes graph properties in a Numba-friendly format."""
        # Calculate root mus and ranks
        root_mus = np.zeros(100, dtype=np.int32)
        root_ranks = np.zeros(100, dtype=np.int32)
        for i, ele in enumerate(self.basic_indices):
            root_mus[ele[0]] += 1
            root_ranks[max(ele[1:3])] += 1
        self.root_mus = root_mus[root_mus != 0]
        self.root_ranks = root_ranks[root_ranks != 0]

        # Build parent lists
        py_parents = [[] for _ in range(self.num_moments)]
        for i, ele in enumerate(self.times_indices):
            p1, p2, _, child = ele
            py_parents[child].append((p1, p2))

        # Convert it to a flattened ragged array for compilation support.
        self.parents_idx = np.zeros(self.num_moments + 1, dtype=np.int32)
        flat_parents_list = []
        for i in range(self.num_moments):  # Prefix sum
            self.parents_idx[i] = len(flat_parents_list)
            flat_parents_list.extend(py_parents[i])
        self.parents_idx[self.num_moments] = len(flat_parents_list)
        self.parents_data = np.array(flat_parents_list, dtype=np.int32)

    def calculate(self, mask: np.ndarray):
        """
        Calculates the cost for a given feature mask.
        Wrapper around Numba.
        """
        return _calculate_jitted(
            mask,
            self.nbasic_orig,
            self.ntimes_orig,
            self.num_moments,
            self.root_ranks,
            self.root_mus,
            self.scalar_indices,
            self.basic_indices,
            self.parents_data,
            self.parents_idx,
            self.neigh_count,
            self.radial_basis_size,
            self.base_cost,
        )
