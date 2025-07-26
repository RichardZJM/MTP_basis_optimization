import numpy as np
import numba


@numba.njit
def _calculate_jitted(
    mask,
    nbasic_orig,
    ntimes_orig,
    num_moments,
    root_ranks,
    root_mus,
    nchildren,
    scalar_indices,
    basic_indices,
    parents_data,
    parents_idx,
    neigh_count,
    radial_basis_size,
):

    nbasic = nbasic_orig
    ntimes = ntimes_orig
    max_ranks = root_ranks.copy()
    max_mus = root_mus.copy()
    local_nchildren = nchildren.copy()
    
    
    # Use a fixed-size NumPy array as a queue
    to_remove = np.empty(num_moments, dtype=np.int32)
    head = 0
    tail = 0

    to_preserve = np.zeros(num_moments, dtype=np.bool_)
    for i in range(len(mask)):
        if mask[i]:
            to_preserve[scalar_indices[i]] = True
        else:
            if local_nchildren[scalar_indices[i]] == 0:
                to_remove[tail] = scalar_indices[i]
                tail += 1

    while head < tail:
        i = to_remove[head]
        head += 1

        if to_preserve[i]:
            continue

        # Check for parents
        is_basic = (parents_idx[i] == parents_idx[i + 1])
        if is_basic:
            nbasic -= 1
            ele = basic_indices[i]
            max_ranks[max(ele[1], ele[2],ele[3])] -= 1
            max_mus[ele[0]] -= 1
            continue

        # Loop over parents
        start = parents_idx[i]
        end = parents_idx[i+1]
        for j in range(start, end):
            parent1, parent2 = parents_data[j]
            
            ntimes -= 1
            local_nchildren[parent1] -= 1
            if local_nchildren[parent1] == 0:
                to_remove[tail] = parent1
                tail += 1

            local_nchildren[parent2] -= 1
            if local_nchildren[parent2] == 0:
                to_remove[tail] = parent2
                tail += 1
    
    # ===== Cost Heurstic =====
    max_rank_val = np.count_nonzero(max_ranks)
    radial_func_count_val = np.count_nonzero(max_mus)
    
    precompute = 4 * max_rank_val
    radial_vals = 4 * radial_func_count_val * radial_basis_size
    basics = 39 * nbasic
    times = 9 * ntimes
    
    return neigh_count * (24 + precompute + radial_vals + basics) + times


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

        self.nchildren = np.zeros(self.num_moments, dtype=np.int32)
        
        # Build parent lists
        py_parents = [[] for _ in range(self.num_moments)]
        for i, ele in enumerate(self.times_indices):
            p1, p2, _, child = ele
            py_parents[child].append((p1, p2))
            self.nchildren[p1] += 1
            self.nchildren[p2] += 1

        # Convert it to a flattened ragged array for compilation support.
        self.parents_idx = np.zeros(self.num_moments + 1, dtype=np.int32)
        flat_parents_list = []
        for i in range(self.num_moments): # Prefix sum
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
            self.nchildren,
            self.scalar_indices,
            self.basic_indices,
            self.parents_data,
            self.parents_idx,
            self.neigh_count,
            self.radial_basis_size,
        )
