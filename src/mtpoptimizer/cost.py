import numpy as np


class MTPCostCalculator:
    """
    Calculates a computational cost heuristic for a pruned MTP tree.
    """

    def __init__(self, mtp_data: dict, neigh_count: int, radial_basis_size: int):
        """
        Initializes the cost calculator with data from a parsed MTP file.

        Args:
            mtp_data (dict): A dictionary containing the parsed MTP data.
        """
        self.mtp = mtp_data
        self.basic_indices = mtp_data["alpha_index_basic"]
        self.times_indices = mtp_data["alpha_index_times"]
        self.scalar_indices = mtp_data["alpha_moment_mapping"]
        self.num_moments = mtp_data["alpha_moments_count"]

        self.neigh_count = neigh_count
        self.radial_basis_size = radial_basis_size

        # Cache the root values and graph structure
        self._prepare_graph()

    def _prepare_graph(self):
        """Pre-computes graph properties for fast cost calculation."""
        # Calculate root mus and ranks
        root_mus = np.zeros(20)  # Assuming max mu is < 20
        root_ranks = np.zeros(20)  # Assuming max rank is < 20
        for i, ele in enumerate(self.basic_indices):
            root_mus[ele[0]] += 1
            root_ranks[max(ele[1:3])] += 1
        self.root_mus = root_mus[root_mus != 0]
        self.root_ranks = root_ranks[root_ranks != 0]

        # Build parent and children lists
        self.parents = [[] for _ in range(self.num_moments)]
        self.nchildren = np.zeros(self.num_moments, dtype=int)
        for i, ele in enumerate(self.times_indices):
            p1, p2, _, child = ele
            self.parents[child].append((p1, p2))
            self.nchildren[p1] += 1
            self.nchildren[p2] += 1

    def _cost_heuristic(
        self,
        max_rank,
        radial_func_count,
        alpha_basic,
        alpha_times,
    ):
        """The core cost formula."""
        precompute = 4 * max_rank
        radial_vals = 4 * radial_func_count * self.radial_basis_size
        basics = 39 * alpha_basic
        times = 9 * alpha_times
        return self.neigh_count * (24 + precompute + radial_vals + basics) + times

    def calculate(self, mask):
        """
        Calculates the cost for a given feature mask.

        Args:
            mask (np.ndarray): A boolean array indicating which scalar moments to keep.

        Returns:
            float: The calculated cost heuristic.
        """
        nbasic = self.mtp["alpha_index_basic_count"]
        ntimes = self.mtp["alpha_index_times_count"]
        max_ranks = self.root_ranks.copy()
        max_mus = self.root_mus.copy()
        local_nchildren = self.nchildren.copy()

        preserve_nodes = np.zeros(self.num_moments, dtype=bool)
        for i, should_keep in enumerate(mask):
            if should_keep:
                node_to_preserve = self.scalar_indices[i]
                preserve_nodes[node_to_preserve] = True

        processed_for_removal = np.zeros(self.num_moments, dtype=bool)

        nodes_to_process_for_removal = []
        for i, ele in enumerate(mask):
            if not ele:
                base_node = self.scalar_indices[i]
                if local_nchildren[base_node] == 0:
                    nodes_to_process_for_removal.append(base_node)

        head = 0
        while head < len(nodes_to_process_for_removal):
            i = nodes_to_process_for_removal[head]
            head += 1

            if processed_for_removal[i] or preserve_nodes[i]:
                continue

            processed_for_removal[i] = True

            if not self.parents[i]:
                nbasic -= 1
                ele = self.basic_indices[i]
                max_ranks[max(ele[1:3])] -= 1
                max_mus[ele[0]] -= 1
                continue

            for parent1, parent2 in self.parents[i]:
                ntimes -= 1
                local_nchildren[parent1] -= 1
                if local_nchildren[parent1] == 0:
                    nodes_to_process_for_removal.append(parent1)

                local_nchildren[parent2] -= 1
                if local_nchildren[parent2] == 0:
                    nodes_to_process_for_removal.append(parent2)

        max_rank_val = np.count_nonzero(max_ranks)
        radial_func_count_val = np.count_nonzero(max_mus)

        return self._cost_heuristic(
            max_rank_val,
            radial_func_count_val,
            nbasic,
            ntimes,
        )
