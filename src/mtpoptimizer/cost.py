"""
Module for computing computational cost heuristics of an MTP.

This module provides functionality to estimate the computational cost of evaluating a pruned MTP structure. Numba is used for high-performance computation.
"""

import numpy as np
import numba


@numba.njit(cache=True)
def _calculate_jitted(
    mask: np.ndarray,
    num_moments: int,
    n_ranks: int,
    n_mus: int,
    scalar_indices: np.ndarray,
    basic_indices: np.ndarray,
    parents_data: np.ndarray,
    parents_idx: np.ndarray,
    neigh_count: int,
    radial_basis_size: int,
    base_cost: float,
) -> float:
    """
    Jit-compiled function to compute the computational cost heuristic.

    Parameters
    ----------
    mask : np.ndarray
        Boolean mask indicating which scalar moments to keep
    num_moments : int
        Total number of moments in the tree
    n_ranks : int
        Number of possible angular ranks
    n_mus : int
        Number of possible radial functions
    scalar_indices : np.ndarray
        Mapping from scalar outputs to tree nodes
    basic_indices : np.ndarray
        Definition of basic moments [mu, l, n, k]
    parents_data : np.ndarray
        Flattened array of parent pairs for each node
    parents_idx : np.ndarray
        Index array for accessing parents_data
    neigh_count : int
        Number of neighbors in evaluation
    radial_basis_size : int
        Size of radial basis
    base_cost : float
        Cost of full MTP for normalization

    Returns
    -------
    float
        Estimated computational cost relative to base_cost
    """

    # Flags for mus and ranks included in preserved basics
    mus_flags = np.zeros(n_mus, dtype=np.bool_)
    rank_flags = np.zeros(n_ranks, dtype=np.bool_)

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

        start = parents_idx[child]
        end = parents_idx[child + 1]
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

    # Recompute nbasic, ntimes, and update root counters by accounting preserved basics
    ntimes_remaining = 0
    nbasic_remaining = 0

    for i in range(num_moments):
        start = parents_idx[i]
        end = parents_idx[i + 1]
        if to_preserve[i]:
            ntimes_remaining += end - start
            if start == end:  # basic
                nbasic_remaining += 1
                ele = basic_indices[i]
                mu = ele[0]
                rank = max(ele[1], ele[2], ele[3])
                if mu < n_mus:
                    mus_flags[mu] = True
                if rank < n_ranks:
                    rank_flags[rank] = True

    # Apply Cost Heuristic
    max_rank_val = 0
    for r in range(len(rank_flags) - 1, -1, -1):
        if rank_flags[r]:
            max_rank_val = r + 1
            break

    n_mu_val = np.count_nonzero(mus_flags)

    precompute = 4 * max_rank_val
    radial_basis = 8 * radial_basis_size + 14
    radial_vals = 4 * n_mu_val * radial_basis_size
    basics = 39 * nbasic_remaining
    times = 9 * ntimes_remaining

    return (
        neigh_count * (24 + precompute + radial_basis + radial_vals + basics) + times
    ) / base_cost


class MTPCostCalculator:
    """
    Calculator for MTP computational cost heuristics.

    This class prepares and maintains the data structures needed for efficient cost calculation of pruned MTP structures. It pre-processes and uses Numba to increase efficiency.
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

    def _prepare_graph(self) -> None:
        mus_set = set()
        rank_set = set()
        for ele in self.basic_indices:
            mus_set.add(ele[0])
            rank_set.add(max(ele[1], ele[2], ele[3]))

        self.n_mus = len(mus_set)
        self.n_ranks = len(rank_set)

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

    def calculate(self, mask: np.ndarray) -> float:
        """
        Calculate computational cost for a pruned MTP structure.

        This method is a high-level wrapper around the Numba-compiled _calculate_jitted function. It computes a normalized cost heuristic that estimates how computationally expensive the pruned MTP would be compared to the full structure.

        Parameters
        ----------
        mask : np.ndarray
            Boolean mask indicating which scalar moments to keep.
            Length must match the number of scalar outputs without species coefficients.

        Returns
        -------
        float
            Estimated computational cost normalized relative to the full MTP.
        """
        return _calculate_jitted(
            mask,
            self.num_moments,
            self.n_ranks,
            self.n_mus,
            self.scalar_indices,
            self.basic_indices,
            self.parents_data,
            self.parents_idx,
            self.neigh_count,
            self.radial_basis_size,
            self.base_cost,
        )
