"""
Core module for Moment Tensor Potentials (MTP) Optimization.

This module implements the main optimization framework for pruning MTP structures while balancing computational cost and accuracy. It supports both serial and MPI-parallel execution modes, using either NSGA-II or MOEA/D algorithms from the pymoo framework.
"""

from typing import Optional, Tuple, List
import numpy as np
import os
import time

from pymoo.core.problem import Problem
from pymoo.core.result import Result
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.moead import ParallelMOEAD
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.core.callback import Callback

from .cost import MTPCostCalculator
from .sse import SSECalculator
from .mtpio import parse_mtp_file

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
    IS_MPI = SIZE > 1
except ImportError:
    COMM = None
    RANK = 0
    SIZE = 1
    IS_MPI = False


if IS_MPI:

    def mpi_worker_routine(problem) -> None:
        """
        The main loop for a worker process in MPI parallel mode.

        This function implements workers which continuously receive population chunks from the master, evaluate them, and send results back until receiving a shutdown signal.

        Parameters
        ----------
        problem : MTPPruningProblem
            The optimization problem instance containing evaluation methods
        """
        while True:
            COMM.Scatter(None, problem.work_buffer, root=0)

            tasks = np.sum(problem.work_buffer[:, 0])

            if tasks == 0:
                COMM.gather(problem.eval_time, root=0)
                break

            results_chunk = problem.evaluate_chunk(problem.work_buffer)
            COMM.Gather(results_chunk, None, root=0)

    def shutdown_workers(problem) -> List[float]:
        """
        Send shutdown signal to all worker processes and gather timing statistics.

        Parameters
        ----------
        problem : MTPPruningProblem
            The optimization problem instance containing MPI buffers

        Returns
        -------
        List[float]
            Evaluation times from all processes, index 0 is master's time
        """
        if RANK == 0:
            print("Master (rank 0) is shutting down workers...")
            end_signal = np.zeros_like(problem.send_buffer).astype(bool)
            COMM.Scatter(end_signal, problem.work_buffer, root=0)
            gatheredTimes = COMM.gather(problem.eval_time, root=0)
            return gatheredTimes


class MTPPruningProblem(Problem):
    """
    Multi-objective optimization problem for pruning MTP structures.

    This class defines the optimization problem for the pymoo framework, handling both serial and MPI-parallel evaluation of solutions.
    """

    def __init__(
        self,
        mtp_file,
        xtwx: np.ndarray,
        xtwy: np.ndarray,
        ytwy: float,
        neigh_count: int,
        regularization: float,
        pop_size: int,
    ):
        """
        Initialize the pymoo optimization problem.

        Parameters
        ----------
        mtp_file:
            File path to the MTP to optimize
        xtwx : np.ndarray
            Pre-computed XᵀWX matrix, shape (n_features, n_features)
        xtwy : np.ndarray
            Pre-computed XᵀWy vector, shape (n_features,)
        ytwy : float
            Pre-computed yᵀWy scalar
        neigh_count : int
            Number of neighbors to optimize for
        regularization : float
            Tikhonov regularization parameter (λ), by default 0.0.
            Adds λI to XᵀWX for numerical stability
        pop_size : int
            The population size to use during optimization
        """
        # All processes initialize the problem to have access to calculators
        mtp_data = parse_mtp_file(mtp_file)
        self.n_species = mtp_data["species_count"]
        radial_basis_size = mtp_data["radial_basis_size"]
        n_var = mtp_data["alpha_scalar_moments"]
        self.cost_calculator = MTPCostCalculator(
            mtp_data, neigh_count, radial_basis_size
        )
        self.sse_calculator = SSECalculator(xtwx, xtwy, ytwy, regularization, rank=RANK)

        self.eval_time = 0
        self.MPI_time = 0
        self.gather_time = 0  # Includes idle time waiting to gather

        work_size = (pop_size + SIZE - 1) // SIZE
        self.work_buffer = np.ascontiguousarray(
            np.zeros((work_size, n_var + 1)).astype(bool)
        )  # +1 for whether to process this row

        if RANK == 0:
            if IS_MPI:
                # Preconstruct the send buffer with
                pad_width = work_size * SIZE
                self.send_buffer = np.ascontiguousarray(
                    np.zeros((pad_width, n_var + 1)).astype(bool)
                )
                self.send_buffer[:pop_size, 0] = True  # Set the flags to process

                self.res_buffer = np.ascontiguousarray(
                    np.zeros((pad_width, 2)).astype(np.float64)
                )

        super().__init__(n_var=n_var, n_obj=2, xl=0, xu=1, type_var=bool)

    def evaluate_chunk(self, x_chunk: np.ndarray) -> np.ndarray:
        """
        Evaluate fitness for a chunk of population members.

        Parameters
        ----------
        x_chunk : np.ndarray
            Array of shape (chunk_size, n_var + 1) containing boolean masks. First column is a flag indicating whether to process the row.

        Returns
        -------
        np.ndarray
            Array of shape (chunk_size, 2) containing [cost, sse] pairs for each evaluated solution.
        """
        start_time = time.perf_counter()
        results = []
        for x_i in x_chunk:
            # Check the flag
            if x_i[0] == False:
                results.append([np.nan, np.nan])
                continue

            # Append the species coeffs for SSE
            full_mask = np.append(np.full((self.n_species), True, dtype=bool), x_i[1:])
            cost = self.cost_calculator.calculate(x_i[1:])
            sse = self.sse_calculator.calculate(full_mask)
            results.append([cost, sse])
        self.eval_time += time.perf_counter() - start_time
        return np.ascontiguousarray(results, dtype=np.float64)

    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs) -> None:
        """
        Population evaluation method required by pymoo framework.

        This method is called by pymoo's `minimize` function to evaluate fitness for each generation of solutions.

        Parameters
        ----------
        X : np.ndarray
            Population matrix of shape (n_individuals, n_var) containing
            boolean masks for each solution
        out : dict
            Dictionary that will store the fitness values under key 'F'
        *args, **kwargs
            Additional arguments passed by pymoo (unused)
        """
        if IS_MPI:  # Only run by master.

            self.send_buffer[0 : len(X), 1:] = X

            MPI_start_time = MPI.Wtime()
            COMM.Scatter(self.send_buffer, self.work_buffer, root=0)
            self.MPI_time += MPI.Wtime() - MPI_start_time

            # Estimate the wall time taken with the difference of root
            MPI_start_time = MPI.Wtime()
            start_time = time.perf_counter()
            results_chunk = self.evaluate_chunk(self.work_buffer)
            root_time_taken = time.perf_counter() - start_time
            COMM.Gather(results_chunk, self.res_buffer, root=0)
            self.gather_time += MPI.Wtime() - MPI_start_time - root_time_taken

            out["F"] = self.res_buffer[: len(X)]

        else:  # Serial execution
            # Set up the flags and chunks
            work_chunk = np.zeros((len(X), self.n_var + 1), dtype=bool)
            work_chunk[:, 0] = True
            work_chunk[:, 1:] = X

            results = self.evaluate_chunk(work_chunk)
            out["F"] = results


class SaveInterval(Callback):
    """
    A custom callback to save the pareto front every so often. Useful for restarting long runs.
    """

    def __init__(self, save_interval: int, output_dir) -> None:
        """
         Initialize the pymoo optimization problem.

        Parameters
        ----------
        save_interval : int
            Interval to save to output folder.
        output_dir : str
            Directory to save to.
        """
        super().__init__()
        self.save_interval = save_interval
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def notify(self, algorithm):
        if algorithm.n_iter % self.save_interval == 0:
            pop_path = os.path.join(
                self.output_dir, "pareto_population_" + str(algorithm.n_iter) + ".csv"
            )
            obj_path = os.path.join(
                self.output_dir, "pareto_objectives_" + str(algorithm.n_iter) + ".csv"
            )
            F = algorithm.pop.get("F")
            X = algorithm.pop.get("X")
            sorted_indices = np.argsort(F[:, 0])
            sorted_F = F[sorted_indices]
            sorted_X = X[sorted_indices]
            np.savetxt(pop_path, sorted_X.astype(int), delimiter=",", fmt="%d")
            np.savetxt(obj_path, sorted_F, delimiter=",")
            print(f"Saved results to {self.output_dir}")


def run_optimization(
    mtp_file: str,
    xtwx: np.ndarray,
    xtwy: np.ndarray,
    ytwy: float,
    neigh_count: int,
    regularization: float = 0,
    output_dir: str = "outputs",
    end_condition: Tuple[str, int] = ("n_gen", 1000),
    pop_size: int = 512,
    seed: Optional[int] = None,
    show_plot: bool = True,
    verbose: bool = True,
    init_pop: Optional[np.ndarray] = None,
    algorithm: str = "nsga",
    mutation_rate: Optional[float] = None,
    save_interval: int = 1000,
) -> Optional[Result]:
    """
    Run the multi-objective optimization to prune an MTP structure.

    This is the entry point for the optimization process. It handles setup, execution, and result processing for both serial and parallel modes.

    Parameters
    ----------
    mtp_file : str
        Path to the MTP file to optimize
    xtwx : np.ndarray
        Precomputed XᵀWX matrix for SSE calculation
    xtwy : np.ndarray
        Precomputed XᵀWy vector for SSE calculation
    ytwy : float
        Precomputed yᵀWy scalar for SSE calculation
    neigh_count : int
        Number of neighbors for cost heuristic
    regularization : float, optional
        L2 regularization parameter, by default 0
    output_dir : str, optional
        Directory to save results, by default "outputs"
    end_condition : tuple[str, int], optional
        Termination criterion, by default ("n_gen", 1000)
    pop_size : int, optional
        Population size, by default 512
    seed : int, optional
        Random seed for reproducibility, by default None
    show_plot : bool, optional
        Whether to show Pareto front plot, by default True
    verbose : bool, optional
        Whether to print progress, by default True
    init_pop : np.ndarray, optional
        Initial population matrix, by default None
    algorithm : str, optional
        "nsga" (default) or "moead"
    mutation_rate : float, optional
        Override default mutation rate, by default None
    save_interval : int, optional
        Save to output folder every save_interval iterations, 1000 by default
    Returns
    -------
    minimize
        pymoo Result object (only from master/serial process)
        None from worker processes in MPI mode

    Files saved in output_dir:
    - pareto_population.csv: Binary masks for Pareto-optimal solutions
    - pareto_objectives.csv: Corresponding objective values
    """
    np.random.seed(seed)

    problem = MTPPruningProblem(
        mtp_file, xtwx, xtwy, ytwy, neigh_count, regularization, pop_size
    )

    if IS_MPI and RANK > 0:  # MPI Workers
        mpi_worker_routine(problem)
        return

    # ===== MASTER / SERIAL BRANCH  past this point =====
    if RANK == 0:
        print("--- MTP Optimizer ---")
        if IS_MPI:
            print(f"Mode: MPI Parallel ({SIZE} processes)")
        else:
            print("Mode: Serial")

    if not init_pop is None:
        print(f"Using user-specified initial population of size {init_pop.shape[0]}!")
        sampling = init_pop.astype(bool)
    else:
        print("Using seeded initial population.")

        assert pop_size >= 2, "Population size must be at least 2."

        all_zeros = np.zeros((1, problem.n_var), dtype=bool)
        all_ones = np.ones((1, problem.n_var), dtype=bool)

        remaining_pop_size = pop_size - 2
        if remaining_pop_size > 0:
            probs = np.linspace(0, 1, pop_size)[1:-1]
            pop = np.random.rand(remaining_pop_size, problem.n_var)
            pop = pop < probs[:, np.newaxis]
            pop = np.concatenate((all_zeros, all_ones, pop), axis=0)
        else:
            pop = np.concatenate((all_zeros, all_ones), axis=0)

        sampling = np.random.permutation(pop)

    callback = SaveInterval(save_interval, output_dir)

    if algorithm == "moead":
        ref_dirs = get_reference_directions("energy", 2, pop_size)
        solver = ParallelMOEAD(
            ref_dirs=ref_dirs,
            sampling=sampling,
            crossover=UniformCrossover(),
            callback=callback,
            mutation=BitflipMutation(prob_var=mutation_rate),
        )
    else:
        solver = NSGA2(
            pop_size=pop_size,
            sampling=sampling,
            crossover=UniformCrossover(),
            callback=callback,
            mutation=BitflipMutation(prob_var=mutation_rate),
            eliminate_duplicates=True,
        )

    res = minimize(problem, solver, end_condition, seed=seed, verbose=verbose)

    # --- Post-processing and saving ---
    print(f"Optimization finished in {res.exec_time:.2f} seconds.")

    if IS_MPI:
        evals = shutdown_workers(problem)
        communication_time = (
            problem.MPI_time + problem.gather_time - (max(evals) - evals[0])
        )
        print(
            f"Evaluation times per process: [{', '.join(f'{x:.2f} s' for x in evals)}]"
        )
        print(
            f"Average fitness evaluation time: {(np.mean(evals))/res.exec_time*100:.2f}%."
        )
        print(
            f"Communication time (Estimated): {(communication_time )/res.exec_time*100:.2f}%."
        )
        print(
            f"Serial time (Estimated): {(res.exec_time-max(evals)-communication_time)/res.exec_time*100:.2f}%."
        )
        print(
            f"Wasted time due to load imbalance (Estimated): {(max(evals) - min(evals))/res.exec_time*100:.2f}%."
        )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sorted_indices = np.argsort(res.F[:, 0])
    sorted_F = res.F[sorted_indices]
    sorted_X = res.X[sorted_indices]

    pop_path = os.path.join(output_dir, "pareto_population_final.csv")
    obj_path = os.path.join(output_dir, "pareto_objectives_final.csv")

    np.savetxt(pop_path, sorted_X.astype(int), delimiter=",", fmt="%d")
    np.savetxt(obj_path, sorted_F, delimiter=",")
    print(f"Saved results to {output_dir}")

    if show_plot:
        plot = Scatter(
            title="Pareto Front", labels=["Cost Heuristic", "Sum of Squared Error"]
        )
        plot.add(res.F, facecolor="none", edgecolor="red", s=40)
        plot.show()

    return res
