import numpy as np
import os
import atexit

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling

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

    def mpi_worker_routine(problem):
        """
        The main loop for a worker process.
        """
        while True:
            continue_eval = COMM.bcast(None, root=0)

            if not continue_eval:
                break

            x_chunk = COMM.scatter(None, root=0)
            results_chunk = problem.evaluate_chunk(x_chunk)
            COMM.gather(results_chunk, root=0)

    def shutdown_workers():
        """
        Master's signal to send a shutdown.
        """
        if RANK == 0:
            print("Master (rank 0) is shutting down workers...")
            COMM.bcast(False, root=0)


class MTPPruningProblem(Problem):
    """
    The core problem definition.
    """

    def __init__(
        self,
        mtp_file,
        bases_file,
        energies_file,
        counts_file,
        neigh_count,
    ):
        # All processes initialize the problem to have access to calculators
        mtp_data = parse_mtp_file(mtp_file)
        radial_basis_size = mtp_data["radial_basis_size"]
        self.cost_calculator = MTPCostCalculator(
            mtp_data, neigh_count, radial_basis_size
        )

        bases = np.genfromtxt(bases_file, delimiter=" ")
        energies = np.genfromtxt(energies_file, delimiter=",")
        counts = np.genfromtxt(counts_file, delimiter=",")
        self.sse_calculator = SSECalculator(bases, energies, counts)

        n_var = mtp_data["alpha_scalar_moments"]

        if RANK == 0:
            mask = np.ones(n_var, dtype=bool)
            print(
                f"Base SSE: {self.sse_calculator.calculate(np.append(mask,True)):.6f}"
            )
            print(f"Base cost: {self.cost_calculator.calculate(mask)}")

        super().__init__(n_var=n_var, n_obj=2, xl=0, xu=1, type_var=bool)

    def evaluate_chunk(self, X_chunk):
        """
        Evaluates a chunk of individuals.
        """
        results = []
        for x_i in X_chunk:
            full_mask = np.append(x_i, True)
            cost = self.cost_calculator.calculate(x_i)
            sse = self.sse_calculator.calculate(full_mask)
            results.append([cost, sse])
        return np.ascontiguousarray(results, dtype=np.float64)

    def _evaluate(self, X, out, *args, **kwargs):
        """
        This method is called by pymoo's `minimize` function.
        In serial mode, it evaluates all individuals directly.
        In MPI mode, serves as a wrapper.
        """
        if IS_MPI:  # Only run by master.
            COMM.bcast(True, root=0)  # Send signal to start workers

            # Ensure contigious for MPI call
            sendbuf = np.ascontiguousarray(X, dtype=bool)
            chunks = np.array_split(sendbuf, SIZE)
            x_chunk = COMM.scatter(chunks, root=0)

            results_chunk = self.evaluate_chunk(x_chunk)
            gathered_results = COMM.gather(results_chunk, root=0)

            out["F"] = np.vstack(gathered_results) if gathered_results else np.array([])

        else:  # Serial execution
            out["F"] = self.evaluate_chunk(X)


def run_optimization(
    mtp_file,
    bases_file,
    energies_file,
    counts_file,
    neigh_count,
    output_dir="outputs",
    end_condition=("n_gen", 1000),
    pop_size=96,
    seed=None,
    show_plot=True,
    verbose=True,
):
    """
    Runs MTP optimization.
    """

    problem = MTPPruningProblem(
        mtp_file, bases_file, energies_file, counts_file, neigh_count
    )

    if IS_MPI and RANK > 0:  # MPI Workers
        mpi_worker_routine(problem)
        return

    # ===== MASTER / SERIAL BRANCH  past this point =====
    if RANK == 0:
        print("--- MTP Optimizer ---")
        if IS_MPI:
            print(f"Mode: MPI Parallel ({SIZE} processes)")
            atexit.register(
                shutdown_workers
            )  # Register the shutdown hook as a safety net
        else:
            print("Mode: Serial")

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=BinaryRandomSampling(),
        crossover=UniformCrossover(),
        mutation=BitflipMutation(),
    )

    res = minimize(problem, algorithm, end_condition, seed=seed, verbose=verbose)

    if IS_MPI:
        shutdown_workers()

    # --- Post-processing and saving ---
    if RANK == 0:
        print(f"Optimization finished in {res.exec_time:.2f} seconds")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        sorted_indices = np.argsort(res.F[:, 0])
        sorted_F = res.F[sorted_indices]
        sorted_X = res.X[sorted_indices]

        pop_path = os.path.join(output_dir, "pareto_population.csv")
        obj_path = os.path.join(output_dir, "pareto_objectives.csv")

        np.savetxt(pop_path, sorted_X.astype(int), delimiter=",", fmt="%d")
        np.savetxt(obj_path, sorted_F, delimiter=",")
        print(f"Saved results to {output_dir}")

        if show_plot and not IS_MPI:
            plot = Scatter(
                title="Pareto Front", labels=["Cost Heuristic", "Sum of Squared Error"]
            )
            plot.add(res.F, facecolor="none", edgecolor="red", s=40)
            plot.show()

    return res
