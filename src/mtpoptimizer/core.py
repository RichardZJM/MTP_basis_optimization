import numpy as np
import os
import time

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation

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
                COMM.gather(problem.eval_time, root=0)
                break

            x_chunk = COMM.scatter(None, root=0)
            results_chunk = problem.evaluate_chunk(x_chunk)
            COMM.gather(results_chunk, root=0)

    def shutdown_workers(eval_time):
        """
        Master's signal to send a shutdown.
        """
        if RANK == 0:
            print("Master (rank 0) is shutting down workers...")
            COMM.bcast(False, root=0)
            gatheredTimes = COMM.gather(eval_time, root=0)
            return gatheredTimes


class MTPPruningProblem(Problem):
    """
    The core problem definition.
    """

    def __init__(
        self,
        mtp_file,
        bases,
        energies,
        counts,
        neigh_count,
        regularization,
    ):
        # All processes initialize the problem to have access to calculators
        mtp_data = parse_mtp_file(mtp_file)
        radial_basis_size = mtp_data["radial_basis_size"]
        self.cost_calculator = MTPCostCalculator(
            mtp_data, neigh_count, radial_basis_size
        )
        self.eval_time = 0
        self.MPI_time = 0
        self.gather_time = 0  # Includes idle time waiting to gather

        try:
            self.sse_calculator = SSECalculator(bases, energies, counts, regularization)
        except RuntimeError as e:
            if RANK == 0:
                raise e  # Let master raise visibly
            else:
                pass  # Worker silently skips or continues

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
        start_time = time.perf_counter()
        results = []
        for x_i in X_chunk:
            full_mask = np.append(x_i, True)
            cost = self.cost_calculator.calculate(x_i)
            sse = self.sse_calculator.calculate(full_mask)
            results.append([cost, sse])
        self.eval_time += time.perf_counter() - start_time
        return np.ascontiguousarray(results, dtype=np.float64)

    def _evaluate(self, X, out, *args, **kwargs):
        """
        This method is called by pymoo's `minimize` function.
        In serial mode, it evaluates all individuals directly.
        In MPI mode, serves as a wrapper.
        """
        if IS_MPI:  # Only run by master.
            MPI_start_time = MPI.Wtime()
            COMM.bcast(True, root=0)  # Send signal to start workers
            self.MPI_time += MPI.Wtime() - MPI_start_time

            # Ensure contigious for MPI call
            sendbuf = np.ascontiguousarray(X, dtype=bool)
            chunks = np.array_split(sendbuf, SIZE)

            MPI_start_time = MPI.Wtime()
            x_chunk = COMM.scatter(chunks, root=0)
            self.MPI_time += MPI.Wtime() - MPI_start_time

            MPI_start_time = MPI.Wtime()
            start_time = time.perf_counter()
            results_chunk = self.evaluate_chunk(x_chunk)
            root_time_taken = time.perf_counter() - start_time
            gathered_results = COMM.gather(results_chunk, root=0)
            self.gather_time += MPI.Wtime() - MPI_start_time - root_time_taken

            out["F"] = np.vstack(gathered_results) if gathered_results else np.array([])

        else:  # Serial execution
            start_time = time.perf_counter()
            out["F"] = self.evaluate_chunk(X)
            self.eval_time += time.perf_counter() - start_time


def run_optimization(
    mtp_file,
    bases,
    energies,
    counts,
    neigh_count,
    regularization=0,
    output_dir="outputs",
    end_condition=("n_gen", 1000),
    pop_size=96,
    seed=None,
    show_plot=True,
    verbose=True,
    init_pop=None,
):
    """
    Runs MTP optimization.
    """

    problem = MTPPruningProblem(
        mtp_file, bases, energies, counts, neigh_count, regularization
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
        print(f"Using inital population of size {init_pop.shape[0]}!")
        sampling = init_pop.astype(bool)

    else:
        print("Using seeded inital population.")
        near_extrema_prob = 0.01
        near_extrema_count = int(pop_size / 50)
        sampling = np.random.permutation(
            np.concatenate(
                (
                    np.zeros((1, problem.n_var)),  # One cost minima
                    np.random.choice(
                        [0, 1],
                        size=(near_extrema_count, problem.n_var),
                        p=[1 - near_extrema_prob, near_extrema_prob],
                    ),  # 5 cost near minima
                    np.ones((1, problem.n_var)),  # One cost maxima
                    np.random.choice(
                        [0, 1],
                        size=(near_extrema_count, problem.n_var),
                        p=[near_extrema_prob, 1 - near_extrema_prob],
                    ),  # 5 cost near maxima
                    np.random.randint(
                        0,
                        2,
                        size=(pop_size - 2 * (1 + near_extrema_count), problem.n_var),
                    ),  # Rest is random
                ),
                axis=0,
            )
        ).astype(bool)

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=sampling,
        crossover=UniformCrossover(),
        mutation=BitflipMutation(),
    )

    res = minimize(problem, algorithm, end_condition, seed=seed, verbose=verbose)

    if IS_MPI:
        evals = shutdown_workers(problem.eval_time)
    communication_time = (
        problem.MPI_time + problem.gather_time - (max(evals) - evals[0])
    )
    print(problem.MPI_time)
    print(f"Evaluation times per process: [{', '.join(f'{x:.2f} s' for x in evals)}]")
    # --- Post-processing and saving ---
    if RANK == 0:
        print(f"Optimization finished in {res.exec_time:.2f} seconds.")
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
            f"Wasted time due to load imbalance: {(max(evals) - min(evals))/res.exec_time*100:.2f}%."
        )

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
