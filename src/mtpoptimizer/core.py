import numpy as np
import os

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
    import atexit

    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
    IS_MPI = SIZE > 1
except ImportError:
    COMM = None
    RANK = 0
    SIZE = 1
    IS_MPI = False

# ==============================================================================
#  BEGIN: MPI-SPECIFIC HELPER FUNCTIONS
# ==============================================================================
if IS_MPI:
    TAG_TASK = 1
    TAG_RESULT = 2
    TAG_SHUTDOWN = 3

    def mpi_worker_routine(problem_args):
        """The main loop for a worker process. It waits for and executes tasks."""
        problem = MTPPruningProblem(**problem_args)
        # print(f"Worker (rank {RANK}) initialized and ready.")

        status = MPI.Status()
        while True:
            task_data = COMM.recv(source=0, tag=MPI.ANY_TAG, status=status)
            if status.Get_tag() == TAG_SHUTDOWN:
                # print(f"Worker (rank {RANK}) received shutdown signal. Exiting.")
                break

            task_index, x_i = task_data
            out = {}
            problem._evaluate(x_i, out)
            COMM.send((task_index, out["F"]), dest=0, tag=TAG_RESULT)

    def shutdown_workers():
        """Master's hook to tell workers to exit cleanly."""
        n_workers = SIZE - 1
        print("Master (rank 0) is shutting down workers...")
        for i in range(n_workers):
            COMM.send(None, dest=i + 1, tag=TAG_SHUTDOWN)


# ==============================================================================
#  END: MPI-SPECIFIC HELPER FUNCTIONS
# ==============================================================================


class MTPPruningProblem(Problem):
    """
    The core problem definition.
    This class is now self-aware of the execution environment (MPI master, serial, or MPI worker).
    """

    def __init__(
        self,
        mtp_file,
        bases_file,
        energies_file,
        counts_file,
        neigh_count,
    ):

        # Currently all process acess IO. Not great but simple.
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
        super().__init__(n_var=n_var, n_obj=2, elementwise=False)

    def _evaluate(self, X, out, *args, **kwargs):
        """
        This method's behavior depends on the execution context.
        """
        # --- BEHAVIOR FOR MPI MASTER ---
        if IS_MPI and RANK == 0:

            n_eval = X.shape[0]
            if n_eval == 0:
                return

            results = [None] * n_eval

            # Use a list as a queue of task indices to be sent
            tasks_to_do = list(range(n_eval))
            tasks_sent_count = 0
            tasks_received_count = 0

            # Send one task to each worker, or until we run out of tasks.
            for worker_rank in range(1, SIZE):
                if tasks_to_do:
                    task_index = tasks_to_do.pop(0)
                    COMM.send(
                        (task_index, X[task_index]), dest=worker_rank, tag=TAG_TASK
                    )
                    tasks_sent_count += 1
                else:
                    break
            # Continue until all results are received.
            status = MPI.Status()
            while tasks_received_count < n_eval:
                # Wait for ANY worker to finish and send a result
                result_data = COMM.recv(
                    source=MPI.ANY_SOURCE, tag=TAG_RESULT, status=status
                )
                worker_rank = status.Get_source()  # Find out who sent it
                task_index, F = result_data

                results[task_index] = F
                tasks_received_count += 1

                # Send a task to the free worker
                if tasks_to_do:
                    new_task_index = tasks_to_do.pop(0)
                    COMM.send(
                        (new_task_index, X[new_task_index]),
                        dest=worker_rank,
                        tag=TAG_TASK,
                    )
                    tasks_sent_count += 1

            out["F"] = np.array(results)

        # --- SERIAL or MPI WORKER ---
        else:
            if X.ndim == 1:  # A single individual from a worker task
                full_mask = np.append(X, True)
                cost = self.cost_calculator.calculate(X)
                sse = self.sse_calculator.calculate(full_mask)
                out["F"] = np.array([cost, sse])
            else:  # A batch of individuals from a serial run
                results = []
                for x_i in X:
                    full_mask = np.append(x_i, True)
                    cost = self.cost_calculator.calculate(x_i)
                    sse = self.sse_calculator.calculate(full_mask)
                    results.append([cost, sse])
                out["F"] = np.array(results)


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
    Runs MTP optimization in serial or with MPI if launched via mpiexec.
    """
    # --- MPI WORKER BRANCH ---
    if IS_MPI and RANK > 0:
        problem_args = {
            "mtp_file": mtp_file,
            "bases_file": bases_file,
            "energies_file": energies_file,
            "counts_file": counts_file,
            "neigh_count": neigh_count,
        }
        mpi_worker_routine(problem_args)
        return

    # --- MASTER / SERIAL BRANCH ---
    if RANK == 0:
        print("--- MTP Optimizer ---")
        if IS_MPI:
            print(f"Mode: MPI Parallel ({SIZE} processes)")
            atexit.register(shutdown_workers)
        else:
            print("Mode: Serial")

    problem = MTPPruningProblem(
        mtp_file, bases_file, energies_file, counts_file, neigh_count
    )

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=BinaryRandomSampling(),
        crossover=UniformCrossover(),
        mutation=BitflipMutation(),
    )

    # print(f"Starting optimization for {n_generations} generations...")
    res = minimize(problem, algorithm, end_condition, seed=seed, verbose=verbose)

    # --- Post-processing and saving (only on master/serial) ---
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
