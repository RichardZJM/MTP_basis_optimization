import numpy as np
import multiprocessing
import os

from pymoo.core.problem import ElementwiseProblem, StarmapParallelization
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling

from .cost import MTPCostCalculator
from .sse import SSECalculator
from .mtpio import parse_mtp_file, write_mtp_file
from .assembly import assemble_new_tree


class MTPPruningProblem(ElementwiseProblem):
    def __init__(self, cost_calculator, sse_calculator, n_var, **kwargs):
        self.cost_calculator = cost_calculator
        self.sse_calculator = sse_calculator
        super().__init__(n_var=n_var, n_obj=2, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        full_mask = np.append(x, True)

        cost = self.cost_calculator.calculate(x)
        sse = self.sse_calculator.calculate(full_mask)

        out["F"] = np.array([cost, sse])


def run_optimization(
    mtp_file,
    bases_file,
    energies_file,
    counts_file,
    neigh_count,
    radial_basis_size,
    output_dir="outputs",
    device="cpu",
    n_generations=1000,
    pop_size=96,
    n_processes=4,
    seed=42,
    show_plot=True,
):
    """
    Runs a multi-objective optimization to prune an MTP potential.

    Args:
        mtp_file (str): Path to the initial MTP file.
        bases_file (str): Path to the bases.txt file.
        energies_file (str): Path to the energies.txt file.
        counts_file (str): Path to the counts.txt file.
        neigh_count (int): Estimated number of neighbors per neighborhood.
        radial_basis_size (int): Size of each radial basis set.
        output_dir (str): Directory to save results.
        device (str): 'cpu' or 'gpu'.
        n_generations (int): Number of generations for NSGA-II.
        pop_size (int): Population size for NSGA-II.
        n_processes (int): Number of parallel processes to use.
        seed (int): Random seed for reproducibility.
        show_plot (bool): Whether to display the Pareto front plot.

    Returns:
        pymoo.Result: The result object from the optimization.
    """
    print("--- MTP Optimizer ---")

    # 1. Load data
    print("1. Loading data...")
    mtp_data = parse_mtp_file(mtp_file)
    if mtp_data is None:
        return None

    bases = np.genfromtxt(bases_file, delimiter=" ")
    energies = np.genfromtxt(energies_file, delimiter=",")
    counts = np.genfromtxt(counts_file, delimiter=",")

    n_var = mtp_data["alpha_scalar_moments"]

    # 2. Initialize calculators
    print(f"2. Initializing calculators (device: {device})...")
    cost_calculator = MTPCostCalculator(mtp_data, neigh_count, radial_basis_size)
    sse_calculator = SSECalculator(bases, energies, counts, device=device)

    # 3. Set up parallelization
    pool = multiprocessing.Pool(n_processes)
    runner = StarmapParallelization(pool.starmap)

    # 4. Define the optimization problem
    problem = MTPPruningProblem(
        cost_calculator=cost_calculator,
        sse_calculator=sse_calculator,
        n_var=n_var,
        elementwise_runner=runner,
    )

    # 5. Define the algorithm
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=BinaryRandomSampling(),
        crossover=UniformCrossover(),
        mutation=BitflipMutation(),
    )

    # 6. Run the optimization
    print(f"3. Starting optimization for {n_generations} generations...")
    res = minimize(
        problem, algorithm, ("n_gen", n_generations), seed=seed, verbose=True
    )
    print(f"Optimization finished in {res.exec_time} seconds.")

    pool.close()

    # 7. Process and save results
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sorted_indices = np.argsort(res.F[:, 0])
    sorted_F = res.F[sorted_indices]
    sorted_X = res.X[sorted_indices]

    pop_path = os.path.join(output_dir, "pareto_population.csv")
    obj_path = os.path.join(output_dir, "pareto_objectives.csv")

    np.savetxt(pop_path, sorted_X.astype(int), delimiter=",", fmt="%d")
    np.savetxt(obj_path, sorted_F, delimiter=",")
    print(f"Saved Pareto front population to {pop_path}")
    print(f"Saved Pareto front objectives to {obj_path}")

    # 8. Plot results
    if show_plot:
        print("4. Displaying plot...")
        plot = Scatter(
            title="Pareto Front", labels=["Cost Heuristic", "Sum of Squared Error"]
        )
        plot.add(res.F, facecolor="none", edgecolor="red", s=40)
        plot.show()

    return res
