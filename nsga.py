import numpy as np
import multiprocessing

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.core.problem import StarmapParallelization

from costs import prune_tree
from ols import calculate_sse
from gols import calculate_sse_gpu
from mtpio import parse_mtp_file, write_mtp_file
from assemblemtp import assemble_new_tree


class mtp(ElementwiseProblem):

    def __init__(self, **kwargs):
        super().__init__(n_var=163, n_obj=2, elementwise_evaluation=True, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.array([prune_tree(x), calculate_sse(np.append(x, True))])


if __name__ == "__main__":

    n_proccess = 12
    pool = multiprocessing.Pool(n_proccess)
    runner = StarmapParallelization(pool.starmap)

    problem = mtp(elementwise_runner=runner)
    problem = mtp()

    algorithm = NSGA2(
        pop_size=96,
        sampling=BinaryRandomSampling(),
        crossover=UniformCrossover(),
        mutation=BitflipMutation(),
    )

    res = minimize(problem, algorithm, ("n_gen", 50), seed=42, verbose=False)
    print(f"Time taken: {res.exec_time} seconds")

    mask = np.ones(163).astype(bool)
    print(prune_tree(mask), calculate_sse(np.append(mask, True)))

    # Sort by first objective in F
    sorted_indices = np.argsort(res.F[:, 0])
    sorted_F = res.F[sorted_indices]
    sorted_X = res.X[sorted_indices]

    # print(sorted_X, sorted_F)
    mtp18 = parse_mtp_file("18.almtp")
    new18 = assemble_new_tree(mtp18, sorted_X[-1])
    write_mtp_file(new18, "tmp.almtp")

    plot = Scatter()
    plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
    plot.add(res.F, facecolor="none", edgecolor="red")
    plot.show()
