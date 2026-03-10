import os

# Disable the numpy thread parallelisim (we use MPI instead)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


import numpy as np
from mtpoptimizer import (
    run_optimization,
    assemble_new_tree,
    parse_mtp_file,
    write_mtp_file,
)

# --- Configuration ---
DATA_DIR = "data"
MTP_FILE = os.path.join(DATA_DIR, "20.almtp")
DATA_DIR = "/home/richa/Documents/Projects/mlip-3/test/examples/09.extract_problem/out"
TR_XTWX_FILE = os.path.join(DATA_DIR, "xtwx.bin")  # Get this from the MLIP-3 fork
TR_XTWY_FILE = os.path.join(DATA_DIR, "xtwy.bin")  # Get this from the MLIP-3 fork

DATA_DIR = "/home/richa/Documents/Projects/mlip-3/test/examples/09.extract_problem/val"
VAL_XTWX_FILE = os.path.join(DATA_DIR, "xtwx.bin")  # Get this from the MLIP-3 fork
VAL_XTWY_FILE = os.path.join(DATA_DIR, "xtwy.bin")  # Get this from the MLIP-3 fork

OUTPUT_DIR = "optimization_results"

if __name__ == "__main__":

    tr_xtwx = np.fromfile(TR_XTWX_FILE, dtype=np.float64)
    tr_xtwy = np.fromfile(TR_XTWY_FILE, dtype=np.float64)
    tr_xtwx = np.reshape(tr_xtwx, (len(tr_xtwy), len(tr_xtwy)))

    val_xtwx = np.fromfile(VAL_XTWX_FILE, dtype=np.float64)
    val_xtwy = np.fromfile(VAL_XTWY_FILE, dtype=np.float64)
    val_xtwx = np.reshape(val_xtwx, (len(val_xtwy), len(val_xtwy)))

    # val_xtwx = None
    # val_xtwy = None
    # val_xtwx = None

    result = run_optimization(
        mtp_file=MTP_FILE,
        tr_xtwx=tr_xtwx,
        tr_xtwy=tr_xtwy,
        tr_ytwy=57642.050295515684411,  # Get this from the MLIP-3 fork
        val_xtwx=val_xtwx,
        val_xtwy=val_xtwy,
        val_ytwy=2525.066019002886605,  # Get this from the MLIP-3 fork
        neigh_count=20.528098,  # Get this from the MLIP-3 fork
        regularization=1e-4,
        output_dir=OUTPUT_DIR,
        end_condition=("n_iter", 1),
        pop_size=96,
        show_plot=True,
        seed=None,
        algorithm="nsga",
    )

    if result:
        print("\n--- Post-processing: Assembling a new MTP ---")
        # Example: Choose the solution with the lowest SSE from the Pareto front
        pareto_front = result.F
        pareto_pop = result.X

        # Get the individual with the lowest SSE
        best_sse_idx = pareto_front[:, 1].argmin()
        best_sse_mask = pareto_pop[best_sse_idx].astype(bool)

        print(f"Lowest SSE found: {pareto_front[best_sse_idx][1]:.4f}")
        print(f"Corresponding cost: {pareto_front[best_sse_idx][0]:.4f}")

        original_mtp = parse_mtp_file(MTP_FILE)
        new_mtp_dict = assemble_new_tree(original_mtp, best_sse_mask)

        output_mtp_path = os.path.join(OUTPUT_DIR, "pruned_mtp.almtp")
        write_mtp_file(new_mtp_dict, output_mtp_path)
