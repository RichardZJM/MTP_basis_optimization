import os
import numpy as np
from mtpoptimizer import (
    run_optimization,
    assemble_new_tree,
    parse_mtp_file,
    write_mtp_file,
)

# --- Configuration ---
DATA_DIR = "data"
MTP_FILE = os.path.join(DATA_DIR, "28.almtp")
BASES_FILE = os.path.join(DATA_DIR, "bases.bin")
ENERGIES_FILE = os.path.join(DATA_DIR, "energies.bin")
COUNTS_FILE = os.path.join(DATA_DIR, "counts.bin")

OUTPUT_DIR = "optimization_results"

if __name__ == "__main__":
    parameter_count = (
        2445  # 2445 linear parameters for level 28 (Cant be found from MTP file)
    )

    bases = np.fromfile(BASES_FILE, dtype=np.float64)
    bases = bases.reshape((int(bases.shape[0] / parameter_count), parameter_count))
    mean = bases.mean(axis=0)
    std_dev = bases.std(axis=0)
    bases = (bases - mean) / std_dev
    bases = np.concatenate((bases, np.ones((bases.shape[0], 1))), axis=1)

    energies = np.fromfile(ENERGIES_FILE, dtype=np.float64)
    counts = np.fromfile(COUNTS_FILE, dtype=np.int32)

    result = run_optimization(
        mtp_file=MTP_FILE,
        bases=bases,
        energies=energies,
        counts=counts,
        neigh_count=24,
        regularization=2e-2,
        output_dir=OUTPUT_DIR,
        end_condition=("n_gen", 5),
        pop_size=12 * 24,
        show_plot=True,
    )

    if result:
        print("\n--- Post-processing: Assembling a new MTP ---")
        # Example: Choose the solution with the lowest SSE from the Pareto front
        pareto_front = result.F
        pareto_pop = result.X

        # Get the individual with the lowest SSE
        best_sse_idx = pareto_front[:, 1].argmin()
        best_sse_mask = pareto_pop[best_sse_idx].astype(bool)

        print(f"Lowest SSE found: {pareto_front[best_sse_idx][1]:.6f}")
        print(f"Corresponding cost: {pareto_front[best_sse_idx][0]}")

        original_mtp = parse_mtp_file(MTP_FILE)
        new_mtp_dict = assemble_new_tree(original_mtp, best_sse_mask)

        output_mtp_path = os.path.join(OUTPUT_DIR, "pruned_mtp.almtp")
        write_mtp_file(new_mtp_dict, output_mtp_path)
