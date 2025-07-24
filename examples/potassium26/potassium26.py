import os
from mtpoptimizer import (
    run_optimization,
    assemble_new_tree,
    parse_mtp_file,
    write_mtp_file,
)

# --- Configuration ---
DATA_DIR = "data"
MTP_FILE = os.path.join(DATA_DIR, "26.almtp")
BASES_FILE = os.path.join(DATA_DIR, "bases.txt")
ENERGIES_FILE = os.path.join(DATA_DIR, "energies.txt")
COUNTS_FILE = os.path.join(DATA_DIR, "counts.txt")

OUTPUT_DIR = "optimization_results"

if __name__ == "__main__":

    result = run_optimization(
        mtp_file=MTP_FILE,
        bases_file=BASES_FILE,
        energies_file=ENERGIES_FILE,
        counts_file=COUNTS_FILE,
        neigh_count=24,
        output_dir=OUTPUT_DIR,
        n_generations=20,
        pop_size=96,
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
