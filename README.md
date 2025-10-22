<a id="readme-top"></a>

<br />
<div align="center">

<h1 align="center">MTP Basis Optimization</h1>

  <p align="center">
    Cost-aware pruning of basis functions for Moment Tensor Potentials (MTP)
    <br />
    <a href="https://github.com/RichardZJM/MTP_basis_optimization"><strong>GitHub »</strong></a>
    <br />
    <br />
    <a href="https://github.com/RichardZJM/MTP_basis_optimization">Read Paper »</a>
    <br />
    <a href="https://github.com/RichardZJM/mlip3-extract">Matrix Problem Extraction Code »</a>

  </p>
</div>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## About The Project

The Moment Tensor Potential (MTP) is a widely used machine-learning interatomic potential that relies on linear regression of a basis set. The basis functions are generated in a recursive style, producing functions with highly asymmetric computational costs. Previously, MTPs were defined using a level-based scheme that selected radial and angular limits by empirical, training-data-agnostic rules. This library applies multiobjective optimization (via [pymoo](https://pymoo.org/algorithms/moo/moead.html)) to select more cost-effective MTPs, achieving up to \$7\times\$ speedups, as reported in the linked paper. The resulting potentials are compatible with the [MLIP-3](https://gitlab.com/ashapeev/mlip-3) package and existing LAMMPS implementations.

Disclaimer: this repository was refactored, modified, and commented with the assistance of AI coding tools at multiple points.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

This pruning technique is a _post-training_ method: you must provide a trained MTP. Because pruned MTPs often inherit characteristics of the base potential, ensure the base potential is adequately fitted before pruning. Use [MLIP-3](https://gitlab.com/ashapeev/mlip-3) to train MTPs with or without active learning.

### Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/RichardZJM/MTP_basis_optimization.git
   ```

2. For serial usage, install as an editable pip package:

   ```sh
   pip install -e /path/to/repo
   ```

   For larger base potentials with MPI support:

   ```sh
   pip install -e /path/to/repo[mpi]
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

After preparing a base potential, extract the matrix problem required by this package. Extraction is performed using a fork of MLIP-3: [https://github.com/RichardZJM/mlip3-extract](https://github.com/RichardZJM/mlip3-extract).

The pruning procedure requires the following components:

- The $\mathbf{X}^\intercal\mathbf{WX}$ matrix (binary file)
- The $\mathbf{X}^\intercal\mathbf{Wy}$ vector (binary file)
- The $\mathbf{y}^\intercal\mathbf{Wy}$ value (scalar)
- The average number of neighbors (scalar)

These are obtained using the `extract_problem` commands from the fork. The four values are inputs to the `run_optimization` command. A cheap-to-run example problems is provided in `examples/ni20` (Nickel, base potential level 20). The dataset for that example is from [23-Single-Element-DNPs](https://github.com/saidigroup/23-Single-Element-DNPs). Examples also include `examples/ni28` (Nickel, level 28) and `examples/sio28` (Silicon–Oxygen, level 28) although these may require high-performance computing resources. All included examples have the matrix problem components precalculated.

The `run_optimization` function accepts the following parameters:

```python
def run_optimization(
    mtp_file,             # Path to the base MTP file
    xtwx,                 # Path to binary XTWX file
    xtwy,                 # Path to binary XTWy file
    ytwy,                  # yTWy value (float64)
    neigh_count,          # Average number of neighbors (float64)
    regularization=0.0,   # Lambda for Tikhonov regularization
    output_dir="outputs",# Directory to write results
    end_condition=("n_gen", 1000), # End condition: ("n_gen", int) or ("time", seconds)
    pop_size=96,          # Population size (int)
    seed=None,            # Random seed (int)
    show_plot=True,       # Show plot when optimization completes
    verbose=True,         # Print generation details while running
    init_pop=None,        # numpy array (p by n) of booleans for initial population
    algorithm="nsga",    # Evolutionary algorithm: "moead" or "nsga"
    mutation_rate=None,   # Mutation rate (float). Single bitflip if None.
):
    ...
```

If the package is installed with MPI support, parallel fitness evaluations can be executed with `mpirun`:

```sh
mpirun -np 12 python ni20.py
```

### Output

After the evolutionary algorithm completes, the output directory contains two files: `pareto_objectives.csv` and `pareto_population.csv`.

- `pareto_objectives.csv` is a CSV file with two columns and $n$ rows, where $n$ is the number of non-dominated individuals in the final front. The first column is the cost of each individual normalized to the base potential (values in \[0, 1]). The second column is the training loss, typically ≥ 1, given as the loss of each individual normalized by the loss of the base potential.

- `pareto_population.csv` is a CSV file where each of $n$ rows is a binary vector indicating which basis functions are retained (1) or pruned (0). Rows correspond to the non-dominated individuals in the final front.

### Fitting pruned potentials

Pruned potentials must be re-fitted with MLIP-3 before production usage. Two initialization strategies are possible:

1. **Random initialization**: prepare a blank potential and fit with MLIP-3.
2. **Inherited initialization**: inherit radial parameters from the base potential and compute linear parameters using least squares. These parameters serve as initial weights for subsequent fitting with MLIP-3.

Empirically, inherited initialization yields better losses for pruned potentials that retain more than 5% of the base potential's computational cost. When random initialization is superior, the improvement is typically marginal. Inherited initialization also tends to better preserve base-potential properties (for example, lattice parameters).

Writing a blank potential (random initialization)—snippet from `examples/ni20`:

```python
original_mtp = parse_mtp_file(MTP_FILE)   # Read base MTP
# Generate blank potential from a binary vector obtained from optimization
new_mtp_dict = assemble_new_tree(original_mtp, best_sse_mask)
output_mtp_path = os.path.join(OUTPUT_DIR, "pruned_mtp.almtp")
write_mtp_file(new_mtp_dict, output_mtp_path)
```

Writing an inherited potential—snippet from `examples/write_inherited`:

```python
from mtpoptimizer.sse import SSECalculator

sample_individual = np.genfromtxt("mask.csv", delimiter=",")  # Read binary vector

# Load data
DATA_DIR = "../ni20/data"
ORIG_MTP_FILE = os.path.join(DATA_DIR, "20.almtp")
XTWX_FILE = os.path.join(DATA_DIR, "xtwx.bin")
XTWY_FILE = os.path.join(DATA_DIR, "xtwy.bin")

xtwx = np.fromfile(XTWX_FILE, dtype=np.float64)
xtwy = np.fromfile(XTWY_FILE, dtype=np.float64)
xtwx = np.reshape(xtwx, (len(xtwy), len(xtwy)))
ytwy = 1308558.94848743616603

original_mtp = parse_mtp_file(ORIG_MTP_FILE)  # Read base potential
calc = SSECalculator(xtwx, xtwy, ytwy, regularization=1e-4)  # Initialize calculator with same parameters as optimizer

mask = sample_individual.astype(bool)  # Ensure binary vector is boolean

# Append a True value for each species since species coefficients are never pruned
full_mask = np.append(np.full((1,), True, dtype=bool), mask)
theta, sse = calc.calculate(full_mask, get_theta=True) # Second argument requests the least squares params

new_mtp_dict = assemble_new_tree(original_mtp, mask, theta)
write_mtp_file(new_mtp_dict, "sample.almtp")
```

The generated potential can then be fitted using the training dataset in MLIP-3 as usual.

See the project's [open issues](https://github.com/RichardZJM/MTP_basis_optimization/issues) for known issues.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contributors

<a href="https://github.com/RichardZJM/MTP_basis_optimization/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=RichardZJM/MTP_basis_optimization" alt="contrib.rocks image" />
</a>

## License and Citing

Distributed under the MIT License. See `LICENSE` for details.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contact

Zijian (Richard) Meng — [RichardZJM](https://github.com/RichardZJM) — [https://richardzjm.com](https://richardzjm.com)

Project repository: [https://github.com/RichardZJM/MTP_basis_optimization](https://github.com/RichardZJM/MTP_basis_optimization)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Acknowledgments

- [Karim Zongo](https://gitlab.com/Kazongogit)
- [Matthew Thoms](https://gitlab.com/mattmtl)
- [Ryan Eric Grant (Supervisor)](https://smithengineering.queensu.ca/directory/faculty/ryan-grant.html)
- [Laurent Karim Béland (Supervisor)](https://smithengineering.queensu.ca/directory/faculty/laurent-karim-beland.html)

This README was made with [Best-README-Template](https://github.com/othneildrew/Best-README-Template).

<p align="right">(<a href="#readme-top">back to top</a>)</p>
