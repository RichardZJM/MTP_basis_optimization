<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->

<a id="readme-top"></a>

<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![project_license][license-shield]][license-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">

<h3 align="center">MTP Basis Optimization</h3>

  <p align="center">
    Cost-aware pruning of basis functions applied to the MTP
    <br />
    <a href="https://github.com/RichardZJM/MTP_basis_optimization"><strong>GitHub »</strong></a>
    <br />
    <br />
    <a href="https://github.com/RichardZJM/MTP_basis_optimization">Read Paper</a>
  </p>
</div>

<!-- ABOUT THE PROJECT -->

## About The Project

The Moment Tensor Potential (MTP) is a popular machine learning interatomic potentials which relies on linear regression of a basis set. This basis set is calculated through a recursive-style approach which means that these basis functions have highly asymmetric costs. Previously, MTPs were defined by level-based scheme which picked which limitations on radial and angular contributions using empirical, training-data-agnostic rules. This library is uses multiobjetive optimization to choose more cost-effective MTPs instead, achieving up to $7\,\times$ speedups over the previous level-based scheme, as shown in the paper. The resultant potential are compatible with the [MLIP-3](https://gitlab.com/ashapeev/mlip-3) package and existing LAMMPS implementations.

Disclaimer: this repository was refactored, modified, and commented with the help of AI coding tools at multiple points.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

As this pruning technique is a _post-training_ method you will need to have a trained MTP prepared ahead of time. Since pruned MTP often inherit certain traits of the base potential, you should ensure the base potential is fitted to satisfaction. You can use [MLIP-3](https://gitlab.com/ashapeev/mlip-3) to train MTPs with or without active learning.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/RichardZJM/MTP_basis_optimization.git
   ```
2. For serial usage, install as an editable pip-compatible package.
   ```sh
   pip install -e /path/to/repo
   ```
   For larger base potentials, you can install it with MPI enabled.
   ```
   pip install -e /path/to/repo[mpi]
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->

## Usage

After you have a base potential prepared, you need to extract the matrix problem for usage in this package. This extraction is performed by my [fork](https://github.com/RichardZJM/mlip3-extract) of the MLIP-3 package.

To begin pruning you will need the four components of the matrix problem:

- [ ] The $\mathbf{X}^\intercal\mathbf{WX}$ Matrix (binary file)
- [ ] The $\mathbf{X}^\intercal\mathbf{WY}$ Vector (binary file)
- [ ] The $\mathbf{Y}^\intercal\mathbf{Y}$ Value (scalar)
- [ ] The average number of neighbors (scalar)

These are obtained using the `extract_problem` commands from my fork. These four values can then be inputted into the `run_optimization` command to run the optimization algorithm. You can try `examples/ni20` which has the matrix problem's components precalculated for Nickel with a base potential of level 20 which can be executed on a single core. The dataset is from [23-Single-Element-DNPs
](https://github.com/saidigroup/23-Single-Element-DNPs). The examples also include Nickel for a base potential of level 28 (`examples/ni28`)and Silicon-Oxygen for a level of 28 (`examples/sio28`). These were included in the paper but may require high performance computing resources to run.

`run_optimization` takes the following commands. You may see a warning if the more regularization is recommended to ensure numerical stability.

```python
def run_optimization(
    mtp_file, # File path to the base MTP
    xtwx, # File path to the binary XTWX file
    xtwy, # File path to the binary XTWY file
    yty, # Value of the yTy (fp64)
    neigh_count, # Average number of neighbors (fp64)
    regularization=0, # Lamba for Tikhonov regularization
    output_dir="outputs", # Directory to write results
    end_condition=("n_gen", 1000), # End conditions n_gen, (int generation count) or time (seconds)
    pop_size=96, # Population size (int)
    seed=None, # Seed if specified (int)
    show_plot=True, # Show a plot when optimization is complete
    verbose=True, # Show the details of each generation
    init_pop=None, # Numpy array of inital population (p by n numpy array of booleans) Used to continue runs.
    algorithim="nsga", # Which evolutionary algorithim to use (moead or nsga)
    mutation_rate=None, # Mutation rate (float). Single bitfilp if None.
):
```

If built with MPI enabled, you can run with parallel fitness evaluations using `mpirun`.

```sh
mpirun -np 12 python ni20.py
```

After the evolutionary algorithm is completed, the output directory will contain two files, `pareto_objectives.csv` and `pareto_population.csv`.

`pareto_objectives.csv` is a comma-separated values folder containing 2 columns and $n$ rows, where $n$ is the number of non-dominated individuals in the final solution of the optimization. The first column consists of floating-point values between 0 and 1, which is the cost of the individuals normalized to the base potential. The second column is the training loss which is typically 1 or larger, which is the loss of the individuals normalized by the loss of the base potential. `pareto_population.csv` is a comma-separated values where rows represent the binary vectors of which basis functions are pruned (0 means pruned, 1 means retrained).

After the optimization is complete, you can select pruned potentials which reflect your desired accuracy-cost balance. These pruned potentials then need to be fitted with MLIP-3. There are two options for initializing these pruned potential.

1. Random initialization, where a blank potential is prepared to be fitting using MLIP-3
2. Inherited initialization, where the radial parameters are inherited from the base potential and the linear parameters are calculated using least squares. This serves as the initial weight for further fitting with MLIP-3.

In our experience, the inherited initialization yielded better losses for pruned potential with more that 5% of the base potential's cost. In the cases where the random initialization performed better, it was always by a slim margin. Inherited initialization also tends to cause the pruned potential's properties to more closely resemble the base potential, especially for basic properties such as lattice parameter.

Writing a blank potential for random initalization can be performed as follows. This example is a snippet from `examples/ni20`.

```python
  original_mtp = parse_mtp_file(MTP_FILE)   # Read the base MTP from the path
  # Generate the blank potential from a binary vector obtained from the optimization
  new_mtp_dict = assemble_new_tree(original_mtp, best_sse_mask)
  output_mtp_path = os.path.join(OUTPUT_DIR, "pruned_mtp.almtp")
  write_mtp_file(new_mtp_dict, output_mtp_path)
```

Writing a trained potential for inherited initialization can be performed as follows. This example is a snippet from `examples/write_inherited`.

```python
  from mtpoptimizer.sse import SSECalculator

  sample_individual = np.genfromtxt("mask.csv", delimiter=",") # Read binary vector

  # Load data
  DATA_DIR = "../ni20/data"
  ORIG_MTP_FILE = os.path.join(DATA_DIR, "20.almtp")
  XTWX_FILE = os.path.join(DATA_DIR, "xtwx.bin")
  XTWY_FILE = os.path.join(DATA_DIR, "xtwy.bin")

  xtwx = np.fromfile(XTWX_FILE, dtype=np.float64)
  xtwy = np.fromfile(XTWY_FILE, dtype=np.float64)
  xtwx = np.reshape(xtwx, (len(xtwy), len(xtwy)))
  yty = 1308558.94848743616603

  original_mtp = parse_mtp_file(ORIG_MTP_FILE)  # Read base potential
  calc = SSECalculator(xtwx, xtwy, yty, regularization=1e-4)  # Initalize calculator with same params as optimizer

  mask = sample_individual.astype(bool) # Ensure the binary vector are booleans

  # We need to append a True value for each species since we never prune species coeffs
  full_mask = np.append(np.full((1), True, dtype=bool), mask)
  theta, sse = calc.calculate(full_mask, True)  # The second argument tell the calcultor to provide theta

  new_mtp_dict = assemble_new_tree(original_mtp, mask, theta)
  write_mtp_file(new_mtp_dict, "sample.almtp")
```

The resultant potential can then be fitted using the training dataset in MLIP-3 as usual.

See the [open issues](https://github.com/RichardZJM/MTP_basis_optimization/issues) for known issues.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->

## Contributors

<a href="https://github.com/RichardZJM/MTP_basis_optimization/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=RichardZJM/MTP_basis_optimization" alt="contrib.rocks image" />
</a>

<!-- LICENSE -->

## License and Citing

Distributed under the project_license. See `LICENSE.txt` for more information.

<!--
If you find use from the work and use it in a scientific publication, llease consider citing this research with the following:

```bibtex
To be submitted
``` -->

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->

## Contact

Zijian (Richard) Meng [RichardZJM](https://github.com/RichardZJM) — [richardzjm.com](www.richardzjm.com)

Project Link: [https://github.com/RichardZJM/MTP_basis_optimization](https://github.com/RichardZJM/MTP_basis_optimization)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->

## Acknowledgments

- [Karim Zongo](https://gitlab.com/Kazongogit)
- [Matthew Thoms](https://gitlab.com/mattmtl)
- [Ryan Eric Grant (Supervisor)](https://smithengineering.queensu.ca/directory/faculty/ryan-grant.html)
- [Laurent Karim Béland (Supervisor)](https://smithengineering.queensu.ca/directory/faculty/laurent-karim-beland.html)

This README was made with [Best-README-Template](https://github.com/othneildrew/Best-README-Template).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/RichardZJM/MTP_basis_optimization.svg?style=for-the-badge
[contributors-url]: https://github.com/RichardZJM/MTP_basis_optimization/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/RichardZJM/MTP_basis_optimization.svg?style=for-the-badge
[forks-url]: https://github.com/RichardZJM/MTP_basis_optimization/network/members
[stars-shield]: https://img.shields.io/github/stars/RichardZJM/MTP_basis_optimization.svg?style=for-the-badge
[stars-url]: https://github.com/RichardZJM/MTP_basis_optimization/stargazers
[issues-shield]: https://img.shields.io/github/issues/RichardZJM/MTP_basis_optimization.svg?style=for-the-badge
[issues-url]: https://github.com/RichardZJM/MTP_basis_optimization/issues
[license-shield]: https://img.shields.io/github/license/RichardZJM/MTP_basis_optimization.svg?style=for-the-badge
[license-url]: https://github.com/RichardZJM/MTP_basis_optimization/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com
