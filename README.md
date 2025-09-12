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

<h3 align="center">MTP Pruning Optimizer</h3>

  <p align="center">
    A evolutionary algorithim based 
    <br />
    <a href="https://github.com/RichardZJM/MTP_basis_optimization"><strong>GitHub »</strong></a>
    <br />
    <br />
    <a href="https://github.com/RichardZJM/MTP_basis_optimization">Read Paper</a>
  </p>
</div>

<!-- ABOUT THE PROJECT -->

## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

Here's a blank template to get started. To avoid retyping too much info, do a search and replace with your text editor for the following: `RichardZJM`, `MTP_basis_optimization`, `twitter_handle`, `linkedin_username`, `email_client`, `email`, `project_title`, `project_description`, `project_license`

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

As this pruning technique is a _post-training_ method you will need to have a trained MTP prepared ahead of time. Since pruned MTP often inherit certain traits of the base potential, you should ensure the base potential is fitted to satisfaction. You can consider using [MLIP-3](https://gitlab.com/ashapeev/mlip-3) to train MTPs with or without active learning.

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

To begin pruning you will need the four components of the Matrix problem:

- [ ] The $\mathbf{X}^\intercal\mathbf{WX}$ Matrix (binary file)
- [ ] The $\mathbf{X}^\intercal\mathbf{WY}$ Vector (binary file)
- [ ] The $\mathbf{Y}^\intercal\mathbf{Y}$ Value (scalar)
- [ ] The average number of neighbors (scalar)

These are obtained using the `extract_problem` commands from my fork. These four values can then be inputted into the `run_optimization` command to run the optimization algorithm. You can try examples/ni20 which has the components precalculated for Nickel with a base potential of level 20. The dataset is from [23-Single-Element-DNPs
](https://github.com/saidigroup/23-Single-Element-DNPs).

`run_optimization` takes the following commands. You may see a warning if the more regularization is recommended to ensure numerical stability.

```python
def run_optimization(
    mtp_file, # File path to the base MTP
    xtwx, # File path to the binary XTWX file
    xtwy, # File path to the binary XTWY file
    yty, # Value of the yTy (float)
    neigh_count, # Average number of neighbors (Float)
    regularization=0, # Lamba for Tikhonov regularization
    output_dir="outputs", # Directory to write results
    end_condition=("n_gen", 1000), # End conditions n_gen, (int generation count) or time (seconds)
    pop_size=96, # Population size
    seed=None, # Seed if specified
    show_plot=True, # Show a plot when optimization is complete
    verbose=True, # Show the details of each generation
    init_pop=None, # Numpy array of inital population (p by n numpy array)
    algorithim="nsga", # Which evolutionary algorithim to use (moead or nsga)
    mutation_rate=None, # Mutation rate (float). Single bitfilp if None.
):
```

If built with MPI enabled, you can run with parallel fitness evaluations using `mpirun`.

```sh
mpirun -np 12 python ni20.py
```

After the evolutionary algorithm is completed, the output directory will contain two files, `pareto_objectives.csv` and `pareto_population.csv`.

`pareto_objectives.csv`

<p align="right">(<a href="#readme-top">back to top</a>)</p>

See the [open issues](https://github.com/RichardZJM/MTP_basis_optimization/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Top contributors:

<a href="https://github.com/RichardZJM/MTP_basis_optimization/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=RichardZJM/MTP_basis_optimization" alt="contrib.rocks image" />
</a>

<!-- LICENSE -->

## License

Distributed under the project_license. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->

## Contact

Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - email@email_client.com

Project Link: [https://github.com/RichardZJM/MTP_basis_optimization](https://github.com/RichardZJM/MTP_basis_optimization)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->

## Acknowledgments

- []()
- []()
- []()

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
