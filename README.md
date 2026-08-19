# adapol: Adaptive Pole Fitting for Quantum Many-Body Physics

[`adapol`](https://github.com/flatironinstitute/adapol) ("add-a-pole") is a Python package for constructing compact pole approximations of Matsubara functions,

$$
G(\mathrm{i}\nu_n) \approx \sum_{k=1}^{M} \frac{R_k}{\mathrm{i}\nu_n - p_k},
$$

with real poles $p_k$ and scalar or matrix-valued residues $R_k$, using the AAA rational approximation algorithm and nonlinear optimization. Given Matsubara data, or an existing pole expansion (for example, a discretized spectral density or a discrete Lehmann representation), `adapol` finds an accurate approximation with a specified maximum number of poles, or as few poles as possible. A typical application is hybridization fitting: constructing a compact bath representation of a given hybridization function.

## Installation

```
pip install adapol
```

The only dependencies are `numpy` and `scipy`.

**Note:** the interface described below requires a newer version of `adapol`, which has not yet been released on PyPI. For now, install from source:

```
pip install git+https://github.com/flatironinstitute/adapol
```

## Usage

`adapol` provides three main functions:

- **`approx_freq_aaa(F, Z, ...)`** fits frequency data `F`, sampled at (typically Matsubara) points `Z`, with a sum of simple poles, using the AAA algorithm. The number of poles is controlled by a pole budget `max_n_poles` and/or a AAA error tolerance `aaa_tol`.
- **`approx_sop_fast(poles, residues, beta, ...)`** approximates a given sum of poles by a (hopefully) smaller one in a single AAA pass. The number of poles is again controlled by `max_n_poles` and/or `aaa_tol`, and an optional `nonlinear_optimization` step refines the pole locations.
- **`approx_sop_tol(poles, residues, tol, beta, ...)`** finds the smallest sum of poles whose actual error (in $L^2(\tau)$ and $l^2(i \omega_n)$) is below the tolerance `tol`.

## Examples

Three example notebooks demonstrate the usage of these functions in detail. We recommend reading them in the following order.

- [`semicircle.ipynb`](https://flatironinstitute.github.io/adapol/latest/examples/semicircle.html) — fitting data with a continuous spectrum (semicircular density): the stopping criteria, the nonlinear optimization option, and the error metric.
- [`discrete.ipynb`](https://flatironinstitute.github.io/adapol/latest/examples/discrete.html) — fitting multi-orbital data with a discrete spectrum, including an experiment on how the required number of poles scales with the number of orbitals.
- [`hubbarddimer.ipynb`](https://flatironinstitute.github.io/adapol/latest/examples/hubbarddimer.html) — analytic continuation benchmark for the Hubbard dimer.

## Documentation

The [reference documentation](https://flatironinstitute.github.io/adapol/latest/api.html) for the three functions also describes in detail how to use them, as well as information on the algorithms they implement. The same information is contained in the docstrings, e.g. `help(adapol.approx_freq_aaa)`.

## Citation

If you use this package in your research, please include a reference to this GitHub repository, and cite the following references:

1. Huang, Zhen, Emanuel Gull, and Lin Lin. "[Robust analytic continuation of Green's functions via projection, pole estimation, and semidefinite relaxation](https://doi.org/10.1103/PhysRevB.107.075151)," Phys. Rev. B 107, 075151 (2023).
2. Huang, Zhen, Denis Golež, Hugo U. R. Strand, and Jason Kaye. "[Automated evaluation of imaginary time strong coupling diagrams by sum-of-exponentials hybridization fitting](https://doi.org/10.21468/SciPostPhys.19.5.121)," SciPost Phys. 19 (5), 121 (2025).

## License

`adapol` is distributed under the GNU General Public License v3.0 (see [LICENSE](LICENSE)).
