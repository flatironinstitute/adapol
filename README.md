# adapol: Adaptive Pole Fitting for Quantum Many-Body Physics
[`adapol`](https://github.com/Hertz4/Adapol) (pronounced "add a pole") is a python package for fitting Matsubara functions with the following form:
```math
G(\mathrm i \omega_k) = \sum_l \frac{V_lV_l^{\dagger}}{\mathrm i\omega_k - E_l}.
```

Current applications include
(1) hybridization fitting, (2) analytic continuation.

We also provide a [TRIQS](https://triqs.github.io/) interface if the Matsubara functions are stored in `triqs` Green's function container.

# Installation
`adapol` has `numpy` and `scipy` as its prerequisites. [`cvxpy`](https://www.cvxpy.org/) is also required for hybridization fitting of matrix-valued (instead of scalar-valued) Matsubara functions.

To install `adapol`, run
```terminal
pip install adapol
```


# Examples
In the `examples` directory, we provide two examples [`discrete.ipynb`](https://github.com/Hertz4/adapol/blob/main/example/discrete.ipynb) and [`semicircle.ipynb`](https://github.com/Hertz4/adapol/blob/main/example/semicircle.ipynb), showcasing how to use `adapol` for both discrete spectrum and continuous spectrum.

In these notebooks, we also demonstrate how to use our code through the triqs interface.


There are two choices for doing hybridization fitting. One can either fit with desired accuracy tolerance `tol`:
```python
bath_energy, bath_hyb, final_error, func = hybfit(Delta, Z, tol=tol)
```
Or fit with specified number of interpolation points `Np`:
```python
bath_energy, bath_hyb, final_error, func = hybfit(Delta, Z, Np = Np)
```
Here `bath_energy` $E$ and `bath_hyb` $V$ are desired quantities of hybridization orbitals. 

In more sophisticated applications, one might need to specify other flags, such as `maxiter`, `cleanflag` and `disp`. See the documentation for details.


### Triqs interface

For triqs users, if the Green's function data `delta_triqs` is stored as a `triqs.gf.Gf` object or `triqs.gf.BlockGf` object, then the hybridization fitting could be done using the `hybfit_triqs` function:
```python
bathhyb, bathenergy, delta_fit, final_error = hybfit_triqs(delta_triqs, tol=tol, debug=True)
```
### Analytic continuation

To use this code for analytic continuation is similar, and we refer to the documentation for details.

# References
To cite this work, please include a reference to this GitHub repository, and
cite the following references:

1. Huang, Zhen, Emanuel Gull, and Lin Lin. "Robust analytic continuation of Green's functions via projection, pole estimation, and semidefinite relaxation." Physical Review B 107.7 (2023): 075151.
2. Mejuto-Zaera, Carlos, et al. "Efficient hybridization fitting for dynamical mean-field theory via semi-definite relaxation." Physical Review B 101.3 (2020): 035143.
3. Nakatsukasa, Yuji, Olivier Sète, and Lloyd N. Trefethen. "The AAA algorithm for rational approximation." SIAM Journal on Scientific Computing 40.3 (2018): A1494-A1522.
