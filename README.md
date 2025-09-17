# adapol: Adaptive Pole Fitting for Quantum Many-Body Physics
[`adapol`](https://github.com/Hertz4/Adapol) (pronounced "add a pole") is a python package for fitting Matsubara functions with the following form (in the fermionic case):
```math
G(\mathrm i \omega_k) = \sum_l \frac{V_lV_l^{\dagger}}{\mathrm i\omega_k - E_l}.
```
Or in the bosonic case,
```math
G(\mathrm i \omega_k) = \sum_l V_lV_l^{\dagger}\frac{E_l}{\mathrm i\omega_k - E_l}. 
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



# Documentation

See the detailed [documentation](https://flatironinstitute.github.io/adapol/) for physical background, algorithms and user manual.

`Adapol` is a stand-alone package. For TRIQS users, we also provide a TRIQS interface. See [user manual](https://flatironinstitute.github.io/adapol/latest/python.html#triqs-interface) for details.

# Examples
In the `tutorial` page, we provide two examples [`discrete.ipynb`](https://flatironinstitute.github.io/adapol/latest/tutorials/discrete.html) and [`semicircle.ipynb`](https://flatironinstitute.github.io/adapol/latest/tutorials/semicircle.html), showcasing how to use `adapol` for both discrete spectrum and continuous spectrum.

In these notebooks, we also demonstrate how to use our code through the triqs interface.

# References
To cite this work, please include a reference to this GitHub repository, and
cite the following references:

1. Huang, Zhen, Emanuel Gull, and Lin Lin. "Robust analytic continuation of Green's functions via projection, pole estimation, and semidefinite relaxation." Physical Review B 107.7 (2023): 075151.
2. Mejuto-Zaera, Carlos, et al. "Efficient hybridization fitting for dynamical mean-field theory via semi-definite relaxation." Physical Review B 101.3 (2020): 035143.
3. Nakatsukasa, Yuji, Olivier Sète, and Lloyd N. Trefethen. "The AAA algorithm for rational approximation." SIAM Journal on Scientific Computing 40.3 (2018): A1494-A1522.
