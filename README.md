## Introduction
[`AAAdapol`](https://github.com/Hertz4/AAAdapol) (pronounced "add a pole") is a python package for fitting Matsubara functions with the following form:
```math
G(\mathrm i \omega_k) = \sum_l \frac{|v_l\rangle\langle v_l|}{\mathrm i\omega_k - E_l}.
```
AAAdapol is short for **A**ntoulas–**A**nderson **Ad**aptive **pol**e-fitting. 

Current applications include
(1) hybridization fitting, (2) analytic continuation.

We also provide a [TRIQS](https://triqs.github.io/) interface if the Matsubara functions are stored in `triqs` Green's function container.

## Installation
`AAAdapol` has `numpy` and `scipy` as its prerequisites. [`cvxpy`](https://www.cvxpy.org/) is also required for hybridization fitting of matrix-valued (instead of scalar-valued) Matsubara functions.

To install `AAAdapol`, run
```terminal
pip install aaadapol
```


## Examples
In the `examples` directory, we provide two examples [`discrete.ipynb`](https://github.com/Hertz4/AAAdapol/blob/main/example/discrete.ipynb) and [`semicircle.ipynb`](https://github.com/Hertz4/AAAdapol/blob/main/example/semicircle.ipynb), showcasing how to use `AAAdapol` for both discrete spectrum and continuous spectrum. We also demonstrate how to use our code through the triqs interface.

Below is a quick introduction through the following toy example:
### Setup
```python
import numpy as np
beta = 20
Z = np.linspace(-25.,25.,26)*np.pi/beta  #Matsubara frequencies
Delta = 1.0/(1j*Z-0.5) + 2.0/(1j*Z+0.2) + 0.5/(1j*Z+0.7) # Matsubara functions on these frequencies
```

### Hybridization Fitting
The hybridization fitting is handled by the `hybfit` function.
```python
from aaadapol import hybfit
```
There are two choices for doing hybridization fitting. One can either fit with desired accuracy tolerance `tol`:
```python
bath_energy, bath_hyb, final_error, func = hybfit(Delta, Z, tol=tol)
```
Or fit with specified number of interpolation points `Np`:
```python
bath_energy, bath_hyb, final_error, func = hybfit(Delta, Z, Np = Np)
```
Here `bath_energy` $E$ and `bath_hyb` $V$ are desired quantities of hybridization orbitals. They satisfy

```math
\Delta(\mathrm i \omega_k)_{mn} \approx \sum_l \frac{V_{lm} V_{ln}^*}{\mathrm i\omega_k - E_l}.
```

In more sophisticated applications, one might need to specify other flags, such as `maxiter`, `cleanflag` and `disp`. See the documentation for details.

One can look at the final error of the hybridization fitting:

```python
print(final_error)
```
### Triqs interface

For triqs users, if the Green's function data `delta_triqs` is stored as a `triqs.gf.Gf` object or `triqs.gf.BlockGf` object, then the hybridization fitting could be done using the `hybfit_triqs` function:
```python
from aaadapol import hybfit_triqs
bathhyb, bathenergy, delta_fit, final_error = hybfit_triqs(delta_triqs, tol=tol, debug=True)
```

### Analytic continuation

To use this code for analytic continuation is similar, and we refer to the documentation for details.
