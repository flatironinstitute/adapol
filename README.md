## Introduction
[AAAdapol](https://github.com/Hertz4/AAAdapol) (pronounced "add a pole") is a python package for fitting Matsubara functions with the following form:
```math
G(\mathrm i \omega_k) = \sum_l \frac{V_{lm} V_{ln}^*}{\mathrm i\omega_k - E_l}.
```
AAAdapol is short for **A**ntoulas–**A**nderson **Ad**aptive **pol**e-fitting.

Current applications include

- hybridization fitting,

- analytic continuation.

## Installation
AAAdapol has the following prerequisites:
- numpy, scipy
- cvxpy, when conducting hybridization fitting for matrix-valued (instead of scalar-valued) Green's functions.



## Toy example
Let us illustrate how to use the code with the following toy example:
# Setup
```python
import numpy as np
beta = 20
Z = np.linspace(-25.,25.,26)*np.pi/beta  #Matsubara frequencies
Delta = 1.0/(1j*Z-0.5) + 2.0/(1j*Z+0.2) + 0.5/(1j*Z+0.7) # Matsubara functions on these frequencies
```

With `Delta` and `Z`, one first initialize the Matsubara object:
```python
Imfreq_obj = Matsubara(Delta, Z)
```

# Hybridization Fitting
There are two choices for doing hybridization fitting. One can either fit with desired accuracy `eps`:
```python
bath_energy, bath_hyb = Imfreq_obj.fitting(tol = 1e-6, flag = "hybfit")
```
Or fit with specified number of interpolation points `Np`:
```python
bath_energy, bath_hyb = Imfreq_obj.fitting(Np = 4, flag = "hybfit")
```
Here `bath_energy` $E$ and `bath_hyb` $V$ are desired quantities of hybridization orbitals. They satisfy

```math
\Delta(\mathrm i \omega_k)_{mn} \approx \sum_l \frac{V_{lm} V_{ln}^*}{\mathrm i\omega_k - E_l}.
```

In more sophisticated applications, one might need to specify other flags, such as `maxiter`, `cleanflag` and `disp`. See comments in `matsubara.py` for details.

One can look at the final error of the hybridization fitting:

```python
print(Imfreq_obj.final_error)
```

# Analytic continuation

Similarly, there are two choices for analytic continuation:

```python
greens_function = Imfreq_obj.fitting(tol = 1e-6, flag = "anacont")
```

or

```python
greens_function = Imfreq_obj.fitting(Np = 4, flag = "anacont")
```

The output now is a function evaluator for the Green's functions. For example, if one wish to evaluate the Green's function on `wgrid`, one can do:

```python
wgrid = np.linspace(-1,1,100)+1j*0.01
G_w = greens_function(wgrid)
```
and then one can plot the spectral function:
```python
import matplotlib.pyplot as plt
plt.plot(wgrid.real, -np.squeeze(G_w).imag/np.pi)
```
