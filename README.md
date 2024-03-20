This is a python package for Matsubara functions in the imaginary frequency domain. 

Current applications include

- hybridization fitting,

- Analytic continuation.

Let us illustrate how to use the code with the following toy example:
## Setup
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

## Hybridization Fitting
There are two choices for doing hybridization fitting. One can either fit with desired accuracy `eps`:
```python
bath_energy, bath_hyb = Imfreq_obj.bathfitting_tol(tol = 1e-6)
```
Or fit with specified number of interpolation points `Np`:
```python
bath_energy, bath_hyb = Imfreq_obj.bathfitting_num_poles(Np = 4)
```
Here `bath_energy` and `bath_hyb` are desired quantities of hybridization orbitals. 

In more sophisticated applications, one might need to specify other flags, such as `maxiter`, `cleanflag` and `disp`. See comments in `matsubara.py` for details.

## Analytic continuation

Similarly, there are two choices for analytic continuation:

```python
greens_function = Imfreq_obj.analytic_cont_tol(tol = 1e-6)
```

or

```python
greens_function = Imfreq_obj.analytic_cont_num_poles(Np = 4)
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
