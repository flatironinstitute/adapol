This is a python package for Matsubara functions in the imaginary frequency domain. Current applications include

(1) hybridization fitting,

(2) Analytic continuation.

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
There are two choices for doing hybridization fitting. One can either fit with desired accuracy 'eps':
```python
bath_energy, bath_hyb = Imfreq_obj.bath_fitting_tol(tol = 1e-6)
```
Or fit with specified number of interpolation points 'Np':
```python
bath_energy, bath_hyb = Imfreq_obj.bath_fitting_tol(Np = 4)
```
