This is a python package for Matsubara functions in the imaginary frequency domain. Current applications include

(1) hybridization fitting,

(2) Analytic continuation.

Let us illustrate how to use the code with the following toy example:

```python
import numpy as np
beta = 20
Z = np.linspace(-25.,25.,26)*np.pi/beta  #Matsubara frequencies
Delta = 1.0/(1j*Z-0.5) + 2.0/(1j*Z+0.2) + 0.5/(1j*Z+0.7)
```

<h3>Hybridization Fitting</h3>

In hybridization fitting, 
```python
bath = Matsubara(Delta, Z)
```
