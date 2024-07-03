.. _theories:

Physical background
======================

Matsubara functions
--------------------------------------------
A Matsubara function :math:`G(\mathrm i\nu_k)` is a matrix-valued function of size :math:`N_{\text{orb}} \times N_{\text{orb}}`, 
where :math:`N_{\text{orb}}` is the number of orbitals, and :math:`\{\mathrm i\nu_k, k\in\mathbb Z\}` are the Matsubara frequencies:

.. math::
    
    \mathrm i\nu_k = \left\{
            \begin{aligned}
        \mathrm i \frac{(2n+1)\pi}{\beta}, & \text{ for fermions},\\
        \mathrm i \frac{2n\pi}{\beta}, & \text{ for bosons}.
            \end{aligned}
        \right.

Here :math:`\beta` is the inverse temperature. 

The Matsubara function admits the following Lehmann representation:

.. math::
    \begin{equation}
    G(\mathrm i\nu_k) = \int_{-\infty}^{\infty} \frac{1}{\mathrm i\nu_k - w} \rho(w)\mathrm dw, \tag{1}
    \label{Lehmann}
    \end{equation}

Here :math:`\rho(w)` is a matrix-valued function, and could either be sum of Dirac delta functions, or a continuous function.

The Matsubara function in the time domain, denoted as :math:`G(\tau)`, is related to :math:`G(\mathrm i\nu_k)` by the Fourier transform:

.. math::

        G(\mathrm i\nu_k) = \int_{0}^{\beta} G(\tau)\mathrm{e}^{\mathrm i\nu_k\tau} \mathrm d\tau.

In other words, :math:`G(\mathrm i\nu_k)` is the Fourier coefficient of :math:`G(\tau)`, and :math:`G(\tau)` satisfies the Kubo-Martin-Schwinger (KMS) (anti-)periodic condition:
:math:`G(\tau+\beta) = \pm G(\tau)`. 


Pole fitting
--------------------------------------------
It has been well known that any Matsubara functions :math:`G(\mathrm i\nu_k)` (i.e. functions defined by :math:`\eqref{Lehmann}`) could be well approximated by a finite sum of poles,

.. math::
    \begin{equation}
    G(\mathrm i\nu_k) \approx \sum_{p=1}^{N_p} \frac{M_p}{\mathrm i\nu_k - E_p}, \tag{2}
    \label{pole}
    \end{equation}

where :math:`N_p` is the number of poles, :math:`M_p` is a matrix-valued function, and :math:`E_p` are the poles.

Our goal is to obtain the poles :math:`E_p` and the weights :math:`M_p` (also known as the residues) numerically.
We hope to obtain such an approximation with :math:`N_p` as small as possible.

Hybridization fitting 
----------------------

The hybridization fitting is a crucial subroutine in dynamical mean-field theory calculations.
In  DMFT, the hybridization function :math:`\Delta(\mathrm i\nu_k)`,
which is obtained in each iteration of the DMFT self-consistent loop, is a matrix-valued Matsubara function of size :math:`N_{\text{orb}}\times N_{\text{orb}}`, where :math:`N_{\text{orb}}` is the number of impurity orbitals.
After obtaining :math:`\Delta(\mathrm i\nu_k)`, one conduct the following **hybridization fitting**:

.. math::

    \begin{equation}
    \Delta(\mathrm i\nu_k) = \sum_{j=1}^{N_b} \frac{V_jV_j^{\dagger}}{\mathrm i\nu_k - E_j}. \tag{3} \label{hybfit}
    \end{equation}

Here :math:`E_j\in\mathbb R` are bath energies, and :math:`V_j` are impurity-bath coupling coefficients, and are vectors of size :math:`(N_{\text{orb}},1)`.
Using :math:`E_p`  and the vectors :math:`V_p`, one can construct the new impurity-bath Hamiltonian :math:`\hat H`:

.. math::
    
    \hat H =\hat H_{\text{loc}} + \sum_{j=1}^{N_b}  E_j \hat c_j^{\dagger} \hat c_j + \sum_{j=1}^{N_b} \sum_{\alpha=1}^{N_{\mathrm{orb}}} (V_{j\alpha}\hat c_j^{\dagger}\hat c_{\alpha} + \text{h.c.}).

Here :math:`\hat H_{\text{loc}}` is the local impurity Hamiltonian.

Note that :math:`N_b` is the number of bath orbitals, and is also the number of terms that are used in the hybridization fitting :math:`\eqref{hybfit}`.


Analytical continuation
--------------------------------------------

The Matsubara function :math:`G(\mathrm i\nu_k)` (as defined in Eq. :math:`\eqref{Lehmann}`) could be extended to a function that is defined on the entire complex plane:

.. math::
    
        G(z) = \int_{-\infty}^{\infty} \frac{1}{z - w} \rho(w)\mathrm dw.

:math:`G(z)` is analytic in the upper half-plane, and in the lower half plane, but has a branch cut in the real axis.
In the special case that :math:`\rho(w)` is sum of Dirac delta functions, :math:`G(z)` is a rational function and is analytic in the entire complex plane except for the poles.

More importantly, :math:`\rho(w)` and :math:`G(z)` are related by the following formula:

.. math::
    
        \rho(w) = -\frac{1}{\pi} \lim_{\eta\to 0^+} \operatorname{Im} G(w+\mathrm i\eta).







Matsubara Green's function and spectral function
----------------------------------------------------

If :math:`G(\mathrm i\nu_k)` is a single-particle Green's function, (or similarly, any correlation functions), the Lehmann's representation states that it can be written as a sum of poles:

.. math::

    G_{ij}(\mathrm i\nu_k) =\frac{1}{Z} \sum_{r, s} \frac{\left\langle\Psi_s\left|\hat{c}_i\right| \Psi_r\right\rangle\left\langle\Psi_r\left|\hat{c}_j^{\dagger}\right| \Psi_s\right\rangle}{z+E_s-E_r}\left(\mathrm{e}^{-\beta E_s} \mp \mathrm{e}^{-\beta E_r}\right),

where :math:`\hat{c}_i, \hat{c}_i^{\dagger}` are the annihilation and creation operator for the :math:`i`-th orbital, :math:`\left|\Psi_s\right\rangle` is the :math:`s`-th eigenvector of the Hamiltonian :math:`\hat H` with energy :math:`E_s`.

By grouping the :math:`(r,s)` indices into a single index :math:`p`, we have a more compact form:

.. math::
    
        G(\mathrm i\nu_k) = \sum_{p=1}^{N_p} \frac{V_pV_p^{\dagger}}{\mathrm i\nu_k - E_p},
    
and in the thermodynamic limit, the sum becomes an integral:

.. math::
    
        G(\mathrm i\nu_k) = \int_{-\infty}^{\infty} \frac{1}{\mathrm i\nu_k - w} \rho(w),

Here :math:`\rho(w)` is positive semi-definite for all :math:`w`. In the scalar case (:math:`N_{\text{orb}}=1`), :math:`\rho(w)` is the *spectral function*. 
In the matrix case, the spectral function is:

.. math::
    
        {\mathrm{Spec}}(w) = \sum_{i=1}^{N_{\text{orb}}}\rho_{ii}(w).

Using the analytic continuation properties,
the spectral function could be evaluated by calculating the Green's function on the real axis with infinitesimal broadening:

.. math::
    
        \mathrm{Spec}(w) = -\frac{1}{\pi} \lim_{\eta\to 0^+}  \sum_{i=1}^{N_{\text{orb}}} \operatorname{Im} (G_{ii}(w+\mathrm i\eta)).
