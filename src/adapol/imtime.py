"""
Imagnary time routines

- L2-norm error between sum-of-simple-poles representations
- Residue optimization (least squares) minimizing the imaginary time L2-norm error
- Pole and residue optimization (nonlinear least squares) minimizing the imaginary time L2-norm error

Author: Hugo U. R. Strand (2026)
"""


import numpy as np


from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize as scipy_minimize


def kernel(tau, omega):
    """
    Compute the analytical continuation kernel (matrix) for given vectors of :math:`\\tau` and :math:`\\omega`.

    .. math::
        K(\\tau, \\omega) = -\\frac{e^{-\\tau \\omega}}{1 + e^{-\\omega}}

    This function calculates a kernel matrix used in exponential decay calculations.
    Positive and negative :math:`\\omega` values separately for numerical stability and accuracy.
    
    Parameters
    ----------
    tau : (n_tau,) array-like
        Imaginary time points.
    omega : (n_omega,) array-like
        Frequency vector.

    Returns
    -------
    kernel : (n_tau, n_omega) ndarray
        Kernel matrix :math:`K_{ij} = K(\\tau_i, \\omega_j)`.

    Examples
    --------
    >>> tau_i = np.array([0, 0.5, 1.0])
    >>> omega_j = np.array([-1, 0.5])
    >>> K_ij = kernel(tau_i, omega_j)
    >>> K_ij.shape
    (3, 2)

    Author: Hugo U. R. Strand (2021)
    https://github.com/jasonkaye/libdlr/blob/main/pydlr/kernel.py#L202
    """

    kernel = np.empty((len(tau), len(omega)))

    p, = np.where(omega > 0.)
    m, = np.where(omega <= 0.)
    w_p, w_m = omega[p].T, omega[m].T

    tau = tau[:, None]

    kernel[:, p] = -np.exp(-tau*w_p) / (1 + np.exp(-w_p))
    kernel[:, m] = -np.exp((1. - tau)*w_m) / (1 + np.exp(w_m))

    return kernel


def dyadic_panel_quadrature(n_per_panel, n_levels):
    """Build a composite Gauss-Legendre quadrature on [0, 1] with panels
    dyadically refined towards both endpoints 0 and 1.

    The panel structure is symmetric about the midpoint 1/2.  Starting from the
    left endpoint, the panels are [0, 2^{-n_levels}], [2^{-n_levels},
    2^{-(n_levels-1)}], ..., [1/4, 1/2], then mirrored for the right half.
    Each panel uses ``n_per_panel`` Gauss-Legendre nodes.

    Parameters
    ----------
    n_per_panel : int
        Number of Gauss-Legendre nodes per panel.
    n_levels : int
        Number of levels of dyadic refinement (must be >= 1).

    Returns
    -------
    nodes : ndarray, shape (N,)
        Quadrature nodes in (0, 1).
    weights : ndarray, shape (N,)
        Corresponding quadrature weights (positive, summing to 1).
    """

    if n_levels < 1:
        raise ValueError("n_levels must be >= 1")

    # Reference Gauss-Legendre nodes and weights on [-1, 1]
    x_ref, w_ref = leggauss(n_per_panel)

    # Build panel endpoints on [0, 1/2], dyadically refined towards 0:
    #   0, 2^{-n_levels}, 2^{-(n_levels-1)}, ..., 2^{-1} = 1/2
    breakpoints_left = [0.0] + [2.0**(-k) for k in range(n_levels, 0, -1)]

    nodes_list = []
    weights_list = []

    # Left-half panels: [0, 1/2]
    for i in range(len(breakpoints_left) - 1):
        a = breakpoints_left[i]
        b = breakpoints_left[i + 1]
        half_len = 0.5 * (b - a)
        mid = 0.5 * (a + b)
        nodes_list.append(mid + half_len * x_ref)
        weights_list.append(half_len * w_ref)

    # Right-half panels: mirror of [0, 1/2] about 1/2, i.e. [1/2, 1]
    for i in range(len(breakpoints_left) - 1):
        a = breakpoints_left[i]
        b = breakpoints_left[i + 1]
        # Mirror: [1-b, 1-a]
        a_r = 1.0 - b
        b_r = 1.0 - a
        half_len = 0.5 * (b_r - a_r)
        mid = 0.5 * (a_r + b_r)
        nodes_list.append(mid + half_len * x_ref)
        weights_list.append(half_len * w_ref)

    nodes = np.concatenate(nodes_list)
    weights = np.concatenate(weights_list)

    # Sort by node position
    order = np.argsort(nodes)
    return nodes[order], weights[order]


def exp_quadrature(lamb, n_per_panel=12):
    """Build a dyadic panel Gauss-Legendre quadrature on :math:`t \\in [0, 1]` 
    suitable for integrating sums of the imaginary time analytical continuation kernel.
    
    .. math:: 
        K(t, w) = \\frac{\\exp(-tw)}{1 + \\exp(-w)} 
        
    for :math:`|w| <= \\Lambda`.

    The number of refinement levels is chosen automatically from `lamb` (:math:`\\Lambda`).

    Parameters
    ----------
    lamb : float
        Maximum absolute (unit-less) frequency :math:`\\Lambda`.
        Controls the number of dyadic refinement levels.
    n_per_panel : int, optional
        Number of Gauss-Legendre nodes per panel (default 12).

    Returns
    -------
    nodes : ndarray
        Quadrature nodes in (0, 1).
    weights : ndarray
        Corresponding quadrature weights.
    """
    if lamb < 1.: lamb = 1.0 # Avoid corner cases, e.g. lamb == 0.
    n_levels = max(int(np.ceil(np.log(lamb) / np.log(2.0))) - 2, 1)
    return dyadic_panel_quadrature(n_per_panel, n_levels)


class ImTimeQuadrature:

    """ Imaginary time quadrature for representing spectral functions bound by lambda to (near) machine precision. """

    def __init__(self, lamb, beta):
        self.beta = beta
        self.lamb = lamb

        self.t_i, self.w_i = exp_quadrature(lamb)

        self.tau_i = self.t_i * beta
        self.sqrt_w_i = np.sqrt(self.w_i)


    def kernel_matrix(self, poles):
        """Compute the kernel matrix K(tau_i, poles_j) for the quadrature nodes and given poles."""
        K_ip = kernel(self.t_i, poles * self.beta)
        return K_ip
    

    def integrate(self, f_i):
        """Integrate a function f(t) sampled at the quadrature nodes t_i"""
        return self.beta * np.einsum('i,i...->...', self.w_i, f_i)


    def l2_norm(self, f):
        """ Compute the imaginary time L2 norm of an function in imaginary time
        using the quadrature. """
        return np.sqrt(self.integrate(np.abs(f(self.tau_i))**2))


    def best_l2_norm_approximation_using_poles(self, f, poles):
        """ Compute the best sum-of-poles approximation of the function :math:`f(\\tau)` 
        using given poles buy determining the residues :math:`R_p` that 
        minimizes the imaginary time L2 norm error. 
        
        This is done by solving a linear least squares problem, 
        where the kernel matrix is weighted by the quadrature weights. 
        
        .. math::
            \\sqrt{w_i} K(\\tau_i, z_p) R_p = \\sqrt{w_i} f(\\tau_i)
        
        """

        F_iX = f(self.tau_i)

        if F_iX.ndim > 1:
            shape_F_i = (len(self.tau_i), -1)
            shape_residues = [len(poles)] + list(F_iX.shape[1:])
        else:
            shape_F_i = (len(self.tau_i), 1)
            shape_residues = (len(poles),)

        F_i = F_iX.reshape(shape_F_i)

        K_ip = self.kernel_matrix(poles)

        A_ip = self.sqrt_w_i[:, None] * K_ip
        b_i = self.sqrt_w_i[:, None] * F_i

        residues, _, _, _ = np.linalg.lstsq(A_ip, b_i, rcond=None)
        residues = residues.reshape(shape_residues)

        return residues
    

    def best_l2_norm_approximation(self, sop, poles, verbose=False):

        func = lambda poles : self.l2_norm_gradient_with_respect_to_poles(sop, poles)

        res = scipy_minimize(
            func, poles, 
            method='L-BFGS-B', 
            jac=True,
            tol=1e-14)
        
        if verbose: 
            print(res)

        poles_opt = res.x

        residues_opt = self.best_l2_norm_approximation_using_poles(sop.imtime_function(self.beta), poles_opt)

        from .sop import SumOfSimplePoles
        sop_opt = SumOfSimplePoles(poles=poles_opt, residues=residues_opt)

        return sop_opt


    def l2_norm_gradient_with_respect_to_poles(self, sop, poles):
        """
        Compute the gradient of the imaginary time L2 norm error with respect to the poles, 
        for a given sum-of-simple-poles representation `sop` and a set of poles `poles` to optimize.

        Todo: write down the mathematical expression for the gradient in the docstring, 
        and verify it with numerical differentiation.

        """

        f_tau = sop.imtime_function(self.beta)
        weights = self.best_l2_norm_approximation_using_poles(f_tau, poles)
        from .sop import SumOfSimplePoles
        sop_approx = SumOfSimplePoles(poles=poles, residues=weights)

        # Compute error and gradient with respect to the poles

        sop_diff = sop - sop_approx

        M_ip = self.sqrt_w_i[:, None] * self.kernel_matrix(sop_diff.p)

        K_0mp = kernel(np.zeros(1), -sop_diff.p) # missing beta factor?

        M2_ip = M_ip * (self.tau_i[:, None] + K_0mp)

        df_i = M_ip @ sop_diff.R

        error = np.sqrt(self.beta) *np.linalg.norm(df_i, axis=0)
        grad = np.real(M2_ip.T @ df_i) * sop_diff.R.conj() / error[None, :] # contrived usage of transpose, cleanup?

        grad[np.isnan(grad)] = 0.0

        return np.sum(error), np.sum(grad, axis=1)[:len(poles)] * self.beta



