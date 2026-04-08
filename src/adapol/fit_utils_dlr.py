"""
This module provides utility functions for fitting poles to DLR representations.

One example applications in xca (github.com/TRIQS/xca).
The main difference is here we evaluate the fitting error using the DLR coefficients.
"""

from math import floor

import numpy as np
import scipy.linalg
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize as scipy_minimize

from .aaa import aaa_matrix_real

## TODO: change dlr name to pol_in 

## TODO: hybfit_matsubara, hybfit_pole

def kernel(tau, omega):
    """
    Compute the kernel matrix for a given tau and omega parameters.
    This function calculates a kernel matrix used in exponential decay calculations, handling positive and negative omega values separately for numerical stability and accuracy.
    Parameters
    ----------
    tau : array-like
        Time points or decay parameters, shape (n_tau,).
    omega : array-like
        Frequency or rate parameters, shape (n_omega,).
    Returns
    -------
    kernel : ndarray
        Kernel matrix of shape (n_tau, n_omega) where:
        - For omega > 0: kernel[:, i] = exp(-tau * omega[i]) / (1 + exp(-omega[i]))
        - For omega <= 0: kernel[:, i] = exp((1 - tau) * omega[i]) / (1 + exp(omega[i]))
    Notes
    -----
    The kernel function is different from the kernel in standard DLR notation with a minus sign. 
    Examples
    --------
    >>> tau = np.array([0, 0.5, 1.0])
    >>> omega = np.array([-1, 0.5])
    >>> K = kernel(tau, omega)
    >>> K.shape
    (3, 2)
    """

    kernel = np.empty((len(tau), len(omega)))

    p, = np.where(omega > 0.)
    m, = np.where(omega <= 0.)
    w_p, w_m = omega[p].T, omega[m].T

    tau = tau[:, None]

    kernel[:, p] = np.exp(-tau*w_p) / (1 + np.exp(-w_p))
    kernel[:, m] = np.exp((1. - tau)*w_m) / (1 + np.exp(w_m))

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


def exp_quadrature(omega_max, n_per_panel=12):
    """Build a dyadic panel Gauss-Legendre quadrature on [0, 1] suitable for
    integrating sums of the kernel K(tau, omega) = exp(-tau*omega) /
    (1 + exp(-omega)) for |omega| <= omega_max.

    The number of refinement levels is chosen automatically from omega_max.

    Parameters
    ----------
    omega_max : float
        Maximum absolute frequency.  Controls the number of dyadic refinement
        levels.
    n_per_panel : int, optional
        Number of Gauss-Legendre nodes per panel (default 12).

    Returns
    -------
    nodes : ndarray
        Quadrature nodes in (0, 1).
    weights : ndarray
        Corresponding quadrature weights.
    """
    n_levels = max(int(np.ceil(np.log(omega_max) / np.log(2.0))) - 2, 1)
    return dyadic_panel_quadrature(n_per_panel, n_levels)

def eval_tau_with_pole_rep(pol, weights, tau_nodes):
    """
    Evaluate the pole representation at given tau nodes.

    Parameters
    ----------
    pol : array-like, shape (n_poles,)
        Array of poles. These are without the beta factor.
    weights : array-like, shape (n_poles, N_orb, N_orb)
        Array of weights corresponding to the poles. These weights are for the kernel without the minus sign, i.e., K(tau, pol) = exp(-tau*pol) / (1 + exp(-pol)).
    tau_nodes : array-like, shape (n_tau,)
        Time points at which to evaluate the pole representation.

    Returns
    -------
    Deltat : ndarray, shape (n_tau, N_orb, N_orb)
        The evaluated pole representation at the given tau nodes.
    """
    K_pol = kernel(tau_nodes, pol )  # Shape: (n_tau, n_poles)
    Deltat = np.einsum('ti,iab->tab', K_pol, weights)  # Shape: (n_tau, N_orb, N_orb)
    return Deltat

def erroreval_dlr(pol, w_dlr, Delta_dlr, beta, weights=None, tau_nodes=None, tau_weights=None):
    """
    Evaluate the fitting error for a given set of poles and weights, comparing with the DLR representation. Also output the gradient with respect to the poles, which can be used for optimization.
    The error is computed in the time domain using a kernel function and dyadic quadrature nodes/weights.

    Parameters
    ----------
    pol : array-like, shape (n_poles,)
        Array of poles. These are without the beta factor.
    
    w_dlr : array-like, shape (n_dlr,)
        Array of DLR frequencies. These are with the beta factor included.
    Delta_dlr : array-like, shape (n_dlr, N_orb, N_orb)
        Array of DLR coefficients corresponding to the DLR frequencies. These Delta_dlr coefficients are for the kernel with the minus sign, i.e., K(tau, w_dlr) = - exp(-tau*w_dlr) / (1 + exp(-w_dlr)), which is automatically the output of DLR decomposition.
    beta : float
        Inverse temperature parameter used to scale the poles.

    weights : array-like, shape (n_poles, N_orb, N_orb), optional
        If None, it will be computed using least squares fitting to the DLR representation.
        Array of weights corresponding to the poles. These weights are for the kernel without the minus sign, i.e., K(tau, pol) = exp(-tau*pol) / (1 + exp(-pol)).
    tau_nodes : array-like, shape (n_tau,), optional
        Quadrature nodes in the time domain. If None, they will be generated automatically based on the maximum absolute value of the poles and DLR frequencies.
    tau_weights : array-like, shape (n_tau,), optional
        Quadrature weights corresponding to the tau_nodes. If None, they will be generated automatically based on the maximum absolute value of the poles and DLR frequencies.

    Returns
    -------
    error : float
        The computed fitting error, which is the norm of the residue in the time domain.
    grad : array-like, shape (n_poles,)
        The gradient of the error with respect to the poles, which can be used for optimization.
    """
    pol_combined = np.concatenate([pol * beta, w_dlr])
    # construct dyadic quadrature nodes and weights if not provided
    if tau_nodes is None or tau_weights is None:
        tau_nodes, tau_weights = exp_quadrature(max(2 * np.max(np.abs(pol_combined)), 1.0))

    
    if weights is None:
        # compute the weights using least squares fitting to the DLR representation
        weights, M = get_weight_dlr(pol, w_dlr, Delta_dlr, beta, tau_nodes=tau_nodes, tau_weights=tau_weights)
    else:
        M = -kernel(tau_nodes, pol_combined) * np.sqrt(tau_weights)[:, None]
    # construct the kernel matrix's derivative with respect to the poles
    M2 = M * (-tau_nodes[:, None]) + M * kernel(np.array([0.0]), -pol_combined)

    # The input weights are for the kernel without the minus sign, while the DLR coefficients are for the kernel with the minus sign. Thus when combining them together, there is no need to change the sign of the weights.
    weights_combined = np.concatenate([weights, Delta_dlr], axis=0)


    # reshape the weights and compute the residue and error in the time domain, as well as the gradient with respect to the poles
    weights_reshape = weights_combined.reshape((weights_combined.shape[0], weights_combined.shape[1]*weights_combined.shape[2]))
    residue = M@weights_reshape

    
    error = np.linalg.norm(residue, axis=0)  
    grad = np.real((M2.T @ residue) * weights_reshape.conj()) / error[ None,:]
    # error =  np.linalg.norm(residue.flatten()) 
    # grad =  np.real((M2.T @ residue) * weights_reshape.conj())  / error
    grad[np.isnan(grad)] = 0.0
    
    return np.sum(error), np.sum(grad, axis=1)[0:len(pol)] * beta

    



def get_weight_dlr(pol, w_dlr, Delta_dlr, beta, tau_nodes=None, tau_weights=None):
    """
    Compute the weights for the poles by fitting to the DLR representation using least squares. 

    Parameters
    ----------
    pol : array-like, shape (n_poles,)
        Array of poles. These are without the beta factor.
    w_dlr : array-like, shape (n_dlr,)
        Array of DLR frequencies. These are with the beta factor included.
    Delta_dlr : array-like, shape (n_dlr, N_orb, N_orb)
        Array of DLR coefficients corresponding to the DLR frequencies. These Delta_dlr coefficients are for the kernel with the minus sign, i.e., K(tau, w_dlr) = - exp(-tau*w_dlr) / (1 + exp(-w_dlr)), which is automatically the output of DLR decomposition.
    beta : float
        Inverse temperature parameter used to scale the poles.
    tau_nodes : array-like, shape (n_tau,), optional
        Quadrature nodes in the time domain. If None, they will be generated automatically based on the maximum absolute value of the poles and DLR frequencies.
    tau_weights : array-like, shape (n_tau,), optional
        Quadrature weights corresponding to the tau_nodes. If None, they will be generated automatically based on the maximum absolute value of the poles and DLR frequencies.

    Returns
    -------
    weights : array-like, shape (n_poles, N_orb, N_orb)
        Array of weights corresponding to the poles, computed by least squares fitting to the DLR representation. These weights are for the kernel without the minus sign, i.e., K(tau, pol) = exp(-tau*pol) / (1 + exp(-pol)).
    M : array-like, shape (n_tau, n_poles + n_dlr)
        The combined kernel matrix for the poles and DLR frequencies, which can be used for error evaluation and gradient computation in the time domain.
    """
    pol_combined = np.concatenate([pol * beta, w_dlr])
    if tau_nodes is None or tau_weights is None:
        tau_nodes, tau_weights = exp_quadrature(max(2 * np.max(np.abs(pol_combined)), 1.0))
    
    M = -kernel(tau_nodes, pol_combined) * np.sqrt(tau_weights)[:, None]

    Delta_dlr_reshape = Delta_dlr.reshape((Delta_dlr.shape[0], Delta_dlr.shape[1]*Delta_dlr.shape[2]))
       
    weights_reshape = -scipy.linalg.lstsq(M[:, :len(pol)], M[:, len(pol):] @ Delta_dlr_reshape, cond=None)[0]
    weights = weights_reshape.reshape((len(pol), Delta_dlr.shape[1], Delta_dlr.shape[2]))

    return weights, M

def polefitting_dlr( Delta_dlr, w_dlr, beta, eps=1e-5, Nw=None,  Np_max=50, Z = None,  statistics="Fermion", verbose=False):
    """
    Perform pole fitting with a given initial pole representation.

    Parameters
    ----------
    Delta_dlr : array-like, shape (n_dlr, N_orb, N_orb)
        Array of DLR coefficients corresponding to the DLR frequencies. These Delta_dlr coefficients are for the kernel with the minus sign, i.e., K(tau, w_dlr) = - exp(-tau*w_dlr) / (1 + exp(-w_dlr)), which is automatically the output of DLR decomposition.
    w_dlr : array-like, shape (n_dlr,)
        Array of DLR frequencies. These are with the beta factor included.
    beta : float    Inverse temperature parameter used to scale the poles.      
    Nw : int, optional
        Number of Matsubara frequencies to use when constructing the frequency grid for fitting. If None, it will be automatically determined based on beta.
    Np_max : int, optional
        Maximum number of poles to consider in the fitting process (default 50).
    eps : float, optional
        Targeted accuracy for the fitting process.
    Z : array-like, shape (n_freq,), optional
        Custom Matsubara frequency grid to use for fitting. If None, it will be automatically generated based on Nw and beta.
    statistics : "Fermion" or "Boson", optional
        Specify the statistics of the system, which determines the form of the kernel and the Matsubara frequency grid. Currently only "Fermion" is supported for this version of pole fitting.
    verbose : bool, optional
        If True, print detailed information about the fitting process, including warnings about spurious poles and optimization results.
    """
 
    if statistics not in ["Fermion"]:
        raise Exception("Currently only Fermionic statistics is supported for this version of pole fitting. Consider use the algorithm in the frequency domain, which supports bosonic functions.")
    if Z is None:
        if Nw is None:
            Nw = max(1000, np.ceil(np.max(np.abs(w_dlr))/np.pi))
            if verbose: print(f"Adapol: Using {Nw} equidistant Matsubara frequencies")
        else:
            if verbose: print(f"Adapol: Using user-provided Nw = {Nw} to construct the Matsubara frequency grid." )
        Z = np.arange(-2*Nw-1, 2*Nw+2, 2) * np.pi / beta * 1j
 
    Deltaiw = np.einsum('ij,jab->iab', 1/(Z[:, None] - w_dlr), Delta_dlr)
    Num_of_nonzero_entries = np.sum(np.max(np.abs(Delta_dlr), axis=0) > 1e-12)
    error_best = np.inf
    weight_best = None
    pol_best = None
    # Np_max = min(Np_max, len(w_dlr)+1)

    for mmax in range(4,Np_max,2):
        
        pol, _, _, _ = aaa_matrix_real(Deltaiw, Z, mmax=mmax)
 
        # discard poles with large imaginary part, which are likely to be spurious poles from the AAA algorithm
        #TODO: print a warning here. 
        #TODO: add comment that this is heuristic,
        if len(pol[np.abs(np.imag(pol))> min(1000*eps, 1e-3)]) > 0:
            if verbose:
                Np = len(pol)
                Np_imag = len(pol[np.abs(np.imag(pol))> min(1000*eps, 1e-3)])
                p_eps = min(1000*eps, 1e-3)
                print(f"Adapol: Warning! AAA with {Np} poles w_i, found {Np_imag} poles with Im[w_i] > {p_eps:2.2E}")
            # pol = pol[np.abs(np.imag(pol))< 1e-3]
    
        pol = np.real(pol)
        pol = merge_degenerate_poles(pol, verbose=verbose)
 
        
        weight = get_weight_dlr(pol, w_dlr, Delta_dlr, beta)[0]
 
        tau_nodes, tau_weights = exp_quadrature(max(2 * np.max(np.abs(np.concatenate([pol * beta, w_dlr]))), 1.0))
        
 
        def fhere(pole):
            return erroreval_dlr(pole, w_dlr, Delta_dlr, beta, tau_nodes=tau_nodes, tau_weights=tau_weights) 
        if verbose:
            error = erroreval_dlr(pol, w_dlr, Delta_dlr, beta, weights=weight, tau_nodes=tau_nodes, tau_weights=tau_weights)[0]
            error_pre_opt = error / Num_of_nonzero_entries
        if len(pol) > 0:
            res = scipy_minimize(
                fhere, pol, method='L-BFGS-B', jac=True,
                options=dict(gtol=1e-14, ftol=1e-14))
            x = res.x
            if verbose:
                error_post_opt = res.fun / Num_of_nonzero_entries
                print(f"Adapol: Weight optimization, errors pre {error_pre_opt:+2.2E} post {error_post_opt:+2.2E}" + \
                      f" ({len(pol)} poles)")
        else:
            x = pol
        
        weight  = get_weight_dlr(x, w_dlr, Delta_dlr, beta, tau_nodes=tau_nodes, tau_weights=tau_weights)[0]
        error = erroreval_dlr(x, w_dlr, Delta_dlr, beta, weights = weight, tau_nodes=tau_nodes, tau_weights=tau_weights)[0]
 

        if Num_of_nonzero_entries > 0:
            error /= Num_of_nonzero_entries

        if error < eps and len(x) <= len(w_dlr):
            if verbose: print(f"Adapol: Desired accuracy {eps:2.2E} reached (error {error:2.2E} with {len(x)} poles)")
            #, in comparison to {len(w_dlr)} original pole representation. Returning the result.")
            return weight, x, error
        elif error < error_best and len(x) <= len(w_dlr):
            error_best = error.copy()
            weight_best = weight.copy()
            pol_best = x.copy() 

    if verbose: print(f"Adapol: Warning! Fit error {error_best:2.2E} larger than tolerance {eps:2.2E}.")
        
    return weight_best, pol_best, error_best
        






def merge_degenerate_poles(pol, rtol=1e-6, verbose=False):
    """Merge near-degenerate poles from AAA into single poles.

    The AAA algorithm (via find_pol) can produce exactly degenerate poles
    from its generalized eigenvalue problem. When two poles coincide, the
    downstream least-squares weight fitting becomes catastrophically
    ill-conditioned (two identical columns in the kernel matrix), producing
    weight matrices with ~1e8 norm that cause blow-up in diagram evaluation.
    
    Parameters
    ----------
    pol : ndarray
        Pole positions (energy units).
    rtol : float
        Relative tolerance for merging. Poles are merged when
        |pol_i - pol_j| < rtol * (max(|pol|) - min(|pol|) + 1).

    Returns
    -------
    pol_merged : ndarray
        Poles with degenerate groups replaced by their mean.
    """
    if len(pol) <= 1:
        return pol

    scale = np.ptp(np.abs(pol))  # range of |pol|
    atol = rtol * (scale + 1.0)  # +1 avoids zero scale

    order = np.argsort(pol)
    pol_sorted = pol[order]

    merged = []
    i = 0
    while i < len(pol_sorted):
        group = [pol_sorted[i]]
        while i + 1 < len(pol_sorted) and abs(pol_sorted[i + 1] - group[0]) < atol:
            i += 1
            group.append(pol_sorted[i])
        merged.append(np.mean(group))
        i += 1
    if len(merged) < len(pol) and verbose:
        print(f"Adapol: Merging {len(pol) - len(merged)} poles of {len(pol)} into {len(merged)} poles.")

    return np.array(merged)


def polefitting_dlr_triqs(
    Delta_triqs,
    eps=1e-5,
    Nw=None,
    Np_max=50,
    Z=None,
    statistics="Fermion",
    verbose=False,
):
    r"""
    The triqs interface for DLR pole fitting.
    The function requires triqs package in python.

    Accepts a TRIQS Green's function container with a DLR-related mesh
    (MeshDLR, MeshDLRImFreq, or MeshDLRImTime). For MeshDLRImFreq and
    MeshDLRImTime inputs, the function converts to MeshDLR internally.

    Examples:
    ----------

        -  Fitting with default tolerance:
            :code:`polefitting_dlr_triqs(delta_triqs)`

        - Fitting with custom tolerance:
            :code:`polefitting_dlr_triqs(delta_triqs, eps=1e-6)`

    Parameters:
    ------------
    :code:`Delta_triqs`: triqs Green's function container
        The input function in DLR representation.
        Accepted mesh types: MeshDLR, MeshDLRImFreq, MeshDLRImTime.

    :code:`eps`, :code:`Nw`, :code:`Np_max`, :code:`Z`, :code:`statistics`, :code:`verbose`:
        same as in polefitting_dlr

    Returns:
    ---------

    :code:`weight`: np.array :math:`(N_p, N_{\mathrm{orb}}, N_{\mathrm{orb}})`
        Weight matrices for the poles.

    :code:`pol`: np.array :math:`(N_p,)`
        Optimized pole positions.

    :code:`error`: float
        Final fitting error.

    If input is BlockGf, returns lists of (weight, pol, error) for each block.
    """
    try:
        from triqs.gf import Gf, BlockGf, MeshDLR, MeshDLRImFreq, MeshDLRImTime, make_gf_dlr
    except ImportError:
        raise ImportError("Failed to import the triqs package (https://triqs.github.io/triqs/latest/). "
                          "Please ensure it is installed.")

    if isinstance(Delta_triqs, Gf):
        if isinstance(Delta_triqs.mesh, (MeshDLRImFreq, MeshDLRImTime)):
            Delta_triqs = make_gf_dlr(Delta_triqs)

        if not isinstance(Delta_triqs.mesh, MeshDLR):
            raise RuntimeError("Error: Delta_triqs.mesh must be an instance of MeshDLR, MeshDLRImFreq, or MeshDLRImTime.")

        Delta_dlr = Delta_triqs.data
        if len(Delta_dlr.shape) == 1:
            Delta_dlr = Delta_dlr[:, None, None] # Handle scalar valued Triqs Green's functions by reshaping them to have shape (n_dlr, 1, 1) for compatibility with the fitting functions.

        w_dlr = np.array(list(Delta_triqs.mesh.values()))
        beta = Delta_triqs.mesh.beta

        weight, pol, error = polefitting_dlr(
            Delta_dlr, w_dlr, beta, eps=eps, Nw=Nw, Np_max=Np_max,
            Z=Z, statistics=statistics, verbose=verbose
        )

        return weight, pol, error

    elif isinstance(Delta_triqs, BlockGf) and isinstance(Delta_triqs.mesh, (MeshDLR, MeshDLRImFreq, MeshDLRImTime)):
        weight_list, pol_list, error_list = [], [], []
        for block, delta_blk in Delta_triqs:
            weight, pol, error = polefitting_dlr_triqs(
                delta_blk, eps=eps, Nw=Nw, Np_max=Np_max,
                Z=Z, statistics=statistics, verbose=verbose
            )
            weight_list.append(weight)
            pol_list.append(pol)
            error_list.append(error)

        return weight_list, pol_list, error_list

    else:
        raise RuntimeError("Error: Delta_triqs must be a Gf or BlockGf with MeshDLR, MeshDLRImFreq, or MeshDLRImTime mesh.")
