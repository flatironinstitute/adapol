"""
The following function is used to support applications in xca (github.com/TRIQS/xca).
The main difference is here we evaluate the fitting error on the time domain.

todo: merge this function with the above pole_fitting function.
"""

import numpy as np
import scipy.linalg
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize as scipy_minimize

from .aaa import aaa_matrix_real


def kernel(tau, omega):
    kernel = np.empty((len(tau), len(omega)))

    p, = np.where(omega > 0.)
    m, = np.where(omega <= 0.)
    w_p, w_m = omega[p].T, omega[m].T

    tau = tau[:, None]

    kernel[:, p] = np.exp(-tau*w_p) / (1 + np.exp(-w_p))
    kernel[:, m] = np.exp((1. - tau)*w_m) / (1 + np.exp(w_m))

    return kernel


def get_weight_t(pol, tgrid, Deltat, beta, statistics="Fermion"):
    M = -kernel(tgrid/beta, pol*beta)
    shape_iaa = Deltat.shape
    shape_iA = (shape_iaa[0], shape_iaa[1]*shape_iaa[2])
    shape_xaa = (len(pol), shape_iaa[1], shape_iaa[2])
    weight = np.linalg.lstsq(M, Deltat.reshape(shape_iA), rcond=None)[0]
    residue = (Deltat.reshape(shape_iA) - M@weight).reshape(shape_iaa)
    
    weight = weight.reshape(shape_xaa)
    return weight, M, residue


def erroreval_t(pol,  tgrid, Deltat, beta, statistics="Fermion"):

    R, M, residue = get_weight_t(pol, tgrid, Deltat, beta, statistics=statistics)

    if len(Deltat.shape)==1:
        y = np.linalg.norm(residue)
        grad = np.real(np.dot(np.conj(residue) ,(R*(M**2))))
    else:
        y = np.linalg.norm(residue.flatten())

        Np = len(pol)
        grad = np.zeros(Np)
        Nw = len(tgrid)
        for k in range(Np):
            for w in range(Nw):
                grad[k] = grad[k] + np.real(np.sum((M[w,k]**2)*(np.conj(residue[w,:,:]) * R[k])))

    grad = -grad/y
    return y, grad


    

def kernel_L2_error_dlr(pol_combined):
    v1 = kernel(np.array([0.0]), pol_combined).flatten()
    v2 = kernel(np.array([0.0]), -pol_combined).flatten()

    K = np.zeros((len(pol_combined), len(pol_combined)))
    # K[i,j] = (1-exp(-(w1+w2))) * kernel(0,w1) * kernel(0,w2).
    # Different implementation for w1+w2>0 or smaller than 0.

    for i in range(len(pol_combined)):
        for j in range(len(pol_combined)):
            pol_sum = pol_combined[i] + pol_combined[j]
            # K[i,j] = (1 - np.exp(-np.abs(pol_sum) )) / np.abs(pol_sum) if np.abs(pol_sum) > 1e-15 else 1.0
            x = np.abs(pol_sum)
            K[i,j] = (-np.expm1(-x) / x ) if x > 1e-7 else 1 - x/2 + x*x/6 - x*x*x/24
            if pol_sum>0:
                K[i,j] *= v1[i] * v1[j]
            else:                
                K[i,j] *= v2[i] * v2[j]
    return K, v1, v2


def erroreval_dlr(pol, weights, w_dlr,Delta_dlr,beta, tau_nodes=None, tau_weights=None):
    # assuming the input dlr is using the kernel with the minus sign
    pol_combined = np.concatenate([pol * beta, w_dlr])
    weights_combined = np.concatenate([weights, Delta_dlr], axis=0)

    if tau_nodes is None or tau_weights is None:
        tau_nodes, tau_weights = exp_quadrature(max(np.max(np.abs(pol_combined)), 1.0))
    M = -kernel(tau_nodes, pol_combined) * tau_weights[:, None]
    M2 = M * (-tau_nodes[:, None]) + M * kernel(np.array([0.0]), -pol_combined)
    # breakpoint()
    weights_reshape = weights_combined.reshape((weights_combined.shape[0], weights_combined.shape[1]*weights_combined.shape[2]))
    residue = M@weights_reshape

    error =  np.linalg.norm(residue, axis=0) 

    grad = np.real((M2.T @ residue) * weights_reshape.conj()) / error[ None,:]
    grad[np.isnan(grad)] = 0.0

    return np.sum(error), np.sum(grad, axis=1)[0:len(pol)]
    



def get_weight_dlr(pol, w_dlr, Delta_dlr, beta, tau_nodes=None, tau_weights=None):
    if tau_nodes is None or tau_weights is None:
        tau_nodes, tau_weights = exp_quadrature(max(np.max(np.abs(pol)), 1.0))
    pol_combined = np.concatenate([pol * beta, w_dlr])
    M = -kernel(tau_nodes, pol_combined) * tau_weights[:, None]

    Delta_dlr_reshape = Delta_dlr.reshape((Delta_dlr.shape[0], Delta_dlr.shape[1]*Delta_dlr.shape[2]))
       
    weights_reshape = -scipy.linalg.lstsq(M[:, :len(pol)], M[:, len(pol):] @ Delta_dlr_reshape, cond=None)[0]
    weights = weights_reshape.reshape((len(pol), Delta_dlr.shape[1], Delta_dlr.shape[2]))

    return weights

def polefitting_dlr(Deltaiw, Z, Delta_dlr, w_dlr, beta, Np_max=50, eps=1e-5,  statistics="Fermion"):
    
    if statistics not in ["Fermion"]:
        raise Exception("Currently only Fermionic statistics is supported for this version of pole fitting. Consider use the algorithm in the frequency domain, which supports bosonic functions.")

    Num_of_nonzero_entries = np.sum(np.max(np.abs(Delta_dlr), axis=0) > 1e-12)
    error_best = np.inf
    weight_best = None
    pol_best = None
                
    for mmax in range(4,Np_max,2):
        
        pol, _, _, _ = aaa_matrix_real(Deltaiw, Z, mmax=mmax)
        pol = pol[np.abs(np.imag(pol))<1e-3]

        pol = np.real(pol)
        
        weight = get_weight_dlr(pol, w_dlr, Delta_dlr, beta)
        pol, weight = aaa_reduce(pol, weight,eps)
 
        tau_nodes, tau_weights = exp_quadrature(max(np.max(np.abs(np.concatenate([pol * beta, w_dlr]))), 1.0))
        error = erroreval_dlr(pol, weight, w_dlr, Delta_dlr, beta, tau_nodes=tau_nodes, tau_weights=tau_weights)[0]
 
        def fhere(pole):
            return erroreval_dlr(pole, get_weight_dlr(pole, w_dlr, Delta_dlr, beta, tau_nodes=tau_nodes, tau_weights=tau_weights), w_dlr, Delta_dlr, beta, tau_nodes=tau_nodes, tau_weights=tau_weights) 
        print("starting optimization with mmax =", mmax, "initial error =", error)
        if len(pol) > 0:
            res = scipy_minimize(
                fhere, pol, method='L-BFGS-B', jac=True,
                options=dict(disp=False, gtol=1e-14, ftol=1e-14))
            x = res.x
        else:
            x = pol
        
        weight  = get_weight_dlr(x, w_dlr, Delta_dlr, beta, tau_nodes=tau_nodes, tau_weights=tau_weights)
        error = erroreval_dlr(x, weight, w_dlr, Delta_dlr, beta, tau_nodes=tau_nodes, tau_weights=tau_weights)[0]

        if Num_of_nonzero_entries > 0:
            error /= Num_of_nonzero_entries

        if error < eps:
            print(f"Desired accuracy {eps} achieved with {len(x)} poles.")
            return weight, x, error
        elif error < error_best:
            error_best = error.copy()
            weight_best = weight.copy()
            pol_best = x.copy() 
    print("Failed to reach the desired accuracy", eps, "returning the best result found.")
    print(f"Best error achieved: {error_best} with {len(pol_best)} poles.")
        
    return weight_best, pol_best, error_best
        



def polefitting(Deltaiw, Z, Deltat,tgrid, Deltat_dense, tgrid_dense, beta,
                Np_max=50, eps=1e-5, Hermitian=True, statistics="Fermion"):
    
    if statistics not in ["Fermion"]:
        raise Exception("Currently only Fermionic statistics is supported for this version of pole fitting. Consider use the algorithm in the frequency domain, which supports bosonic functions.")

    Num_of_nonzero_entries = np.sum(np.max(np.abs(Deltat), axis=0) > 1e-12)
    error_best = np.inf
    weight_best = None
    pol_best = None
                
    for mmax in range(4,Np_max,2):
        pol, _, _, _ = aaa_matrix_real(Deltaiw, Z, mmax=mmax)
        pol = pol[np.abs(np.imag(pol))<1e-3]
        pol = np.real(pol)
        weight, _, residue = get_weight_t(pol, tgrid, Deltat,beta)
        pol, weight = aaa_reduce(pol, weight,eps)

        def fhere(pole):
            return erroreval_t(pole, tgrid, Deltat,beta, statistics=statistics) 

        if len(pol) > 0:
            res = scipy_minimize(
                fhere, pol, method='L-BFGS-B', jac=True,
                options=dict(disp=False, gtol=1e-14, ftol=1e-14))
            x = res.x
        else:
            x = pol
            
        weight, _, residue = get_weight_t(x, tgrid, Deltat,beta)
        M = -kernel(tgrid_dense/beta, x*beta)
        residue_dense = M@weight.reshape((weight.shape[0], weight.shape[1]*weight.shape[2])) - Deltat_dense.reshape((Deltat_dense.shape[0], Deltat_dense.shape[1]*Deltat_dense.shape[2]))
        error = np.linalg.norm(residue_dense.flatten()) / np.sqrt(len(tgrid_dense))

        if Num_of_nonzero_entries > 0:
            error /= Num_of_nonzero_entries

        if error < eps:
            print(f"Desired accuracy {eps} achieved with {len(x)} poles.")
            return weight, x, error
        elif error < error_best:
            
            error_best = error.copy()
            weight_best = weight.copy()
            pol_best = x.copy() 
    print("Failed to reach the desired accuracy", eps, "returning the best result found.")
    print(f"Best error achieved: {error_best} with {len(pol_best)} poles.")
    return weight_best, pol_best, error_best


# duplicated from fit_utils.py, consider merge them together later

def aaa_reduce(pol, R, eps=1e-6):
    Np = R.shape[0]
    Rnorm = np.zeros(Np)
    for i in range(Np):
        Rnorm[i] = np.linalg.norm(R[i])
    nonz_index = Rnorm > eps
    return pol[nonz_index], R[nonz_index]

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
