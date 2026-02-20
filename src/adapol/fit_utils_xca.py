"""
The following function is used to support applications in xca (github.com/TRIQS/xca).
The main difference is here we evaluate the fitting error on the time domain.

todo: merge this function with the above pole_fitting function.
"""

import numpy as np
import scipy.linalg
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
            K[i,j] = (1 - np.exp(-np.abs(pol_sum) )) / np.abs(pol_sum) if np.abs(pol_sum) > 1e-15 else 1.0
            if pol_sum>0:
                K[i,j] *= v1[i] * v1[j]
            else:                
                K[i,j] *= v2[i] * v2[j]
    return K, v1, v2

def erroreval_dlr(pol, weights, w_dlr,Delta_dlr,beta ):
    # assuming the input dlr is using the kernel with the minus sign
    pol_combined = np.concatenate([pol * beta, w_dlr])
    weights_combined = np.concatenate([weights, Delta_dlr], axis=0)


    K, v1, v2 = kernel_L2_error_dlr(pol_combined)
    K11 = K[:len(pol), :len(pol)]
    K12 = K[:len(pol), len(pol):]

    Delta_dlr_reshape = Delta_dlr.reshape((Delta_dlr.shape[0], Delta_dlr.shape[1]*Delta_dlr.shape[2]))
    
    weights_reshape = np.linalg.lstsq(K11, K12 @ Delta_dlr_reshape, rcond=None)[0]
    weights = weights_reshape.reshape((len(pol), Delta_dlr.shape[1], Delta_dlr.shape[2]))

    pol_matrix = pol_combined[:, None] + pol_combined[None, :]
    grad1 = K * v2[:, None]
    grad2 = v2[:, None] * v2[None, :]/pol_matrix
    grad3 = - K / pol_matrix
    grad = beta * (grad1 + grad2 + grad3)
    
    
    error_mat = np.sqrt(np.einsum('iab,ij,jab->ab', weights_combined.conj(), K, weights_combined))
    gradient = np.zeros_like(pol, dtype=np.complex128)

    for l in range(len(pol_combined)):
        for l_prime in range(len(pol_combined)):
            if l < len(pol):
                gradient_l_mat = weights_combined[l].conj() * weights_combined[l_prime]*grad[l,l_prime] 
                gradient[l] += np.sum(gradient_l_mat /(2*error_mat))

            if l_prime < len(pol):
                gradient_l_prime_mat = weights_combined[l].conj() * weights_combined[l_prime]*grad[l_prime,l]
                gradient[l_prime] += np.sum(gradient_l_prime_mat /(2*error_mat))
   


    return np.sum(error_mat).real, gradient.real

def get_weight_dlr(pol, w_dlr, Delta_dlr, beta):
    pol_combined = np.concatenate([pol * beta, w_dlr])
    K, _, _ = kernel_L2_error_dlr(pol_combined)
    K11 = K[:len(pol), :len(pol)]
    K12 = K[:len(pol), len(pol):]

    Delta_dlr_reshape = Delta_dlr.reshape((Delta_dlr.shape[0], Delta_dlr.shape[1]*Delta_dlr.shape[2]))
    
    weights_reshape = -scipy.linalg.lstsq(K11, K12 @ Delta_dlr_reshape, cond=None)[0]
    weights = weights_reshape.reshape((len(pol), Delta_dlr.shape[1], Delta_dlr.shape[2]))
    return weights

def polefitting_dlr(Deltaiw, Z, Delta_dlr, w_dlr, beta, Np_max=50, eps=1e-5,  statistics="Fermion"):
    
    if statistics not in ["Fermion"]:
        raise Exception("Currently only Fermionic statistics is supported for this version of pole fitting. Consider use the algorithm in the frequency domain, which supports bosonic functions.")

    Num_of_nonzero_entries = np.sum(np.max(np.abs(Delta_dlr), axis=0) > 1e-12)
                
    for mmax in range(4,Np_max,2):
        
        pol, _, _, _ = aaa_matrix_real(Deltaiw, Z, mmax=mmax)
        pol = pol[np.abs(np.imag(pol))<1e-3]
        print(f"AAA poles: {pol}")
        pol = np.real(pol)
        weight = get_weight_dlr(pol, w_dlr, Delta_dlr, beta)
        pol, weight = aaa_reduce(pol, weight,eps)
        print(f"AAA poles: {len(pol)}")
        
        error = erroreval_dlr(pol, weight, w_dlr, Delta_dlr, beta)[0]
        print("AAA", error)
        def fhere(pole):
            return erroreval_dlr(pole, get_weight_dlr(pole, w_dlr, Delta_dlr, beta), w_dlr, Delta_dlr, beta) 

        if len(pol) > 0:
            res = scipy_minimize(
                fhere, pol, method='L-BFGS-B', jac=True,
                options=dict(disp=True, gtol=1e-14, ftol=1e-14))
            x = res.x
        else:
            x = pol
        
        weight  = get_weight_dlr(x, w_dlr, Delta_dlr, beta)
        error = erroreval_dlr(x, weight, w_dlr, Delta_dlr, beta)[0]
        print("L-BFGS", error)
        if Num_of_nonzero_entries > 0:
            error /= Num_of_nonzero_entries

        if error < eps:
            return weight, x, error
        
    return weight, x, error


def polefitting(Deltaiw, Z, Deltat,tgrid, Deltat_dense, tgrid_dense, beta,
                Np_max=50, eps=1e-5, Hermitian=True, statistics="Fermion"):
    
    if statistics not in ["Fermion"]:
        raise Exception("Currently only Fermionic statistics is supported for this version of pole fitting. Consider use the algorithm in the frequency domain, which supports bosonic functions.")

    Num_of_nonzero_entries = np.sum(np.max(np.abs(Deltat), axis=0) > 1e-12)
                
    for mmax in range(4,Np_max,2):
        pol, _, _, _ = aaa_matrix_real(Deltaiw, Z, mmax=mmax)
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
            return weight, x, error
        
    return weight, x, np.linalg.norm(residue)


# duplicated from fit_utils.py, consider merge them together later

def aaa_reduce(pol, R, eps=1e-6):
    Np = R.shape[0]
    Rnorm = np.zeros(Np)
    for i in range(Np):
        Rnorm[i] = np.linalg.norm(R[i])
    nonz_index = Rnorm > eps
    return pol[nonz_index], R[nonz_index]
