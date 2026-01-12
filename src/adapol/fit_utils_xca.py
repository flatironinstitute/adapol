"""
The following function is used to support applications in xca (github.com/TRIQS/xca).
The main difference is here we evaluate the fitting error on the time domain.

todo: merge this function with the above pole_fitting function.
"""

import numpy as np

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


def polefitting(Deltaiw, Z, Deltat,tgrid, Deltat_dense, tgrid_dense, beta,
                Np_max=50, eps=1e-5, Hermitian=True, statistics="Fermion"):
    
    if statistics not in ["Fermion"]:
        raise Exception("Currently only Fermionic statistics is supported for this version of pole fitting. Consider use the algorithm in the frequency domain, which supports bosonic functions.")

    Num_of_nonzero_entries = 0
    for i in range(Deltaiw.shape[1]):
        for j in range(Deltaiw.shape[2]):
            if np.max(np.abs((Deltat[:,i,j])))>1e-12:
                Num_of_nonzero_entries += 1
    
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
        error =error/Num_of_nonzero_entries

        if error<eps:
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
