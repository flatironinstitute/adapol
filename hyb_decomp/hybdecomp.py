import sys
sys.path.insert(0, "../")
import numpy as np
from aaa import *
from ac_pes import *

import matplotlib.pyplot as plt
import scipy

def eval_with_pole(pol, Z, weight):
    pol_t = np.reshape(pol,[pol.size,1])
    M = 1/(Z-pol_t)
    M = M.transpose()
    if len(weight.shape)==1:
        return M@weight
    else:
        G = M@np.reshape(weight, (weight.shape[0], weight.shape[1]*weight.shape[2]))
        return np.reshape(G, (G.shape[0],  weight.shape[1], weight.shape[2]))

def obtain_orbitals(pol,weight,eps=1e-7):
    eiglist = np.array([])
    polelist = []
    veclist = []
    matlist = []
    print(weight.shape[0])
    for i in range(weight.shape[0]):
        eigval, eigvec = np.linalg.eig(weight[i])
        for j in range(eigval.shape[0]):
            if eigval[j]>eps:
                polelist.append(pol[i])
                veclist.append(eigvec[:,j]*np.sqrt(eigval[j]))
                matlist.append((eigvec[:,j,None]*np.conjugate(eigvec[:,j].T))*(eigval[j]))
    polelist = np.array(polelist)
    veclist = np.array(veclist)
    matlist = np.array(matlist)
    print((veclist.shape))
    return polelist, veclist, matlist

def hybridization_fitting(Delta, Z, mmax = 6, maxiter = 50):
    #pole estimation
    r0 = aaa_matrix_real(Delta, 1j*Z,mmax=mmax)
    pol = np.real(r0.pol())
    weight, _, residue = get_weight(pol, 1.0j*Z, Delta,cleanflag=True)
    pol, weight = aaa_reduce(pol, weight, 1e-5)

    fhere = lambda pole: erroreval(pole,1j*Z,Delta,cleanflag=True)
    res = scipy.optimize.minimize(fhere,pol, method='L-BFGS-B', jac=True,options= {"disp" : False,"maxiter":4000, "gtol":1e-15,"ftol":1e-15})
    fhere = lambda pole: erroreval(pole,1j*Z,Delta,cleanflag=False)
    res = scipy.optimize.minimize(fhere,res.x, method='L-BFGS-B', jac=True,options= {"disp" : True,"maxiter":maxiter, "gtol":1e-20,"ftol":1e-20})
    weight, _, residue = get_weight(res.x, 1j*Z, Delta,cleanflag=False)
    err = np.max(np.abs(residue))

    
    return res.x, weight


if __name__ == "__main__":
    file = open('hyb_data/omega_20231129.txt', "r")
    Z = np.loadtxt(file,dtype=np.complex128)
    file.close()
    Zid = abs(Z)<200

    import h5py
    with h5py.File("hyb_data/Delta.0.h5", "r") as f:
        print(list(f.keys()))
        Delta = f[list(f.keys())[-1]]['Hyb'][:]

    Z = Z[Zid]

    mmax = 6
    for i in range(Delta.shape[1]):
        Delta0 = Delta[Zid,i,:,:]
        pol, weight = hybridization_fitting(Delta0, Z, mmax = 6, maxiter = 100)
        polelist, veclist, matlist = obtain_orbitals(pol,weight,eps=1e-7)
        # This is the result that we want: polelist, veclist
