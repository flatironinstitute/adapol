import sys
# sys.path.insert(0, "../")
import numpy as np
from aaa import *
from ac_pes import *

import matplotlib.pyplot as plt
import scipy
class bath(object):
    def __init__(self, Delta, Z):
        self.Delta = Delta
        self.Z = Z
    def hyb_fit(self, tol = None,Np = None, mmin=None,mmax = 50, maxiter = 50,cleanflag=False,fast=False, disp=False, complex = True,eps = 1e-7):
        """Conduct bath fitting, either with fixed error tolerance tol, or with fixed number of poles Np

        Examples:
        --------
        Recommended choice: fit with number of poles Np
            bath.hyb_fit(Np = Np, maxiter = 500, cleanflag = True) # Replacing SDR with least square to boost calculation speed
        Other choices: 
            bath.hyb_fit(Np = Np, maxiter = 50)   # do this if have enough time to kill. Will be more accurate than above but take much more time
        Otger choices:
            bath.hyb_fit(tol = 1e-6, maxiter = 50) # Fit to tolerance 1e-6

        

        Parameters
        ----------
        tol: Fitting error tolreance, float
            default: None

        Np: number of poles, integer
            default: None

        # The user needs to specify either tol or eps. Specifying both or None will return to error.
        # If specifying tol, the number of poles are increased until reaching fitting error < eps.
        # If specifying Np, the fit is done with Np poles.

        mmin, mmax: number of minimum or maximum poles.
            If Np is specified, then mmin, mmax will be set to be Np, Np+1
            If tol is specified, then mmin default = 4, mmax default = 50

        maxiter: int
            maximum number of iterations
            default: 50

        disp: bool
            whether to display optimization details
            default: False

        complex: bool
            whether to allow complex-valued weight matrices (or only real-valued matrices)
            default: True
        
        cleanflag: bool
            whether to use least square to replace semidefinite programming (SDP) to fasten calculation
            default: False
        
        fast: bool
            when cleanflag=False, whether to only enforce semidefinite-ness on 2-2 subblocks
            default: False


        eps: float, optional
            Truncation threshold for bath orbitals while doing SVD of weight matrices
            default:1e-7
        
        """

        pol, weight, fitting_error = pole_fitting(self.Delta, self.Z, tol = tol,Np = Np, mmin=mmin,mmax = mmax, maxiter = maxiter,cleanflag=cleanflag,fast=False, disp=disp, complex = complex) 
        self.pol, self.weight, self.fitting_error = pol, weight, fitting_error
    
        self.obtain_orbitals(eps=eps) 
        self.final_error = np.max(np.abs(self.Delta - eval_with_pole(self.bathenergy, 1j*self.Z, self.bath_mat )))
    def obtain_orbitals(self,eps=1e-7):
        polelist = []
        veclist = []
        matlist = []
        for i in range(self.weight.shape[0]):
            eigval, eigvec = np.linalg.eig(self.weight[i])
            for j in range(eigval.shape[0]):
                if eigval[j]>eps:
                    polelist.append(self.pol[i])
                    veclist.append(eigvec[:,j]*np.sqrt(eigval[j]))
                    matlist.append((eigvec[:,j,None]*np.conjugate(eigvec[:,j].T))*(eigval[j]))

                    
        self.bathenergy, self.bathhyb, self.bath_mat = np.array(polelist), np.array(veclist), np.array(matlist)




def pole_fitting(Delta, Z, tol = None,Np = None, mmin=None,mmax = 50, maxiter = 50,cleanflag=False,fast=False, disp=True, complex = True):
    #pole estimation
    #tol needs to be fixed
    if Np == None and tol == None:
        raise Exception("One needs to specify either the number of poles or the fitting error tolerance.")
    if Np != None and tol != None: 
        raise Exception("One can not specify both the number of poles and the fitting error tolerance. Only specify one of them.")
    if Np == None:
        if mmin ==None:
            mmin = 4
    else:
        if Np % 2 ==1:
            Np = Np + 1
        mmin, mmax = Np, Np
        
    for m in range(mmin, mmax+1, 2):
        
        if len(Delta.shape)==1: 
            r0 = aaa_real(Delta, 1j*Z,mmax=m)
        else:
            r0 = aaa_matrix_real(Delta, 1j*Z,mmax=m)
        pol = np.real(r0.pol())
        weight, _, residue = get_weight(pol, 1.0j*Z, Delta,cleanflag=cleanflag,complex=complex,fast = fast)
        # print(np.max(np.abs(residue)))
        if tol!=None:
            if np.max(np.abs(residue))>tol*10:
                continue
        if Np==None: 
            pol, weight = aaa_reduce(pol, weight, 1e-5)
        # print("Number of poles is ", len(pol))
        if cleanflag == True:
            if maxiter>0:
                fhere = lambda pole: erroreval(pole,1j*Z,Delta,cleanflag=cleanflag,complex=complex)
                res = scipy.optimize.minimize(fhere,pol, method='L-BFGS-B', jac=True,options= {"disp" :disp,"maxiter":maxiter, "gtol":1e-10,"ftol":1e-10})
        else: 
            fhere = lambda pole: erroreval(pole,1j*Z,Delta,cleanflag=True,complex=complex)
            res = scipy.optimize.minimize(fhere,pol, method='L-BFGS-B', jac=True,options= {"disp" : False, "gtol":1e-10,"ftol":1e-10})
            if maxiter>0:
                fhere = lambda pole: erroreval(pole,1j*Z,Delta,cleanflag=False,complex= complex,fast = fast)
                res = scipy.optimize.minimize(fhere,res.x, method='L-BFGS-B', jac=True,options= {"disp" :disp,"maxiter":maxiter, "gtol":1e-10,"ftol":1e-10})
        
        weight, _, residuenew = get_weight(res.x, 1j*Z, Delta,cleanflag=cleanflag,fast=fast,complex=complex)
        if check_weight_psd(weight)==False:
            weight, _, residuenew = get_weight(res.x, 1j*Z, Delta,cleanflag=False,complex=complex)
        err = np.linalg.norm(residuenew)
        if tol != None:
            if np.max(np.abs(residuenew))<tol:
                return res.x, weight, np.max(np.abs(residuenew))
        else:
            return res.x, weight, np.max(np.abs(residuenew))

        

    if tol != None:
        print("Fail to reach desired fitting error!") 
    return res.x, weight, np.max(np.abs(residuenew))
def check_weight_psd(weight,atol=1e-6):
    check_psd = True
    for i in range(weight.shape[0]):
        val, _ = np.linalg.eig(weight[i])
        check_psd = check_psd and np.min(val.real)>-atol
    return check_psd
if __name__ == "__main__":
    file = open('Examples/hyb_data/omega_20231129.txt', "r")
    Z = np.loadtxt(file,dtype=np.complex128)
    file.close()
    Zid = abs(Z)<200

    import h5py
    with h5py.File("Examples/hyb_data/Delta.0.h5", "r") as f:
        # print(list(f.keys()))
        Delta = f[list(f.keys())[-1]]['Hyb'][:]

    Z = Z[Zid]


    #for i in range(Delta.shape[1]):
    for i in range(1):
        Delta0 = Delta[Zid,i,:,:]
        pol, weight = hybridization_fitting(Delta0, Z, mmax = 6, maxiter = 20)
        polelist, veclist, matlist = obtain_orbitals(pol,weight,eps=1e-7)
        # This is the result that we want: polelist, veclist

    # #plotting to see the result
    # Delta0_fit = eval_with_pole(polelist, 1j*Z, matlist) 

    # totalnum = 0
    # for i1 in range(8):
    #     for j1 in range(8):
    #         if np.linalg.norm(Delta0[:,i1,j1])>1e-2:
    #             totalnum = totalnum +1
    # tn = 0
    # ns = 5
    # fig, axs = plt.subplots(int(totalnum/ns)+3, ns, figsize=(20,20))
    # for i1 in range(8):
    #     for j1 in range(8):
    #         if np.linalg.norm(Delta0[:,i1,j1])>1e-3:
    #             p1 = tn//ns
    #             p2 = tn%ns
    #             axs[p1, p2].plot(Delta0[:, i1,j1].real)
    #             axs[p1, p2].plot(Delta0_fit[:, i1,j1].real,"--")
    #             axs[p1, p2].plot(Delta0[:, i1,j1].imag)
    #             axs[p1, p2].plot(Delta0_fit[:, i1,j1].imag,"--")
    #             axs[p1, p2].set_title("i = "+str(i1)+", j = "+str(j1))
    #             tn = tn+1
    # plt.show()
    # np.max(np.abs(Delta0-Delta0_fit))