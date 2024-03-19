import numpy as np
import sys
sys.path.insert(0, "../src")
from aaa import *
from ac_pes import *
import matplotlib.pyplot as plt
import scipy
from hybdecomp import *


if __name__ == "__main__":

    ## read in Matsubara data
    file = open('Examples/hyb_data/omega_20231129.txt', "r")
    Z = np.loadtxt(file,dtype=np.complex128)
    file.close()

    import h5py
    with h5py.File("Examples/hyb_data/Delta.0.h5", "r") as f:
        # print(list(f.keys()))
        Delta = f[list(f.keys())[-1]]['Hyb'][:]
    Delta0 = Delta[:,0,:,:]
    ## Example: scalar data 
    # Analytic continuation uses the hybridization_fitting function
    # Delta0 is Matsubara data. A length-Nw np array or a Nw*1*1 np array.
    # here Z is (2n+1)*pi/beta, not 1j*(2n+1)*pi/beta
    # mmax needs to be an even integer, and controls the number of poles used to fit. (Number of poles <= mmax-1)
    pol, weight, err = pole_fitting(Delta0, Z,Np = 4, cleanflag = True,maxiter = 500,disp=False)
    print(err)
    polelist, veclist, matlist = obtain_orbitals(pol,weight,eps=1e-7)
    #reconstruct G(iw)
    Delta0_reconstruct = eval_with_pole(polelist, 1j*Z, matlist)

    def plotcompare(Delta0, Delta0_reconstruct,i,j,eps):
        fig, axs = plt.subplots(i, j, figsize=(5,5))
        tn=0
        for i1 in range(8):
            for j1 in range(8):
                if np.linalg.norm(Delta0[:,i1,j1])>eps:
                    p1 = tn//j
                    p2 = tn%j
                    axs[p1, p2].plot(Delta0[:, i1,j1].real)
                    axs[p1, p2].plot(Delta0_reconstruct[:, i1,j1].real,"--")
                    axs[p1, p2].plot(Delta0[:, i1,j1].imag)
                    axs[p1, p2].plot(Delta0_reconstruct[:, i1,j1].imag,"--")
                    axs[p1, p2].set_title("i = "+str(i1)+", j = "+str(j1))
                    axs[p1,p2].set_xticks([])
                    tn = tn+1
        plt.show()
    plotcompare(Delta0, Delta0_reconstruct,5,8,1e-4)
    # # reconstruct retarded Green's function on real axis and spectral function
    # wlist = np.linspace(-12,12,1000)
    # G_retarded = eval_with_pole(pol, wlist+0.01*1j, weight)
    # spec = -np.imag(G_retarded)/np.pi
    # plt.plot(wlist,spec)
    # plt.show()

