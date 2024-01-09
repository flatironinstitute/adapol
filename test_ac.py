import numpy as np
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
    Delta0 = Delta[:,0,0,0]

    ## Example: scalar data 
    # Analytic continuation uses the hybridization_fitting function
    # Delta0 is Matsubara data. A length-Nw np array or a Nw*1*1 np array.
    # here Z is (2n+1)*pi/beta, not 1j*(2n+1)*pi/beta
    # mmax needs to be an even integer, and controls the number of poles used to fit. (Number of poles <= mmax-1)
    pol, weight = hybridization_fitting(Delta0, Z, mmax = 6, maxiter = 200,disp=False)
    #reconstruct G(iw)
    Delta0_reconstruct = eval_with_pole(pol, 1j*Z, weight)
    #plot G and G_reconstruct
    plt.plot(Delta0.real)
    plt.plot(Delta0.imag)
    plt.plot(Delta0_reconstruct.real,"--")
    plt.plot(Delta0_reconstruct.imag,"--")
    plt.show()
    # reconstruct retarded Green's function on real axis and spectral function
    wlist = np.linspace(-12,12,1000)
    G_retarded = eval_with_pole(pol, wlist+0.01*1j, weight)
    spec = -np.imag(G_retarded)/np.pi
    plt.plot(wlist,spec)
    plt.show()
    breakpoint()
