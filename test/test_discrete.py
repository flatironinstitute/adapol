import numpy as np
import sys
sys.path.insert(0, "../src")
from aaa import *
from ac_pes import *
import matplotlib.pyplot as plt
import scipy
from hybdecomp import *

def make_G_with_random_discrete_pole(Np,Z):
    pol = np.random.randn(Np)
    pol = pol/np.max(np.abs(pol))
    vec = scipy.stats.ortho_group.rvs(dim=Np)
    weight = np.array([vec[:,i,None]*np.transpose(np.conj(vec[:,i])) for i in range(Np)])
    #Z is real
    pol_t = np.reshape(pol,[pol.size,1])
    M = 1/(1j*Z-pol_t)
    M = M.transpose()
    if len(weight.shape)==1:
        weight = weight/sum(weight)
        G = M@weight
    else:
        Np = weight.shape[0]
        Norb = weight.shape[1]
        Nw = len(Z)
        G = (M@(weight.reshape(Np,Norb*Norb))).reshape(Nw,Norb,Norb)
    return pol, vec, weight, G

if __name__ == "__main__":
    beta = 20
    N = 105
    Z = (np.linspace(-N,N,N+1))*np.pi/beta
    tol = 1e-6
    for Np in range(2,10):
        pol_true, vec_true, weight_true, Delta = make_G_with_random_discrete_pole(Np,Z)
        
        bath_disc = bath(Delta = Delta,Z = Z)
        bath_disc.hyb_fit(tol = tol, maxiter = 50, disp = False, cleanflag = True)
        print("Fitting error is ",bath_disc.final_error)
        print("Weight PSD is ", check_weight_psd(bath_disc.weight))
        


        