import numpy as np
import sys
sys.path.insert(0, "../src")
import matplotlib.pyplot as plt
import scipy
from matsubara import *
def Kw(w,v):
    return 1/(1j*v-w)

def semicircular(x):
    return 2*np.sqrt(1-x**2)/np.pi
def make_G_with_cont_spec(N1,Z,rho,a=-1.0, b=1.0,eps=1e-12):
    N1 = 3
    H = np.random.rand(N1,N1) + 1j* np.random.rand(N1,N1)
    H = H + np.conj(H.T)

    G = np.zeros((Z.shape[0], H.shape[0], H.shape[1]), dtype = np.complex128)
    en, vn = np.linalg.eig(H)
    en = en/np.max(np.abs(en))#np.random.rand(en.shape[0])
    for i in range(en.shape[0]):
        for n in range(len(Z)):
            f = lambda w: Kw(w-en[i],Z[n])*rho(w)
            gn = scipy.integrate.quad(f, a, b,epsabs=eps,epsrel=eps,complex_func=True)[0]
            G[n,:,:] = G[n,:,:] + gn*vn[:,i]*np.conj(np.transpose(vn[None,:,i]))
        
    return H, G

if __name__ == "__main__":
    beta = 20
    N = 55
    Z = (np.linspace(-N,N,N+1))*np.pi/beta

    dim = 3
    H, Delta = make_G_with_cont_spec(dim,Z, semicircular)
    tol = 1e-8
    for Np in [4,6,8,10,12]:
        # pol, weight, err = pole_fitting(Delta, Z, Np = Np , maxiter = 500,disp=False,cleanflag=True)
        ImFreq_obj = Matsubara(Delta = Delta,Z = Z)
        bath_energy, bath_hyb = ImFreq_obj.bathfitting_num_poles(Np = Np, maxiter = 500, disp = False, cleanflag = True)
        print("When number of poles is ", len(ImFreq_obj.pol))
        print("Fitting error is ",ImFreq_obj.final_error)
        print("Weight PSD is ", check_weight_psd(ImFreq_obj.weight))

        