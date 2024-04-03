import numpy as np
import sys
sys.path.insert(0, "../")
import matplotlib.pyplot as plt
import scipy
from matsubara import *

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


def tst_discrete(Np): 
    beta = 20
    N = 105
    Z = (np.linspace(-N,N,N+1))*np.pi/beta
    tol = 1e-6 
    pol_true, vec_true, weight_true, Delta = make_G_with_random_discrete_pole(Np,Z)
    ImFreq_obj = Matsubara(Delta = Delta, Z = Z)
    # bath_energy, bath_hyb = ImFreq_obj.bathfitting_tol(tol = tol, maxiter = 50, disp = False, cleanflag = True)
    bath_energy, bath_hyb = ImFreq_obj.fitting(tol = tol, maxiter = 50, cleanflag = True, flag="hybfit")
    assert ImFreq_obj.final_error<tol
    assert check_weight_psd(ImFreq_obj.weight) == True

def test_discrete_2():
    tst_discrete(2)
def test_discrete_3():
    tst_discrete(3)
def test_discrete_4():
    tst_discrete(4)
def test_discrete_5():
    tst_discrete(5)
def test_discrete_6():
    tst_discrete(6)
def test_discrete_7():
    tst_discrete(7)
def test_discrete_8():
    tst_discrete(8)
        
        


        