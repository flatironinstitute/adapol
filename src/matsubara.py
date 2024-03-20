import sys
# sys.path.insert(0, "../")
import numpy as np
from aaa import *
from fit_utils import *

import matplotlib.pyplot as plt
import scipy

class Matsubara(object):
    def __init__(self, Delta, Z):
        """ Initialization of a Matsubara object
        Parameters:
        --------
        Delta: Matsubara functions in the imaginary frequency domain, np array
            size: Nw * Norb * Norb
            if Delta is a 1d-array, it will be reshaped into Np * 1 * 1
        
        Z: Matsubara frequencies, 1d array
            size: Nw*1
            Note: Z = k*pi/beta not 1j*k*pi/beta

        """
        if len(Delta.shape)==1:
            Delta = Delta.reshape(Delta.shape[0],1,1)
        self.Delta = Delta
        self.Z = Z

    def bathfitting_tol(self, tol=1e-3, maxiter = 500, mmin = 4, mmax = 50, eps = 1e-7, cleanflag=False, disp = False):
        """Conduct bath fitting, with fixed error tolerance tol.

        The number of poles is increased until reaching desired accuracy.

        Examples:
        --------
            bath.hyb_fit(tol = tol, maxiter = 50) # fix until tol accuracy
            bath.hyb_fit(tol = tol, maxiter = 50, cleanflag = False) # fit until tol accuracy, with improved accuracy (and bigger computational cost)
        
        Parameters:
        --------
        tol: Fitting error tolreance, float
            default: 1e-3

        mmin, mmax: number of minimum or maximum poles, integer
            default: mmin = 4, mmax = 50
            
        
        maxiter: int
            maximum number of iterations
            default: 50

        eps: float, optional
            Truncation threshold for bath orbitals while doing SVD of weight matrices
            default:1e-7
        
        cleanflag: bool
            whether to use least square to replace semidefinite programming (SDP) to fasten calculation
            default: False

        disp: bool
            whether to display optimization details
            default: False


        Output:
        -------
        bathenergy: energy of bath orbitals, 1d array of shape (Nb)
        bathhyb: coefficient of bath orbitals, 2d array of shape (Nb, Norb)


        """
        self.pol, self.weight, self.fitting_error = pole_fitting(self.Delta, self.Z, tol = tol, mmin = mmin, mmax=mmax, \
                                                                    maxiter = maxiter, cleanflag=cleanflag, disp=disp)
        self.obtain_orbitals(eps=eps) 
        self.final_error = np.max(np.abs(self.Delta - eval_with_pole(self.bathenergy, 1j*self.Z, self.bath_mat )))

        return self.bathenergy, self.bathhyb
    def bathfitting_num_poles(self, Np = 4, maxiter = 500, eps = 1e-7, cleanflag = False, disp = False):
        """Conduct bath fitting, with fixed number of poles.
        Examples:
        --------
            bath.hyb_fit(Np = Np, maxiter = 500) # fix with Np poles
            bath.hyb_fit(Np = Np, maxiter = 500, cleanflag = False) # fit with Np poles, with improved accuracy (and bigger computational cost)
        
        Parameters:
        --------
        Np: number of Matsubara points used for fitting, integer
            default: 4
            Np needs to be an even integer, and number of poles is Np - 1.
        
        maxiter: int
            maximum number of iterations
            default: 50

        eps: float, optional
            Truncation threshold for bath orbitals while doing SVD of weight matrices
            default:1e-7
        
        cleanflag: bool
            whether to use least square to replace semidefinite programming (SDP) to fasten calculation
            default: False

        disp: bool
            whether to display optimization details
            default: False

        
        Output:
        -------
        bathenergy: energy of bath orbitals, 1d array of shape (Nb)
        bathhyb: coefficient of bath orbitals, 2d array of shape (Nb, Norb)

        """

        self.pol, self.weight, self.fitting_error = pole_fitting(self.Delta, self.Z, Np = Np, \
                                                                    maxiter = maxiter, cleanflag=cleanflag, disp=disp)
        self.obtain_orbitals(eps=eps) 
        self.final_error = np.max(np.abs(self.Delta - eval_with_pole(self.bathenergy, 1j*self.Z, self.bath_mat )))

        return self.bathenergy, self.bathhyb
    

    def analytic_cont_tol(self, tol=1e-3, maxiter = 500, mmin = 4, mmax = 50, cleanflag=False, disp = False):
        '''
        Analytic continuation for fixed error tolerance tol

        Input: same as bath_fitting_tol

        Output: Green's function evaluator
        '''
        self.pol, self.weight, self.fitting_error = pole_fitting(self.Delta, self.Z, tol = tol, mmin = mmin, mmax=mmax, \
                                                                    maxiter = maxiter, cleanflag=cleanflag, disp=disp)
        self.func = lambda Z: eval_with_pole(self.pol, Z, self.weight)
        return self.func
    
    def analytic_cont_num_poles(self, Np = 4, maxiter = 500, cleanflag = False, disp = False):

        '''
        Analytic continuation for fixed number of poles

        Input: same as bath_fitting_num_poles

        Output: Green's function evaluator
        ''' 
        self.pol, self.weight, self.fitting_error = pole_fitting(self.Delta, self.Z, Np = Np, \
                                                                    maxiter = maxiter, cleanflag=cleanflag, disp=disp)
        self.func = lambda Z: eval_with_pole(self.pol, Z, self.weight)

        return self.func



    def obtain_orbitals(self,eps=1e-7):
        '''
        obtaining bath orbitals through svd
        '''
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

    




