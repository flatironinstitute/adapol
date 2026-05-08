"""
Sum of simple poles (SOP) rational function representation.

(Used internally in Adapol routines.) 

Author: Hugo U. R. Strand, 2026
"""


import numpy as np


from adapol.imtime import ImTimeQuadrature
from adapol.imtime import kernel as imtime_kernel


class SumOfSimplePoles:

    """ Rational function represented as a sum of simple poles, 

    .. math::
        s(z) = \\sum_k R_k / (z - p_k)
         
    where :math:`p_k` are the poles and :math:`R_k` are the residues. """
    
    def __init__(self, poles, residues):
        assert( len(poles) == residues.shape[0] )
        self.p = poles
        self.R = residues


    def __call__(self, z):
        C = 1. / (z[:, None] - self.p[None, :])
        return np.einsum('zp,p...->z...', C, self.R)
    

    def fit_residues_to_freq_samples(self, Z, F):
        """ Fit the residues using sample points :math:`Z` and values :math:`F` in (complex) frequency space,
        by solving the linear system :math:`C R = F`, using least squares, where :math:`C_{jk} = 1/(Z_j - p_k)`. """
        
        n = len(Z)
        C = 1. / (Z[:, None] - self.p[None, :])
        residues, _, _, _ = np.linalg.lstsq(C, F.reshape(n, -1), rcond=None)
        self.R = residues.reshape([len(self.p)] + list(F.shape[1:]))


    def imtime_l2_norm(self, beta):
        itq = self.get_imtime_quadrature(beta)
        f_tau = self.imtime_function(beta)
        return itq.l2_norm(f_tau)

    
    def best_imtime_lstsq_l2_norm_approximation_using_poles(self, poles, beta):
        itq = self.get_imtime_quadrature(beta)
        f_tau = self.imtime_function(beta)
        residues = itq.best_l2_norm_approximation_using_poles(f_tau, poles)
        sop = SumOfSimplePoles(poles=poles, residues=residues)
        return sop


    def best_imtime_non_linear_lstsq_l2_norm_approximation_using_pole_guess(self, poles, beta, verbose=False):
        itq = self.get_imtime_quadrature(beta)
        sop_opt = itq.best_l2_norm_approximation(self, poles, verbose=verbose)
        return sop_opt


    def get_imtime_quadrature(self, beta):
        w_max = 2 * np.max(np.abs(self.p))
        itq = ImTimeQuadrature(lamb=w_max*beta, beta=beta)
        return itq


    def eval_imtime(self, tau_i, beta):
        """ Evaluate the sum of simple poles at imaginary time points :math:`\\tau_i`
        using the kernel :math:`K(\\tau, \\omega)` for inverse temperature :math:`\\beta`. """
        K_ip = imtime_kernel(tau_i / beta, self.p * beta)
        return np.einsum('tp,p...->t...', K_ip, self.R)
    

    def imtime_function(self, beta):
        """ Return a function of imaginary time :math:`\\tau` that evaluates the sum of simple poles at :math:`\\tau` using the kernel :math:`K(\\tau, \\omega)` for inverse temperature :math:`\\beta`. """
        return lambda tau: self.eval_imtime(tau, beta)


    def __add__(self, other):
        assert(isinstance(other, SumOfSimplePoles))
        p = np.concatenate([self.p, other.p])
        R = np.concatenate([self.R, other.R])
        return SumOfSimplePoles(poles=p, residues=R)


    def __mul__(self, other):
        if isinstance(other, (int, float, complex)):
            return SumOfSimplePoles(poles=self.p, residues=other * self.R)
        else:
            raise NotImplementedError("Multiplication only implemented for scalars.")


    def __rmul__(self, other): return self * other
    def __neg__(self): return self * -1.
    def __sub__(self, other): return self + (other * -1.)
