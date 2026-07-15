"""Adapol/TRIQS: Adaptive Pole Approximation of Frequency Data

User facing TRIQS based API for approximating frequency data 
with a sum of simple poles, using the AAA algorithm.

Author: Hugo U. R. Strand (2026)"""

import numpy as np


from .adapol import _frequency_data_driver
from .adapol import _sum_of_simple_poles_driver


def approximate_gf_imfreq_with_max_n_poles(G_w, max_n_poles, verbose=False):
    """ Approximate a Green's function defined on the imaginary frequency axis,
    with a sum of simple poles, by running the AAA algorithm with a maximum
    number of poles `max_n_poles`.
    
    Parameters
    ----------
    G_w : triqs.gf.Gf
        Green's function defined on the imaginary frequency axis, to approximate.
    max_n_poles : int
        Maximum number of poles to use in the approximation.
    verbose : bool, optional
        If True, print verbose output during the approximation process.
        
    Returns
    -------
    poles : ndarray
        Poles of the approximating sum of simple poles.
    residues : ndarray
        Residues of the approximating sum of simple poles.
    error : float
        Maximum absolute error of the approximation at the sample points.
    """
    Z, F = _gf_imfreq_to_data(G_w)
    return _frequency_data_driver(F, Z, max_n_poles=max_n_poles, tol=None, verbose=verbose)


def approximate_gf_imfreq_with_fixed_error_tolerance(G_w, tol, verbose=False):
    """ Approximate a Green's function defined on the imaginary frequency axis,
    with a sum of simple poles, by running the AAA algorithm with a fixed error
    tolerance `tol`.
    
    Parameters
    ----------
    G_w : triqs.gf.Gf
        Green's function defined on the imaginary frequency axis, to approximate.
    tol : float
        Fixed error tolerance for the approximation.
    verbose : bool, optional
        If True, print verbose output during the approximation process.

    Returns
    -------
    poles : ndarray
        Poles of the approximating sum of simple poles.
    residues : ndarray
        Residues of the approximating sum of simple poles.
    error : float
        Maximum absolute error of the approximation at the sample points.
    """
    Z, F = _gf_imfreq_to_data(G_w)
    return _frequency_data_driver(F, Z, max_n_poles=None, tol=tol, verbose=verbose)


def approximate_gf_dlr_with_max_n_poles(
        G_dlr, max_n_poles, verbose=False, nonlinear_optimization=False):
    """ Approximate a Green's function defined on a DLR mesh, with a sum of simple poles,
    by running the AAA algorithm with a maximum number of poles `max_n_poles`.
    
    Parameters
    ----------
    G_dlr : triqs.gf.Gf
        Green's function defined on a DLR mesh, to approximate.
    max_n_poles : int
        Maximum number of poles to use in the approximation.
    verbose : bool, optional
        If True, print verbose output during the approximation process.
    nonlinear_optimization : bool, optional
        If True, perform nonlinear optimization of poles and residues after AAA compression.

    Returns
    -------
    poles : ndarray
        Poles of the approximating sum of simple poles.
    residues : ndarray
        Residues of the approximating sum of simple poles.
    error : float
        Normalized imaginary-time L2 norm of the difference between the
        original and approximating sum of simple poles.
    """
    poles, residues, beta, Z = _gf_dlr_to_data(G_dlr)
    return _sum_of_simple_poles_driver(
        poles, residues, max_n_poles=max_n_poles, tol=None, beta=beta, 
        verbose=verbose, nonlinear_optimization=nonlinear_optimization, Z=Z)


def approximate_gf_dlr_with_fixed_error_tolerance(
        G_dlr, tol, verbose=False, nonlinear_optimization=False):
    """ Approximate a Green's function defined on a DLR mesh, with a sum of simple poles,
    by running the AAA algorithm with a fixed error tolerance `tol`.
    
    Parameters
    ----------
    G_dlr : triqs.gf.Gf
        Green's function defined on a DLR mesh, to approximate.
    tol : float
        Fixed error tolerance for the approximation.
    verbose : bool, optional
        If True, print verbose output during the approximation process.
    nonlinear_optimization : bool, optional
        If True, perform nonlinear optimization of poles and residues after AAA compression.
    
    Returns
    -------
    poles : ndarray
        Poles of the approximating sum of simple poles.
    residues : ndarray
        Residues of the approximating sum of simple poles.
    error : float
        Normalized imaginary-time L2 norm of the difference between the
        original and approximating sum of simple poles."""
    
    poles, residues, beta, Z = _gf_dlr_to_data(G_dlr)
    return _sum_of_simple_poles_driver(
        poles, residues, max_n_poles=None, tol=tol, beta=beta, 
        verbose=verbose, nonlinear_optimization=nonlinear_optimization, Z=Z)


def approximate_gf_dlr_with_fixed_error_tolerance_in_imaginary_time(
        G_dlr, tol, verbose=False, nonlinear_optimization=False):
    """ Approximate a Green's function defined on a DLR mesh, with a sum of simple poles,
    by running the AAA algorithm with a fixed error tolerance `tol` in imaginary time.
    
    Parameters
    ----------
    G_dlr : triqs.gf.Gf
        Green's function defined on a DLR mesh, to approximate.
    tol : float
        Fixed error tolerance for the approximation.
    verbose : bool, optional
        If True, print verbose output during the approximation process.
    nonlinear_optimization : bool, optional
        If True, perform nonlinear optimization of poles and residues after AAA compression.
    
    Returns
    -------
    poles : ndarray
        Poles of the approximating sum of simple poles.
    residues : ndarray
        Residues of the approximating sum of simple poles.
    error : float
        Normalized imaginary-time L2 norm of the difference between the
        original and approximating sum of simple poles."""

    comp = TriqsDLRCompression(
        G_dlr, tol=tol, nonlinear_optimize=nonlinear_optimization, 
        nonlinear_post_optimize=nonlinear_optimization, verbose=verbose)
    return comp.poles, comp.residues, comp.error


def _gf_imfreq_to_data(G_w):
    Z = np.array([complex(w) for w in G_w.mesh])
    F = G_w.data.copy()
    return Z, F


def _gf_dlr_to_data(G_dlr):
    from triqs.gfs import MeshDLR, make_gf_dlr_imfreq
    if type(G_dlr.mesh) != MeshDLR:
        raise ValueError('G_dlr must be defined on a MeshDLR')
    beta = G_dlr.mesh.beta
    poles = np.array([float(w) for w in G_dlr.mesh]) / beta
    residues = G_dlr.data.copy()
    G_w = make_gf_dlr_imfreq(G_dlr)
    Z = np.array([complex(w) for w in G_w.mesh])
    return poles, residues, beta, Z


class TriqsDLRCompression:

    def __init__(self, G, tol=1e-14, 
                 nonlinear_optimize=False, nonlinear_post_optimize=False, 
                 max_upwind_steps=4, verbose=True):

        self.G = G
        self.tol = tol
        self.nonlinear_optimize = nonlinear_optimize
        self.nonlinear_post_optimize = nonlinear_post_optimize
        self.verbose = verbose

        from triqs.gfs import MeshDLR, make_gf_dlr

        self.G_dlr = G if type(G.mesh) == MeshDLR else make_gf_dlr(G)
        self.dlr_freq = np.array([float(w) for w in self.G_dlr.mesh])
        self.G_dlr_coeff = self.G_dlr.data.copy()
        self.beta = self.G_dlr.mesh.beta

        poles = self.dlr_freq / self.beta
        residues = self.G_dlr_coeff.copy()

        from triqs.gfs import MeshDLRImFreq, make_gf_dlr_imfreq

        self.G_w = G if type(G.mesh) == MeshDLRImFreq else make_gf_dlr_imfreq(G)
        self.Z = np.array([complex(w) for w in self.G_w.mesh])

        from .sop_compr import SumOfPolesCompression
        
        self.sop_comp = SumOfPolesCompression(
            poles=poles, residues=residues,
            Z=self.Z,
            beta=self.beta, tol=tol, 
            nonlinear_optimize=nonlinear_optimize, 
            nonlinear_post_optimize=nonlinear_post_optimize, 
            max_upwind_steps=max_upwind_steps, verbose=verbose)
        
        sc = self.sop_comp
        self.poles, self.residues, self.aaa_steps, self.error = sc.poles, sc.residues, sc.aaa_steps, sc.error