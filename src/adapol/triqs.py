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
        Residues of the approximating sum of simple poles."""
    poles, residues, beta, Z = _gf_dlr_to_data(G_dlr)
    return _sum_of_simple_poles_driver(
        poles, residues, max_n_poles=None, tol=tol, beta=beta, 
        verbose=verbose, nonlinear_optimization=nonlinear_optimization, Z=Z)


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