"""
Adapol: Adaptive Pole Approximation of Frequency Data

User facing API for approximating frequency data 
with a sum of simple poles, using the AAA algorithm.

Author: Hugo U. R. Strand (2026)
"""


import numpy as np


from .aaa import aaa
from .sop import SumOfSimplePoles


def approximate_frequency_data_with_max_n_poles(F, Z, max_n_poles, verbose=False):
    """ Approximate frequency data :math:`F` sampled at points :math:`Z` 
    with a sum of simple poles, by running the AAA algorithm with a maximum 
    number of poles :math:`max_n_poles`.
     
    Parameters
    ----------
    F : array_like
        Frequency data to approximate, sampled at points :math:`Z`.
    Z : array_like
        Sample points in (complex) frequency space, at which :math:`F` is sampled.
    max_n_poles : int
        Maximum number of poles to use in the approximation. Note that the actual
        number of poles produced might be smaller than this, for two reasons:
        (1) an odd number of poles is always produced, from symmetry
        considerations, and (2) the AAA cleanup step might remove poles with
        small residues.
    verbose : bool, optional
        If True, print verbose output during the approximation process.
    
    Returns
    -------
    poles : ndarray
        Poles of the approximating sum of simple poles.
    residues : ndarray
        Residues of the approximating sum of simple poles.
    """
    return _frequency_data_driver(F, Z, max_n_poles=max_n_poles, tol=None, verbose=verbose)


def approximate_frequency_data_with_fixed_error_tolerance(F, Z, tol, verbose=False):
    """ Approximate frequency data :math:`F` sampled at points :math:`Z`
    with a sum of simple poles, by running the AAA algorithm 
    with a fixed error tolerance `tol`.
     
    Parameters
    ----------
    F : array_like
        Frequency data to approximate, sampled at points :math:`Z`.
    Z : array_like
        Sample points in (complex) frequency space, at which :math:`F` is sampled.
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
    return _frequency_data_driver(F, Z, max_n_poles=None, tol=tol, verbose=verbose)


def approximate_sum_of_simple_poles_with_max_n_poles(
        poles, residues, max_n_poles, beta, verbose=False, nonlinear_optimization=False):
    """ Approximate a sum of simple poles defined by `poles` and `residues`
    with a sum of simple poles with at most `max_n_poles` poles, by running the AAA algorithm.

    Parameters
    ----------
    poles : array_like
        Poles of the original sum of simple poles to approximate.
    residues : array_like
        Residues of the original sum of simple poles to approximate.
    max_n_poles : int
        Maximum number of poles to use in the approximation. Note that the actual
        number of poles produced might be smaller than this, for two reasons:
        (1) an odd number of poles is always produced, from symmetry
        considerations, and (2) the AAA cleanup step might remove poles with
        small residues.
    beta : float
        Inverse temperature, used to define the L2 norm in imaginary time.
    verbose : bool, optional
        If True, print verbose output during the approximation process.
    nonlinear_optimization : bool, optional
        If True, perform a non-linear optimization of the poles after the AAA approximation, 
        to further reduce the error.

    Returns
    -------
    poles : ndarray
        Poles of the approximating sum of simple poles.
    residues : ndarray
        Residues of the approximating sum of simple poles.
    """
    return _sum_of_simple_poles_driver(
        poles, residues, max_n_poles=max_n_poles, tol=None, beta=beta, 
        verbose=verbose, nonlinear_optimization=nonlinear_optimization)


def approximate_sum_of_simple_poles_with_fixed_error_tolerance(
        poles, residues, tol, beta, verbose=False, nonlinear_optimization=False):
    """ Approximate a sum of simple poles defined by `poles` and `residues`
    with a sum of simple poles with a fixed error tolerance `tol`, 
    by running the AAA algorithm.

    Parameters
    ----------
    poles : array_like
        Poles of the original sum of simple poles to approximate.
    residues : array_like
        Residues of the original sum of simple poles to approximate.
    tol : float
        Fixed error tolerance for the approximation.
    beta : float
        Inverse temperature, used to define the L2 norm in imaginary time.
    verbose : bool, optional
        If True, print verbose output during the approximation process.
    nonlinear_optimization : bool, optional
        If True, perform a non-linear optimization of the poles after the AAA approximation, 
        to further reduce the error.

    Returns
    -------
    poles : ndarray
        Poles of the approximating sum of simple poles.
    residues : ndarray
        Residues of the approximating sum of simple poles.
    """

    return _sum_of_simple_poles_driver(
        poles, residues, max_n_poles=None, tol=tol, beta=beta, 
        verbose=verbose, nonlinear_optimization=nonlinear_optimization)


def _frequency_data_driver(F, Z, max_n_poles, tol, verbose=False,
    cleanup=True, cleanup_residue_tol=1e-12, cleanup_imag_tol=1e-8):

    # Fixme: max_steps != n_poles 
    # Fixme: tol is only controlling AAA

    max_steps = _max_steps_from_max_n_poles(max_n_poles)

    bra = aaa(
        Z, F, tol=tol, max_steps=max_steps, constrained=True,
        cleanup=cleanup, cleanup_residue_tol=cleanup_residue_tol, cleanup_imag_tol=cleanup_imag_tol,
        verbose=verbose)

    sop = bra.get_sop()
    sop.fit_residues_to_freq_samples(Z, F)

    return sop.p, sop.R


def _sum_of_simple_poles_driver(poles, residues, max_n_poles, tol, beta, verbose=False,
    cleanup=True, cleanup_residue_tol=1e-12, cleanup_imag_tol=1e-8, 
    nonlinear_optimization=False, Z=None):

    if Z is None:
        # Frequency grid
        w_max = np.abs(poles).max() * 2
        n_max = int(2 * beta * w_max / np.pi) + 1
        Z = 1.j * np.pi / beta * np.arange(-n_max, n_max + 1)

    # Eval sop
    sop = SumOfSimplePoles(poles=poles, residues=residues)
    F = sop(Z)

    max_steps = _max_steps_from_max_n_poles(max_n_poles)

    # Run AAA
    bra = aaa(
        Z, F, tol=tol, max_steps=max_steps, constrained=True,
        cleanup=cleanup, cleanup_residue_tol=cleanup_residue_tol, cleanup_imag_tol=cleanup_imag_tol,
        verbose=verbose)

    # Fit residues
    sop_aaa = bra.get_sop()

    if nonlinear_optimization:
        sop_opt = sop.best_imtime_non_linear_lstsq_l2_norm_approximation_using_pole_guess(
            poles=sop_aaa.p, beta=beta, verbose=verbose)
    else:
        sop_opt = sop.best_imtime_lstsq_l2_norm_approximation_using_poles(sop_aaa.p, beta)

    return sop_opt.p, sop_opt.R


def _max_steps_from_max_n_poles(max_n_poles):
    return (max_n_poles + 1) // 2 if max_n_poles is not None else None