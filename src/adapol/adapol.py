"""
Adapol: Adaptive Pole Approximation of Frequency Data

User facing API for approximating frequency data with a sum of simple poles, using the AAA algorithm.

Author: Hugo U. R. Strand (2026)
"""


from .aaa_bra import aaa_bra
from .sop import SumOfSimplePoles


def approximate_frequency_data_with_n_poles(F, Z, n_poles, verbose=False):
    return _frequency_data_driver(F, Z, n_poles=n_poles, tol=None, verbose=verbose)


def approximate_frequency_data_with_fixed_error_tolerance(F, Z, tol, verbose=False):
    return _frequency_data_driver(F, Z, n_poles=None, tol=tol, verbose=verbose)


def approximate_sum_of_simple_poles_with_n_poles(
        poles, residues, n_poles, beta, verbose=False, nonlinear_optimization=False):

    return _sum_of_simple_poles_driver(
        poles, residues, n_poles=n_poles, tol=None, beta=beta, 
        verbose=verbose, nonlinear_optimization=nonlinear_optimization)


def approximate_sum_of_simple_poles_with_fixed_error_tolerance(
        poles, residues, tol, beta, verbose=False, nonlinear_optimization=False):

    return _sum_of_simple_poles_driver(
        poles, residues, n_poles=None, tol=tol, beta=beta, 
        verbose=verbose, nonlinear_optimization=nonlinear_optimization)


def _frequency_data_driver(F, Z, n_poles, tol, verbose=False,
    cleanup=True, cleanup_residue_tol=1e-14, cleanup_imag_tol=1e-14):

    # Fixme: max_steps != n_poles 
    # Fixme: tol is only controlling AAA

    bra = aaa_bra(
        Z, F, tol=None, max_steps=n_poles, constrained=True,
        cleanup=cleanup, cleanup_residue_tol=cleanup_residue_tol, cleanup_imag_tol=cleanup_imag_tol,
        verbose=verbose)

    sop = bra.get_sop()
    sop.fit_residues_to_freq_samples(Z, F)

    return sop.p, sop.R


def _sum_of_simple_poles_driver(poles, residues, n_poles, tol, beta, verbose=False,
    cleanup=True, cleanup_residue_tol=1e-14, cleanup_imag_tol=1e-14, nonlinear_optimization=False):

    # Frequency grid
    w_max = np.abs(poles).max() * 2
    n_max = int(2 * beta * w_max / np.pi) + 1
    Z = np.pi / beta * np.arange(-n_max, n_max + 1)

    # Eval sop
    sop = SumOfSimplePoles(poles=poles, residues=residues)
    F = sop(Z)

    # Run AAA
    bra = aaa_bra(
        Z, F, tol=tol, max_steps=n_poles, constrained=True,
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
