""" The pole and residue steps shared by the sum of poles routines.

An approximation pass consists of an AAA pole step, determining the pole
locations, followed by a residue step, determining the residues (and, optionally,
relocating the poles) by minimizing the imaginary time L2 norm error.

`adapol.approx_sop_fast` makes a single pass, while `adapol.approx_sop_tol`
repeats the pass, searching for the smallest approximation that meets a
tolerance, and caches the two steps separately.

Author: Hugo U. R. Strand, 2026
"""


import numpy as np

from .aaa import aaa


def _fermionic_matsubara_frequency_grid(poles, beta):
    w_max = np.abs(poles).max()
    n_max = int(4 * beta * w_max / np.pi) + 1
    n = np.arange(-n_max, n_max)
    return 1.j * np.pi / beta * (2 * n + 1)


def _aaa_pole_step(Z, F, max_steps=None, tol=None, cleanup=True,
                   cleanup_residue_tol=1e-12, cleanup_imag_tol=1e-8,
                   verbose=False, prefix=''):

    """ Determine pole locations by running the conjugate pair constrained AAA
    algorithm on the frequency data `F` sampled at the (complex) frequencies `Z`.

    The AAA residues are not returned, since the subsequent residue step
    determines the residues anew.

    Returns (poles, aaa_steps, aaa_residual), where `aaa_steps` is the number of
    AAA steps actually taken and `aaa_residual` the AAA residual, i.e. the maximum
    absolute deviation from `F` over the sample points not used as support points. """

    bra = aaa(
        Z, F, tol=tol, max_steps=max_steps, constrained=True,
        cleanup=cleanup, cleanup_residue_tol=cleanup_residue_tol,
        cleanup_imag_tol=cleanup_imag_tol, verbose=verbose, prefix=prefix)

    return bra.get_sop().p, bra.aaa_steps, bra.residual


def _imtime_residue_step(sop, poles, beta, nonlinear=False, verbose=False):

    """ Approximate the sum of simple poles `sop` using the given `poles`, by
    minimizing the imaginary time L2 norm error at inverse temperature `beta`.

    The residues are fit by linear least squares for the given poles, or, if
    `nonlinear` is True, the pole locations and residues are jointly optimized,
    using `poles` as the initial guess.

    Returns (poles, residues, err), where `err` is the imaginary time L2 norm of
    the difference between `sop` and the approximation. """

    if nonlinear:
        sop_opt = sop.best_imtime_non_linear_lstsq_l2_norm_approximation_using_pole_guess(
            poles=poles, beta=beta, verbose=verbose)
    else:
        sop_opt = sop.best_imtime_lstsq_l2_norm_approximation_using_poles(poles, beta)

    err = (sop - sop_opt).imtime_l2_norm(beta=beta)

    return sop_opt.p, sop_opt.R, err
