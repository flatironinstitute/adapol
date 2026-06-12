"""
Adapol: Adaptive Pole Approximation of Frequency Data

User-facing API for approximating frequency data
with a sum of simple poles, using the AAA algorithm.

Author: Hugo U. R. Strand (2026)
"""


import numpy as np


from .aaa import aaa
from .sop import SumOfSimplePoles


def approximate_freq_aaa(F, Z, max_n_poles=None, aaa_tol=None, verbose=False):
    """Approximate frequency data :math:`F` sampled at points :math:`Z`
    with a sum of simple poles, by running the AAA algorithm.

    The approximation is built in two steps:

    1. **Pole step:** the AAA algorithm is run on the data :math:`(Z, F)` to
       determine the pole locations.
    2. **Residue step:** the residues are then determined by a linear
       least-squares fit that minimizes the error against the
       frequency-domain data :math:`F` at the sample points :math:`Z`.

    - If only `max_n_poles` is set, AAA runs until at most `max_n_poles` poles
      are used.
    - If only `aaa_tol` is set, AAA runs until the AAA error tolerance
      `aaa_tol` is reached.
    - If both are set, AAA stops as soon as either the AAA error tolerance
      `aaa_tol` is reached or `max_n_poles` poles are used, whichever happens
      first.

    Note
    ----
    `aaa_tol` controls only the pole step (AAA); it does not directly bound the
    final error, which also depends on the subsequent residue fit.

    Parameters
    ----------
    F : array_like
        Frequency data to approximate, sampled at points :math:`Z`.
    Z : array_like
        Sample points in (complex) frequency space, at which :math:`F` is sampled.
    max_n_poles : int, optional
        Maximum number of poles to use in the approximation. Note that the actual
        number of poles produced might be smaller than this, for two reasons:
        (1) an odd number of poles is always produced, from symmetry
        considerations, and (2) the AAA cleanup step might remove poles with
        small residues.
    aaa_tol : float, optional
        Error tolerance for the AAA algorithm. This is the maximum absolute
        error over the frequency-domain data :math:`F` at the sample points
        :math:`Z` used by the AAA algorithm.
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
    if max_n_poles is None and aaa_tol is None:
        raise ValueError("At least one of `max_n_poles` or `aaa_tol` must be provided.")

    return _frequency_data_driver(F, Z, max_n_poles=max_n_poles, tol=aaa_tol, verbose=verbose)


def approximate_sop_fast(
        poles, residues, beta, max_n_poles=None, aaa_tol=None,
        verbose=False, nonlinear_optimization=False):
    """Approximate a sum of simple poles defined by `poles` and `residues`
    with a sum of a (possibly) smaller number of simple poles, by running the
    AAA algorithm and, optionally, a non-linear optimization step.

    The original sum of simple poles is first evaluated on an equispaced grid
    on the imaginary-frequency axis to produce the frequency-domain data used
    by the AAA algorithm (see Note below for the grid definition).

    The approximation is then built in two steps:

    1. **Pole step:** the AAA algorithm is run on this frequency-domain data to
       determine the pole locations.
    2. **Residue step:** the residues are determined by minimizing the
       imaginary-time :math:`L^2(\\tau)` norm of the difference from the
       original sum of simple poles. By default this is a linear least-squares
       fit of the residues for the AAA poles; if `nonlinear_optimization` is
       True, the pole locations and residues are instead jointly optimized
       (see the `nonlinear_optimization` parameter below).

    Note that, unlike `approximate_freq_aaa`, the residues here are fit in
    imaginary time, not in the frequency domain.

    - If only `max_n_poles` is set, AAA runs until at most `max_n_poles` poles
      are used.
    - If only `aaa_tol` is set, AAA runs until the AAA error tolerance
      `aaa_tol` is reached.
    - If both are set, AAA stops as soon as either the AAA error tolerance
      `aaa_tol` is reached or `max_n_poles` poles are used, whichever happens
      first.

    Note
    ----
    Setting `aaa_tol` does **not** guarantee that the final imaginary-time L2 norm
    `error` is below `aaa_tol`; this is not possible to guarantee within the AAA
    algorithm alone. However, this function returns the final imaginary-time L2
    norm `error` of approximation (see Returns below).
    
    Note
    ----
    The frequency-domain data used by the AAA algorithm is obtained by
    evaluating the original sum of simple poles on an equispaced
    imaginary-frequency grid :math:`Z_n = i \\pi n / \\beta`, with spacing
    :math:`\\pi / \\beta`, for integer :math:`n = -n_{max}, \\ldots, n_{max}`,
    where :math:`n_{max} = \\lfloor 4 \\beta \\, \\omega_{max} / \\pi \\rfloor + 1`
    and :math:`\\omega_{max} = \\max_k |poles_k|` is the largest pole magnitude.
    This grid extends to roughly :math:`4 \\, \\omega_{max}` on the imaginary
    axis.

    Parameters
    ----------
    poles : array_like
        Poles of the original sum of simple poles to approximate.
    residues : array_like
        Residues of the original sum of simple poles to approximate.
    beta : float
        Inverse temperature, used to define the L2 norm in imaginary time
        and the imaginary-frequency grid on which the AAA data is sampled.
    max_n_poles : int, optional
        Maximum number of poles to use in the approximation. Note that the actual
        number of poles produced might be smaller than this, for two reasons:
        (1) an odd number of poles is always produced, from symmetry
        considerations, and (2) the AAA cleanup step might remove poles with
        small residues.
    aaa_tol : float, optional
        Error tolerance for the AAA algorithm. This is the maximum absolute
        error over the imaginary-frequency-domain data used by the AAA
        algorithm (the original sum of simple poles evaluated on the grid
        described above).
    verbose : bool, optional
        If True, print verbose output during the approximation process.
    nonlinear_optimization : bool, optional
        If True, run a non-linear optimization step after the AAA approximation,
        using the AAA poles only as an initial guess. This optimization keeps
        the number of poles fixed and jointly relocates the pole positions and
        refits the residues to minimize the imaginary-time L2 norm error
        between the original and approximating sum of simple poles. It is solved
        with the L-BFGS-B method (using an analytic gradient) and is run to
        optimizer convergence, i.e. to reduce the error as much as possible
        rather than to meet a prescribed error tolerance.

    Returns
    -------
    poles : ndarray
        Poles of the approximating sum of simple poles.
    residues : ndarray
        Residues of the approximating sum of simple poles.
    error : float
        L2 norm of the difference in imaginary time between the original
        and approximating sum of simple poles.
    """
    if max_n_poles is None and aaa_tol is None:
        raise ValueError("At least one of `max_n_poles` or `aaa_tol` must be provided.")

    return _sum_of_simple_poles_driver(
        poles, residues, max_n_poles=max_n_poles, tol=aaa_tol, beta=beta,
        verbose=verbose, nonlinear_optimization=nonlinear_optimization)


def approximate_sop_tol(
        poles, residues, tol, beta, verbose=False, nonlinear_optimization=False):
    """Approximate a sum of simple poles defined by `poles` and `residues`
    with the smallest sum of simple poles whose imaginary-time
    :math:`L^2(\\tau)` error is below the tolerance `tol`.

    Unlike `approximate_sop_fast`, where the tolerance only controls the AAA
    pole step, here `tol` is imposed on the **final** imaginary-time error,
    i.e. after the residues have been fit. Since AAA only determines pole
    locations and the residues (and hence the final error) are only known after
    the residue step, the requested error cannot be reached by a single AAA run.
    Instead, the AAA + residue pipeline is run repeatedly to search for the
    minimal number of poles that achieves `tol`.

    Each candidate fit is built in two steps:

    1. **Pole step:** the original sum of simple poles is evaluated on an
       equispaced imaginary-frequency grid (see `approximate_sop_fast` for the
       grid definition) and AAA is run on this data to determine pole locations.
    2. **Residue step:** the residues are determined by minimizing the
       imaginary-time :math:`L^2(\\tau)` norm of the difference from the
       original sum of simple poles -- a linear least-squares fit of the
       residues for the AAA poles, or, if `nonlinear_optimization` is True, a
       joint optimization of the pole locations and residues.

    The pole-count search proceeds in two phases: first the number of AAA poles
    is increased until the imaginary-time error drops below `tol`, then a
    bisection on the number of AAA steps locates the smallest pole count that
    still meets `tol`.

    Parameters
    ----------
    poles : array_like
        Poles of the original sum of simple poles to approximate.
    residues : array_like
        Residues of the original sum of simple poles to approximate.
    tol : float
        Target tolerance on the final imaginary-time :math:`L^2(\\tau)` norm of
        the difference between the original and approximating sum of simple
        poles.
    beta : float
        Inverse temperature, used to define the L2 norm in imaginary time
        and the imaginary-frequency grid on which the AAA data is sampled.
    verbose : bool, optional
        If True, print verbose output during the approximation process.
    nonlinear_optimization : bool, optional
        If True, the residue step jointly optimizes the pole locations and
        residues (rather than fitting residues only) to minimize the
        imaginary-time :math:`L^2(\tau)` error.

    Returns
    -------
    poles : ndarray
        Poles of the approximating sum of simple poles.
    residues : ndarray
        Residues of the approximating sum of simple poles.
    error : float
        L2 norm of the difference in imaginary time between the original
        and approximating sum of simple poles.

    Raises
    ------
    ValueError
        If the target tolerance `tol` cannot be achieved within the internal
        maximum number of search steps.
    """

    from .sop_compr import SumOfPolesCompression
    sc = SumOfPolesCompression(
        poles, residues, beta, tol=tol, 
        nonlinear_optimize=nonlinear_optimization, 
        nonlinear_post_optimize=nonlinear_optimization, verbose=verbose)
    return sc.poles, sc.residues, sc.error    


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

    error = np.max(np.abs(sop(Z) - F))

    return sop.p, sop.R, error


def _sum_of_simple_poles_driver(poles, residues, max_n_poles, tol, beta, verbose=False,
    cleanup=True, cleanup_residue_tol=1e-12, cleanup_imag_tol=1e-8, 
    nonlinear_optimization=False, Z=None):

    if Z is None:
        Z = _equispaced_imaginary_frequency_grid(poles, beta)

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

    error = (sop - sop_opt).imtime_l2_norm(beta=beta)

    return sop_opt.p, sop_opt.R, error


def _max_steps_from_max_n_poles(max_n_poles):
    return (max_n_poles + 1) // 2 if max_n_poles is not None else None


def _equispaced_imaginary_frequency_grid(poles, beta):
    w_max = np.abs(poles).max()
    n_max = int(4 * beta * w_max / np.pi) + 1
    return 1.j * np.pi / beta * np.arange(-n_max, n_max + 1)