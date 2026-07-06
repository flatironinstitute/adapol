"""
Adapol: Adaptive Pole Approximation of Frequency Data

User-facing API for approximating frequency data
with a sum of simple poles, using the AAA algorithm.

Authors: Hugo U. R. Strand, Jason Kaye (2026)
"""


import numpy as np


from .aaa import aaa
from .sop import SumOfSimplePoles


def approx_freq_aaa(F, Z, max_n_poles=None, aaa_tol=None, verbose=False):
    """Approximate frequency data :math:`F` sampled at points :math:`Z`
    with a sum of simple poles, by running the AAA algorithm.

    Parameters
    ----------
    F : (N, ...) array_like
        Frequency data to approximate, sampled at points :math:`Z`.
    Z : (N,) array_like
        Sample points in (complex) frequency space, at which :math:`F` is sampled.
    max_n_poles : int, optional
        Maximum number of poles to use in the approximation.
    aaa_tol : float, optional
        Error tolerance for the AAA algorithm.
    verbose : bool, optional
        If True, print verbose output during the approximation process.

    Returns
    -------
    poles : (M,) ndarray
        Poles of the approximating sum of simple poles (`M <= max_n_poles`).
    residues : (M, ...) ndarray
        Residues of the approximating sum of simple poles.
    error : float
        Maximum absolute error of the approximation at the sample points.

    Notes
    -----
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
    The number of poles produced might be smaller than `max_n_poles`, for two reasons:

    1. an odd number of poles is always produced, from symmetry considerations, and 
    2. the AAA cleanup step might remove poles with small residues.

    Note
    ----
    The error tolerance `aaa_tol` controls the maximum absolute error 
    over the frequency-domain data :math:`F` at the sample points :math:`Z` 
    in the AAA algorithm (pole step); it does **not** bound the
    final error, which also depends on the subsequent residue fit.

    Examples
    --------
    Fit a simple function with two poles :math:`F(Z) = 1/(Z-1) + 0.5/(Z+2)`
    sampled on an equispaced imaginary-frequency grid :math:`Z \\in [-10i, 10i]`.

    >>> import numpy as np
    >>> np.set_printoptions(precision=2, suppress=True)
    >>> from adapol.adapol import approx_freq_aaa
    >>> # Sample frequency data F at points Z
    >>> Z = 1j * np.linspace(-10, 10, 100)
    >>> F = 1 / (Z - 1) + 0.5 / (Z + 2)  # Example frequency data with two poles
    >>> # Approximate F with a sum of simple poles using AAA
    >>> poles, residues, error = approx_freq_aaa(F, Z, aaa_tol=1e-12)
    >>> poles
    array([-2.  ,  0.03,  1.  ])
    >>> residues
    array([ 0.5+0.j, -0. -0.j,  1. +0.j])
    >>> float(error) < 1e-12
    True

    Note that the fit contains three poles, not two,due to the constrained AAA
    algorithm. However, the additional pole has a residue that is numerically zero.

    Also tensor valued functions can be fitted, e.g. a 2x2 matrix valued function
    :math:`\\dim(F(Z)) = 2 \\times 2` with two poles and two matrix residues:

    >>> R1 = np.array([[1, 0.1j], [-0.1j, 0]])[None, ...]
    >>> R2 = np.array([[0, 0.1], [0.1, 1]])[None, ...]
    >>> F = R1 / (Z[:, None, None] - 1) + R2 / (Z[:, None, None] + 2)
    >>> poles, residues, error = approx_freq_aaa(F, Z, aaa_tol=1e-12)
    >>> poles
    array([-2.  , -0.27,  1.  ])
    >>> residues
    array([[[ 0. -0.j ,  0.1+0.j ],
            [ 0.1-0.j ,  1. +0.j ]],
    <BLANKLINE>
           [[-0. -0.j , -0. -0.j ],
            [-0. +0.j , -0. +0.j ]],
    <BLANKLINE>
           [[ 1. +0.j ,  0. +0.1j],
            [ 0. -0.1j,  0. +0.j ]]])
    >>> float(error) < 1e-12
    True

    """

    if max_n_poles is None and aaa_tol is None:
        raise ValueError("At least one of `max_n_poles` or `aaa_tol` must be provided.")

    return _frequency_data_driver(F, Z, max_n_poles=max_n_poles, tol=aaa_tol, verbose=verbose)


def approx_sop_fast(
        poles, residues, beta, max_n_poles=None, aaa_tol=None,
        nonlinear_optimization=False, verbose=False):
    """Approximate a sum of simple poles defined by `poles` and `residues`
    with a sum of a (possibly) smaller number of simple poles, by running the
    AAA algorithm and, optionally, a non-linear optimization step.

    Parameters
    ----------
    poles : (K,) array_like
        Poles :math:`P_k` of the original sum of simple poles to approximate.
    residues : (K, ...) array_like
        Residues :math:`R_k` of the original sum of simple poles to approximate.
    beta : float
        Inverse temperature, used to define the L2 norm in imaginary time
        and the imaginary-frequency grid on which the AAA data is sampled.
    max_n_poles : int, optional
        Maximum number of poles to use in the approximation.
    aaa_tol : float, optional
        Error tolerance for the AAA algorithm. This is the maximum absolute
        error over the imaginary-frequency-domain data used by the AAA
        algorithm (the original sum of simple poles evaluated on the grid
        described below).
    nonlinear_optimization : bool, optional
        If True, run a non-linear optimization step after the AAA approximation,
        using the AAA poles only as an initial guess.
    verbose : bool, optional
        If True, print verbose output during the approximation process.

    Returns
    -------
    poles : (M,) ndarray
        Poles of the approximating sum of simple poles.
    residues : (M, ...) ndarray
        Residues of the approximating sum of simple poles.
    error : float
        L2 norm of the difference in imaginary time between the original
        and approximating sum of simple poles.

    Notes
    -----
    The original sum of simple poles

    .. math::
        F(Z) = \\sum_{k=1}^K \\frac{R_k}{Z - P_k}

    is first evaluated on an equispaced grid :math:`Z_n` on the imaginary-frequency
    axis (see below), to produce the frequency-domain data :math:`F_n = F(Z_n)`
    used by the AAA algorithm.

    The approximation is then built in two steps:

    1. **Pole step:** the AAA algorithm is run on this frequency-domain data
       :math:`(Z_n, F_n)` to determine the pole locations.
    2. **Residue step:** the residues are determined by minimizing the
       imaginary-time :math:`L^2(\\tau)` norm of the difference from the
       original sum of simple poles. By default this is a linear least-squares
       fit of the residues for the AAA poles; if `nonlinear_optimization` is
       True, the pole locations and residues are instead jointly optimized,
       for details see below.

    Note that, unlike `approx_freq_aaa`, the residues here are fit in
    imaginary time, not in the frequency domain.

    - If only `max_n_poles` is set, AAA runs until at most `max_n_poles` poles
      are used.
    - If only `aaa_tol` is set, AAA runs until the AAA error tolerance
      `aaa_tol` is reached.
    - If both are set, AAA stops as soon as either the AAA error tolerance
      `aaa_tol` is reached or `max_n_poles` poles are used, whichever happens
      first.

    **Frequency grid:** The frequency-domain data used by the AAA algorithm is obtained by
    evaluating the original sum of simple poles on an equispaced
    imaginary-frequency grid :math:`Z_n = i \\pi n / \\beta`, with spacing
    :math:`\\pi / \\beta`, for integer :math:`n = -n_{max}, \\ldots, n_{max}`,
    where :math:`n_{max} = \\lfloor 4 \\beta \\, \\omega_{max} / \\pi \\rfloor + 1`
    and :math:`\\omega_{max} = \\max_k |poles_k|` is the largest pole magnitude.
    This grid extends to roughly :math:`4 \\, \\omega_{max}` on the imaginary
    axis.

    **Non-linear optimization:** The optional non-linear optimization step keeps the number of poles fixed
    and jointly relocates the pole positions and refits the residues to minimize
    the imaginary-time L2 norm error between the original and approximating
    sum of simple poles. It is solved with the L-BFGS-B method (using an analytic
    gradient) and is run to optimizer convergence, i.e. to reduce the error as
    much as possible rather than to meet a prescribed error tolerance.

    Note
    ----
    The number of poles produced might be smaller than `max_n_poles`, for two reasons:
    
    1. an odd number of poles is always produced, from symmetry considerations, and 
    2. the AAA cleanup step might remove poles with small residues.

    Note
    ----
    Setting `aaa_tol` does **not** guarantee that the final imaginary-time L2 norm
    `error` is below `aaa_tol`; this is not possible to guarantee within the AAA
    algorithm alone. However, this function returns the final imaginary-time L2
    norm `error` of approximation (see Returns above).

    Examples
    --------
    Approximate a sum of three simple poles
    :math:`s(z) = 1/(z-1) + 0.5/(z+2) + 0.3/(z-0.5)`,
    specified by its poles and residues, using AAA. Here the input is already
    minimal, so the three poles are recovered (up to ordering), with the
    residues re-fit in imaginary time for inverse temperature :math:`\\beta`.

    >>> import numpy as np
    >>> np.set_printoptions(precision=2, suppress=True)
    >>> from adapol.adapol import approx_sop_fast
    >>> poles = np.array([1.0, -2.0, 0.5])
    >>> residues = np.array([1.0, 0.5, 0.3])
    >>> poles, residues, error = approx_sop_fast(poles, residues, beta=20.0, aaa_tol=1e-12)
    >>> poles
    array([-2. ,  0.5,  1. ])
    >>> residues
    array([0.5, 0.3, 1. ])
    >>> float(error) < 1e-9
    True

    Unlike `approx_freq_aaa`, the residues are fit by minimizing the
    imaginary-time :math:`L^2(\\tau)` error, so `error` is an imaginary-time
    norm rather than a frequency-domain sample error.

    Tensor-valued residues are supported as well, e.g. a 2x2 matrix-valued
    sum of poles with :math:`\\dim(R_k) = 2 \\times 2`:

    >>> R1 = np.array([[1, 0.1j], [-0.1j, 0]])
    >>> R2 = np.array([[0, 0.1], [0.1, 1]])
    >>> R3 = np.array([[0.5, 0.0], [0.0, 0.5]])
    >>> poles = np.array([1.0, -2.0, 0.3])
    >>> residues = np.array([R1, R2, R3])
    >>> poles, residues, error = approx_sop_fast(poles, residues, beta=20.0, aaa_tol=1e-12)
    >>> poles
    array([-2. ,  0.3,  1. ])
    >>> residues
    array([[[ 0. +0.j ,  0.1+0.j ],
            [ 0.1-0.j ,  1. +0.j ]],
    <BLANKLINE>
           [[ 0.5+0.j , -0. -0.j ],
            [-0. +0.j ,  0.5+0.j ]],
    <BLANKLINE>
           [[ 1. +0.j ,  0. +0.1j],
            [ 0. -0.1j, -0. +0.j ]]])
    >>> float(error) < 1e-9
    True

    """
    if max_n_poles is None and aaa_tol is None:
        raise ValueError("At least one of `max_n_poles` or `aaa_tol` must be provided.")

    return _sum_of_simple_poles_driver(
        poles, residues, max_n_poles=max_n_poles, tol=aaa_tol, beta=beta,
        nonlinear_optimization=nonlinear_optimization, verbose=verbose)


def approx_sop_tol(
        poles, residues, tol, beta, nonlinear_optimization=False, verbose=False):
    """Approximate a sum of simple poles defined by `poles` and `residues`
    with the smallest sum of simple poles whose imaginary-time
    :math:`L^2(\\tau)` error is below the tolerance `tol`.

    Parameters
    ----------
    poles : (K,) array_like
        Poles of the original sum of simple poles to approximate.
    residues : (K, ...) array_like
        Residues of the original sum of simple poles to approximate.
    tol : float
        Target tolerance on the final imaginary-time :math:`L^2(\\tau)` norm of
        the difference between the original and approximating sum of simple
        poles.
    beta : float
        Inverse temperature, used to define the L2 norm in imaginary time
        and the imaginary-frequency grid on which the AAA data is sampled.
    nonlinear_optimization : bool, optional
        If True, the residue step jointly optimizes the pole locations and
        residues (rather than fitting residues only) to minimize the
        imaginary-time :math:`L^2(\tau)` error.
    verbose : bool, optional
        If True, print verbose output during the approximation process.

    Returns
    -------
    poles : (M,) ndarray
        Poles of the approximating sum of simple poles.
    residues : (M, ...) ndarray
        Residues of the approximating sum of simple poles.
    error : float
        L2 norm of the difference in imaginary time between the original
        and approximating sum of simple poles.

    Raises
    ------
    ValueError
        If the target tolerance `tol` cannot be achieved within the internal
        maximum number of search steps.

    Notes
    -----
    Unlike `approx_sop_fast`, where the tolerance only controls the AAA
    pole step, here `tol` is imposed on the **final** imaginary-time error,
    i.e. after the residues have been fit. Since AAA only determines pole
    locations and the residues (and hence the final error) are only known after
    the residue step, the requested error cannot be reached by a single AAA run.
    Instead, the AAA + residue pipeline is run repeatedly to search for the
    minimal number of poles that achieves `tol`.

    Each candidate fit is built in two steps:

    1. **Pole step:** the original sum of simple poles is evaluated on an
       equispaced imaginary-frequency grid (see `approx_sop_fast` for the
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

    Examples
    --------
    Compress a continuous spectral density into a small sum of simple poles.
    The density is first discretized as a sum of 200 simple poles on the real
    axis, which is then compressed to the smallest sum of poles whose
    imaginary-time :math:`L^2(\\tau)` error is below `tol`.

    >>> import numpy as np
    >>> from adapol.adapol import approx_sop_tol
    >>> w = np.linspace(-2, 2, 200)                            # real-frequency grid
    >>> dw = w[1] - w[0]
    >>> rho = np.sqrt(np.maximum(4 - w**2, 0.0)) / (2 * np.pi)  # semicircle density
    >>> poles, residues, error = approx_sop_tol(w, rho * dw, tol=1e-5, beta=20.0)
    >>> len(poles) < 30      # 200 input poles compressed to a handful
    True
    >>> float(error) < 1e-5  # final imaginary-time error is below the tolerance
    True

    Unlike `approx_sop_fast`, the number of poles is not prescribed but
    chosen automatically as the smallest count meeting `tol` on the final
    imaginary-time error.

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