"""Adapol/TRIQS: Adaptive Pole Approximation of Frequency Data

User facing TRIQS based API for approximating frequency data 
with a sum of simple poles, using the AAA algorithm.

Author: Hugo U. R. Strand (2026)"""

import numpy as np


from .adapol import _frequency_data_driver
from .adapol import _sum_of_simple_poles_driver


def approx_gf_imfreq_aaa(G_w, max_n_poles=None, aaa_tol=None, verbose=False):
    """Approximate a Green's function :math:`G` given on an imaginary-frequency
    mesh with a sum of simple poles, by running the AAA algorithm.

    This is the TRIQS front-end to `adapol.approx_freq_aaa`, taking the
    frequency data :math:`F` and the sample points :math:`Z` from the Green's
    function and its mesh.

    Parameters
    ----------
    G_w : triqs.gf.Gf
        Green's function to approximate, defined on an imaginary-frequency
        mesh (`MeshImFreq` or `MeshDLRImFreq`).
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
        Maximum absolute error of the approximation at the mesh points.

    Raises
    ------
    ValueError
        If neither `max_n_poles` nor `aaa_tol` is given.

    Notes
    -----
    - If only `max_n_poles` is set, AAA runs until at most `max_n_poles` poles
      are used.
    - If only `aaa_tol` is set, AAA runs until the AAA error tolerance
      `aaa_tol` is reached.
    - If both are set, AAA stops as soon as either the AAA error tolerance
      `aaa_tol` is reached or `max_n_poles` poles are used, whichever happens
      first.

    See `adapol.approx_freq_aaa` for details on the algorithm, on why the
    number of poles produced may be smaller than `max_n_poles`, and on why
    `aaa_tol` does not bound the returned `error`.

    See Also
    --------
    adapol.approx_freq_aaa : Underlying array-based routine.
    """
    if max_n_poles is None and aaa_tol is None:
        raise ValueError("At least one of `max_n_poles` or `aaa_tol` must be provided.")

    Z, F = _gf_imfreq_to_data(G_w)
    return _frequency_data_driver(
        F, Z, max_n_poles=max_n_poles, tol=aaa_tol, verbose=verbose)


def approx_gf_dlr_fast(
        G_dlr, max_n_poles=None, aaa_tol=None,
        nonlinear_optimization=False, verbose=False):
    """Approximate a Green's function :math:`G` given in the discrete Lehmann
    representation (DLR) with a sum of a (possibly) smaller number of simple
    poles, by running the AAA algorithm and, optionally, a non-linear
    optimization step.

    This is the TRIQS front-end to `adapol.approx_sop_fast`, taking the
    original poles and residues from the DLR frequencies and coefficients of
    the Green's function.

    Parameters
    ----------
    G_dlr : triqs.gf.Gf
        Green's function to approximate, defined on a DLR mesh (`MeshDLR`,
        `MeshDLRImFreq` or `MeshDLRImTime`).
    max_n_poles : int, optional
        Maximum number of poles to use in the approximation.
    aaa_tol : float, optional
        Error tolerance for the AAA algorithm. This is the maximum absolute
        error over the imaginary-frequency-domain data used by the AAA
        algorithm (the DLR expansion evaluated on the grid described below).
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
        Normalized imaginary-time :math:`L^2(\\tau)` norm of the difference
        between the original DLR expansion and the approximating sum of simple
        poles (see Notes).

    Raises
    ------
    ValueError
        If neither `max_n_poles` nor `aaa_tol` is given, or if `G_dlr` is not
        defined on a DLR mesh.

    Notes
    -----
    The DLR expansion of :math:`G` is a sum of simple poles

    .. math::
        G(Z) = \\sum_{k=1}^K \\frac{R_k}{Z - P_k}

    with poles :math:`P_k = \\omega_k / \\beta` given by the DLR frequencies
    :math:`\\omega_k` of the mesh, and residues :math:`R_k` given by the DLR
    coefficients of `G_dlr`. The approximation is then built in two steps:

    1. **Pole step:** the DLR expansion is evaluated on the DLR
       imaginary-frequency nodes and AAA is run on this data to determine the
       pole locations. Note that this is the DLR Matsubara node set of the
       mesh of `G_dlr`, and not the equispaced fermionic Matsubara grid that
       `adapol.approx_sop_fast` uses by default.
    2. **Residue step:** the residues are determined by minimizing the
       imaginary-time :math:`L^2(\\tau)` norm of the difference from the DLR
       expansion. By default this is a linear least-squares fit of the
       residues for the AAA poles; if `nonlinear_optimization` is True, the
       pole locations and residues are instead jointly optimized.

    The imaginary-time :math:`L^2(\\tau)` norm is normalized by the inverse
    temperature :math:`\\beta`,

    .. math::
        \\lVert f \\rVert_{L^2(\\tau)} =
            \\left( \\frac{1}{\\beta} \\int_0^\\beta |f(\\tau)|^2 \\, d\\tau \\right)^{1/2}.

    - If only `max_n_poles` is set, AAA runs until at most `max_n_poles` poles
      are used.
    - If only `aaa_tol` is set, AAA runs until the AAA error tolerance
      `aaa_tol` is reached.
    - If both are set, AAA stops as soon as either the AAA error tolerance
      `aaa_tol` is reached or `max_n_poles` poles are used, whichever happens
      first.

    Note
    ----
    Setting `aaa_tol` does **not** guarantee that the returned imaginary-time
    L2 norm `error` is below `aaa_tol`. Use `approx_gf_dlr_tol` to impose a
    tolerance on the final error instead.

    See Also
    --------
    adapol.approx_sop_fast : Underlying array-based routine.
    approx_gf_dlr_tol : Smallest approximation meeting a final error tolerance.
    """
    if max_n_poles is None and aaa_tol is None:
        raise ValueError("At least one of `max_n_poles` or `aaa_tol` must be provided.")

    poles, residues, beta, Z = _gf_dlr_to_data(G_dlr)
    return _sum_of_simple_poles_driver(
        poles, residues, max_n_poles=max_n_poles, tol=aaa_tol, beta=beta,
        nonlinear_optimization=nonlinear_optimization, Z=Z, verbose=verbose)


def approx_gf_dlr_tol(G_dlr, tol, nonlinear_optimization=False, verbose=False):
    """Approximate a Green's function :math:`G` given in the discrete Lehmann
    representation (DLR) with the smallest sum of simple poles whose
    imaginary-time :math:`L^2(\\tau)` error is below the tolerance `tol`.

    This is the TRIQS front-end to `adapol.approx_sop_tol`, taking the
    original poles and residues from the DLR frequencies and coefficients of
    the Green's function.

    Parameters
    ----------
    G_dlr : triqs.gf.Gf
        Green's function to approximate, defined on a DLR mesh (`MeshDLR`,
        `MeshDLRImFreq` or `MeshDLRImTime`).
    tol : float
        Target tolerance on the final imaginary-time :math:`L^2(\\tau)` norm of
        the difference between the original DLR expansion and the approximating
        sum of simple poles.
    nonlinear_optimization : bool, optional
        If True, the residue step jointly optimizes the pole locations and
        residues (rather than fitting residues only) to minimize the
        imaginary-time :math:`L^2(\\tau)` error.
    verbose : bool, optional
        If True, print verbose output during the approximation process.

    Returns
    -------
    poles : (M,) ndarray
        Poles of the approximating sum of simple poles.
    residues : (M, ...) ndarray
        Residues of the approximating sum of simple poles.
    error : float
        Normalized imaginary-time :math:`L^2(\\tau)` norm of the difference
        between the original DLR expansion and the approximating sum of simple
        poles (with the :math:`1/\\beta` normalization defined in
        `approx_gf_dlr_fast`).

    Raises
    ------
    ValueError
        If the target tolerance `tol` cannot be achieved within the internal
        maximum number of search steps, or if `G_dlr` is not defined on a DLR
        mesh.

    Notes
    -----
    Unlike `approx_gf_dlr_fast`, where the tolerance only controls the AAA
    pole step, here `tol` is imposed on the **final** imaginary-time error,
    i.e. after the residues have been fit. Since the residues (and hence the
    final error) are only known after the residue step, the AAA + residue
    pipeline is run repeatedly to search for the minimal number of poles that
    achieves `tol`: the number of AAA poles is first increased until the
    imaginary-time error drops below `tol`, then a bisection on the number of
    AAA steps locates the smallest pole count that still meets `tol`.

    As in `approx_gf_dlr_fast`, the AAA data is the DLR expansion evaluated on
    the DLR imaginary-frequency nodes of the mesh of `G_dlr`.

    See Also
    --------
    adapol.approx_sop_tol : Underlying array-based routine.
    approx_gf_dlr_fast : Single-pass compression with an AAA stopping criterion.
    """
    comp = TriqsDLRCompression(
        G_dlr, tol=tol, nonlinear_optimize=nonlinear_optimization,
        nonlinear_post_optimize=nonlinear_optimization, verbose=verbose)
    return comp.poles, comp.residues, comp.error


def _gf_imfreq_to_data(G_w):
    Z = np.array([complex(w) for w in G_w.mesh])
    F = G_w.data.copy()
    return Z, F


def _gf_dlr_to_data(G_dlr):
    from triqs.gfs import MeshDLR, MeshDLRImFreq, MeshDLRImTime
    from triqs.gfs import make_gf_dlr, make_gf_dlr_imfreq

    if type(G_dlr.mesh) not in [MeshDLR, MeshDLRImFreq, MeshDLRImTime]:
        raise ValueError('G_dlr must be defined on a DLR mesh')

    G_c = G_dlr if type(G_dlr.mesh) == MeshDLR else make_gf_dlr(G_dlr)

    beta = G_c.mesh.beta
    poles = np.array([float(w) for w in G_c.mesh]) / beta
    residues = G_c.data.copy()

    G_w = make_gf_dlr_imfreq(G_c)
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
