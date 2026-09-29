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
        Tolerance on the AAA residual, i.e. on the pole step. It does not bound
        the final error, which also depends on the subsequent residue fit.
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
    - If only `aaa_tol` is set, AAA runs until the AAA residual tolerance
      `aaa_tol` is reached.
    - If both are set, AAA stops as soon as either the AAA residual tolerance
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
        Tolerance on the AAA residual. This is the maximum absolute deviation
        from the imaginary-frequency-domain data used by the AAA algorithm (the
        DLR expansion evaluated on the grid described below), over the sample
        points not yet used as AAA support points.
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
    - If only `aaa_tol` is set, AAA runs until the AAA residual tolerance
      `aaa_tol` is reached.
    - If both are set, AAA stops as soon as either the AAA residual tolerance
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


def approx_gf_dlr_tol(G_dlr, tol, nonlinear_optimization=False,
                      restrict_to_dlr_window=True, verbose=False):
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
    restrict_to_dlr_window : bool, optional
        If True (default), drop the poles outside the DLR window
        :math:`|\\beta \\omega| \\le \\beta \\omega_{\\max}` of the mesh of
        `G_dlr` and refit the residues of the remaining poles, see
        `TriqsDLRCompression`.
    verbose : int or bool, optional
        Amount of printed output. 0 (or False) is silent, 1 (or True) prints one
        line per pass through the AAA and residue fit pipeline, showing the pole
        count and error of each candidate fit, and 2 additionally prints the
        indented per step output of the AAA algorithm itself.

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
    RuntimeError
        If `restrict_to_dlr_window` is True and a pole outside the DLR window
        cannot be dropped without changing the approximation, see
        `TriqsDLRCompression`.

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

    The conjugate-pair constrained AAA always produces an odd number of poles,
    and the surplus pole has a numerically zero residue at a location fixed only
    by round-off, often far outside the DLR window, where it is not
    representable in the DLR basis of `G_dlr`. With `restrict_to_dlr_window`
    such poles are dropped and the remaining residues refit, and the returned
    `error` is that of the returned poles and residues. A surplus pole that
    lands inside the window is kept, it is representable and harmless there.

    See Also
    --------
    adapol.approx_sop_tol : Underlying array-based routine.
    approx_gf_dlr_fast : Single-pass compression with an AAA stopping criterion.
    """
    comp = TriqsDLRCompression(
        G_dlr, tol=tol, nonlinear_optimize=nonlinear_optimization,
        nonlinear_post_optimize=nonlinear_optimization,
        restrict_to_dlr_window=restrict_to_dlr_window, verbose=verbose)
    return comp.poles, comp.residues, comp.error


def _gf_imfreq_to_data(G_w):
    Z = np.array([complex(w) for w in G_w.mesh])
    F = G_w.data.copy()
    return Z, F


def _gf_dlr_to_data(G_dlr):
    from triqs.gfs import MeshDLR
    from triqs.gfs import MeshDLRImFreq
    from triqs.gfs import MeshDLRImTime
    from triqs.gfs import make_gf_dlr
    from triqs.gfs import make_gf_dlr_imfreq

    if type(G_dlr.mesh) not in [MeshDLR, MeshDLRImFreq, MeshDLRImTime]:
        raise ValueError('G_dlr must be defined on a DLR mesh')

    G_c = G_dlr if type(G_dlr.mesh) is MeshDLR else make_gf_dlr(G_dlr)

    beta = G_c.mesh.beta
    poles = np.array([float(w) for w in G_c.mesh]) / beta
    residues = G_c.data.copy()

    G_w = make_gf_dlr_imfreq(G_c)
    Z = np.array([complex(w) for w in G_w.mesh])

    return poles, residues, beta, Z


class TriqsDLRCompression:

    """Compress a Green's function :math:`G` given in the discrete Lehmann
    representation (DLR) to the smallest sum of simple poles whose
    imaginary-time :math:`L^2(\\tau)` error is below the tolerance `tol`, with
    all poles inside the DLR window of the mesh of `G`.

    This is the TRIQS front-end to `adapol.sop_compr.SumOfPolesCompression`,
    taking the original poles and residues from the DLR frequencies and
    coefficients of `G`, and the AAA sample points from its DLR Matsubara
    nodes. See `approx_gf_dlr_tol` for a description of the tolerance search.

    Parameters
    ----------
    G : triqs.gf.Gf
        Green's function to compress, defined on a DLR mesh (`MeshDLR`,
        `MeshDLRImFreq` or `MeshDLRImTime`).
    tol : float, optional
        Target tolerance on the final imaginary-time :math:`L^2(\\tau)` error.
    nonlinear_optimize : bool, optional
        If True, the residue step jointly optimizes the pole locations and
        residues, rather than fitting the residues only.
    nonlinear_post_optimize : bool, optional
        If True, retry the result of the search with one pole less using the
        joint pole and residue optimization.
    max_upwind_steps : int, optional
        Maximum number of passes that grow the AAA step count before the
        tolerance must be met.
    restrict_to_dlr_window : bool, optional
        If True (default), drop the poles outside the DLR window
        :math:`|\\beta \\omega| \\le \\Lambda = \\beta \\omega_{\\max}` of the
        mesh of `G` and refit the residues of the remaining poles (see Notes).
    verbose : int or bool, optional
        Amount of printed output, see `approx_gf_dlr_tol`.

    Attributes
    ----------
    poles : (M,) ndarray
        Poles of the approximating sum of simple poles.
    residues : (M, ...) ndarray
        Residues of the approximating sum of simple poles.
    error : float
        Normalized imaginary-time :math:`L^2(\\tau)` norm of the difference
        between the DLR expansion of `G` and the approximating sum of simple
        poles, for the returned `poles` and `residues`.
    aaa_steps : int
        Number of AAA steps of the returned approximation.
    Lambda : float
        The DLR window :math:`\\Lambda = \\beta \\omega_{\\max}` of the mesh.
    dropped_poles : (K,) ndarray
        Poles dropped for lying outside the DLR window (empty if none).
    dropped_residues : (K, ...) ndarray
        Residues of the dropped poles, before the refit.
    error_before_window : float
        Error of the approximation before the out-of-window poles were dropped.
    window_residual : float
        Maximum absolute change of the approximation on the DLR
        imaginary-time nodes caused by dropping and refitting (zero if no
        pole was dropped).

    Raises
    ------
    ValueError
        If the tolerance search fails to reach `tol`, or if `G` is not defined
        on a DLR mesh.
    RuntimeError
        If the DLR window restriction would drop every pole, or if a dropped
        pole carried real weight: after the refit the error is not below
        `tol`, or the pointwise deviation from the DLR expansion on the DLR
        imaginary-time nodes exceeds both 10 times that of the fit before the
        drop and `tol`.

    Notes
    -----
    The conjugate-pair constrained AAA always produces an odd number of poles,
    and the surplus pole has a numerically zero residue. Since it does not
    contribute to the fit, its location is fixed only by round-off, and it
    routinely lands far outside the DLR window, where it is not representable
    in a DLR basis built for :math:`\\Lambda`. Any round trip of the
    approximation through that basis (coefficients to values and back, a
    :math:`\\tau` reflection, a convolution) then silently loses the small
    residue that round-off leaves on it, with an error that is amplified by
    orders of magnitude.

    With `restrict_to_dlr_window`, every pole with
    :math:`|\\beta \\omega| > \\Lambda (1 + s)` is dropped and the residues of
    the remaining poles are refit to the DLR expansion of `G` by a linear
    least-squares fit in :math:`L^2(\\tau)`, keeping the remaining pole
    locations fixed (also with `nonlinear_optimize`), and `error` is
    recomputed for the result. The relative slack
    :math:`s = \\max(10^{-9}, 50 \\epsilon)`, with :math:`\\epsilon` the DLR
    accuracy of the mesh, keeps a physical pole located at
    :math:`\\omega_{\\max}`, which may come back from the fit slightly
    outside the window, while the overshoot of the surplus pole is typically
    orders of magnitude larger.

    Since a dropped pole contributes about its residue at :math:`\\tau \\to 0^+`
    however far out it sits, while its :math:`L^2(\\tau)` weight vanishes, the
    result is also checked pointwise against the DLR expansion on the DLR
    imaginary-time nodes, raising `RuntimeError` if the drop was not
    harmless.

    The window restriction does not remove a surplus pole that happens to land
    inside the window. Such a pole is representable in the DLR basis and
    harmless there, but the pole count may still include it.
    """

    # floor and eps multiplier of the relative slack on the DLR window, see `window_slack`
    window_slack_floor = 1e-9
    window_slack_per_eps = 50.0

    def __init__(self, G, tol=1e-14, 
                 nonlinear_optimize=False, nonlinear_post_optimize=False, 
                 max_upwind_steps=4, restrict_to_dlr_window=True, verbose=True):

        self.G = G
        self.tol = tol
        self.nonlinear_optimize = nonlinear_optimize
        self.nonlinear_post_optimize = nonlinear_post_optimize
        self.restrict_to_dlr_window = restrict_to_dlr_window
        self.verbose = verbose

        from triqs.gfs import MeshDLR
        from triqs.gfs import MeshDLRImFreq
        from triqs.gfs import MeshDLRImTime
        from triqs.gfs import make_gf_dlr

        if type(G.mesh) not in [MeshDLR, MeshDLRImFreq, MeshDLRImTime]:
            raise ValueError('G must be defined on a DLR mesh')

        self.G_dlr = G if type(G.mesh) is MeshDLR else make_gf_dlr(G)
        self.dlr_freq = np.array([float(w) for w in self.G_dlr.mesh])
        self.G_dlr_coeff = self.G_dlr.data.copy()
        self.beta = self.G_dlr.mesh.beta
        self.Lambda = self.beta * self.G_dlr.mesh.w_max

        poles = self.dlr_freq / self.beta
        residues = self.G_dlr_coeff.copy()

        from triqs.gfs import make_gf_dlr_imfreq

        self.G_w = G if type(G.mesh) is MeshDLRImFreq else make_gf_dlr_imfreq(G)
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

        self.error_before_window = self.error
        self.dropped_poles = self.poles[:0].copy()
        self.dropped_residues = self.residues[:0].copy()
        self.window_residual = 0.

        if restrict_to_dlr_window:
            self._restrict_poles_to_dlr_window()


    @property
    def window_slack(self):

        """ Relative slack on the DLR window below which an out-of-window pole is taken as physics.

        A pole physically at w_max may come back from the fit slightly outside the window, while the
        overshoot of the surplus pole is typically orders of magnitude larger. The eps term is a
        conservative bound carried over from triqs_xca. The pointwise check in `_restrict_poles_to_dlr_window` is the safety net
        for the cases where this estimate is wrong. """

        return max(self.window_slack_floor, self.window_slack_per_eps * self.G_dlr.mesh.eps)


    def _restrict_poles_to_dlr_window(self):

        """ Drop the poles outside the DLR window and least-squares refit the survivors.

        The fit constrains the error and not the pole locations, and a pole at |beta*omega| > Lambda
        is not representable in a DLR basis built for Lambda. The out-of-window pole usually carries
        numerical dust but can carry weight, so the survivors are refit to the DLR expansion and the
        result is checked against it, in L2 and pointwise, raising if a dropped pole was load-bearing. """

        poles, residues = np.asarray(self.poles), np.asarray(self.residues)
        beta_omega = self.beta * poles
        outside = np.abs(beta_omega) > self.Lambda * (1 + self.window_slack)

        if not np.any(outside):
            return

        keep = ~outside
        n_keep = int(np.sum(keep))

        if n_keep == 0:
            raise RuntimeError(
                f'TriqsDLRCompression: every one of the {len(poles)} fitted poles lies outside '
                f'the DLR window (max|beta*omega| = {np.max(np.abs(beta_omega)):2.2E} vs '
                f'Lambda = {self.Lambda:2.2E}), so the fit cannot be repaired. Widen w_max, '
                f'or pass restrict_to_dlr_window=False.')

        from triqs.gfs import make_gf_dlr_imtime
        from .sop import SumOfSimplePoles

        dlr_sop = self.sop_comp.sop
        fit = SumOfSimplePoles(poles=poles, residues=residues)
        refit = dlr_sop.best_imtime_lstsq_l2_norm_approximation_using_poles(poles[keep], self.beta)

        # Pointwise on the DLR tau nodes: a dropped pole contributes ~ -R_far at tau -> 0+ however far
        # out it sits, while its L2 mass and its least-squares footprint on the survivors vanish in that limit.
        # Both fits are measured against the DLR expansion, since the pointwise error of a fit can be
        # an order of magnitude above its L2 error, which makes an L2 based threshold misfire.
        tau = np.array([float(t) for t in make_gf_dlr_imtime(self.G_dlr).mesh])
        g_tau = dlr_sop.eval_imtime(tau, self.beta)
        fit_tau, refit_tau = fit.eval_imtime(tau, self.beta), refit.eval_imtime(tau, self.beta)
        residual = np.max(np.abs(fit_tau - refit_tau))
        dev_fit = np.max(np.abs(g_tau - fit_tau))
        dev_refit = np.max(np.abs(g_tau - refit_tau))
        error = (dlr_sop - refit).imtime_l2_norm(beta=self.beta)

        # a refit well off the DLR expansion, or above tol, means a dropped pole was load-bearing
        if dev_refit > max(10 * dev_fit, self.tol) or error >= self.tol:
            omega_out = poles[outside]
            beta_omega_out = beta_omega[outside]
            overshoot_out = np.abs(beta_omega_out) / self.Lambda - 1
            weight_out = np.abs(residues[outside]).reshape(len(omega_out), -1).max(axis=1)
            pole_info = ', '.join(
                f'omega={om:2.8E} (|beta*omega|={abs(bw):2.8E}, overshoot={ov:2.3E} rel. to '
                f'Lambda, residue abs-max={w:2.2E})'
                for om, bw, ov, w in zip(omega_out, beta_omega_out, overshoot_out, weight_out))
            raise RuntimeError(
                f'TriqsDLRCompression: dropping {len(poles) - n_keep} out-of-window pole(s) and '
                f'refitting gives a pointwise deviation {dev_refit:2.2E} from the DLR expansion '
                f'(against {dev_fit:2.2E} before the drop) and an error {error:2.2E} (against '
                f'{self.error_before_window:2.2E} before the drop, tol = {self.tol:2.2E}). The '
                f'dropped pole carried real weight, which the poles inside the DLR window cannot '
                f'absorb. w_max = {self.G_dlr.mesh.w_max:2.4E}, Lambda = beta*w_max = '
                f'{self.Lambda:2.2E}, eps = {self.G_dlr.mesh.eps:2.2E}, window_slack = '
                f'{self.window_slack:2.2E}; dropped pole(s): {pole_info}. If the pole sits AT the '
                f'window edge, raise eps or w_max so the fit resolves it; otherwise widen w_max, or '
                f'pass restrict_to_dlr_window=False.')

        self.dropped_poles = poles[outside].copy()
        self.dropped_residues = residues[outside].copy()
        self.window_residual = residual
        self.poles = poles[keep].copy()
        self.residues = np.ascontiguousarray(refit.R)
        self.error = error

        if self.verbose:
            print(f'Adapol: dropped {len(poles) - n_keep} of {len(poles)} poles outside the DLR '
                  f'window (max|beta*omega| = {np.max(np.abs(beta_omega)):2.2E} > Lambda = '
                  f'{self.Lambda:2.2E}) and refit: {n_keep} poles, error {self.error:2.2E}, '
                  f'pointwise change {residual:2.2E}')
