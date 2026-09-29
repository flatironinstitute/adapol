""" Test Adapol's TRIQS API.

Author: Hugo U. R. Strand (2026)"""

import pytest

pytest.importorskip(
    "triqs",
    reason="Triqs is not installed. Skipping test_triqs_xca. "
           "Please ensure that it is installed to run the entire test suite."
)


import numpy as np
from triqs.gfs import Gf
from triqs.gfs import MeshDLRImFreq
from triqs.gfs import MeshImFreq
from triqs.gfs import SemiCircular
from triqs.gfs import inverse
from triqs.gfs import iOmega_n
from triqs.gfs import make_gf_dlr
from triqs.gfs import make_gf_dlr_imtime

from adapol.triqs import TriqsDLRCompression
from adapol.triqs import approx_gf_dlr_fast
from adapol.triqs import approx_gf_dlr_tol
from adapol.triqs import approx_gf_imfreq_aaa


def test_gf_imfreq_n_poles(max_n_poles=5):

    m = MeshImFreq(beta=100.0, statistic='Fermion', n_iw=1000)

    G_iw = Gf(mesh=m, target_shape=[])
    G_iw << inverse(iOmega_n - SemiCircular(1.0))

    poles, residues, diff = approx_gf_imfreq_aaa(
        G_iw, max_n_poles=max_n_poles, verbose=True)

    print(f'max_n_poles = {max_n_poles}, n_poles = {len(poles)}')
    assert( len(poles) <= max_n_poles )
    print(f'Max difference between G_iw and its approximation = {diff:2.2E}')


def test_gf_imfreq_tol(tol=1e-8):

    m = MeshImFreq(beta=100.0, statistic='Fermion', n_iw=1000)

    G_iw = Gf(mesh=m, target_shape=[])
    G_iw << inverse(iOmega_n - SemiCircular(1.0))

    poles, residues, diff = approx_gf_imfreq_aaa(
        G_iw, aaa_tol=tol, verbose=True)

    print(f'diff = {diff:2.2E}, tol = {tol}, n_poles = {len(poles)}')

    assert( diff < tol )


def test_gf_dlr_n_poles(max_n_poles=5, nonlinear_optimization=False):

    m = MeshDLRImFreq(beta=10.0, statistic='Fermion', eps=1e-12, w_max=10.0)

    G_iw = Gf(mesh=m, target_shape=[])
    G_iw << inverse(iOmega_n - SemiCircular(1.0))

    G_dlr = make_gf_dlr(G_iw)

    poles, residues, diff = approx_gf_dlr_fast(
        G_dlr, max_n_poles=max_n_poles, verbose=True, nonlinear_optimization=nonlinear_optimization)

    print(f'max_n_poles = {max_n_poles}, n_poles = {len(poles)}')
    assert( len(poles) <= max_n_poles )
    print(f'diff = {diff:2.2E}, n_poles = {len(poles)}')


def test_gf_dlr_tol(tol=1e-8, nonlinear_optimization=False):

    m = MeshDLRImFreq(beta=10.0, statistic='Fermion', eps=1e-12, w_max=10.0)

    G_iw = Gf(mesh=m, target_shape=[])
    G_iw << inverse(iOmega_n - SemiCircular(1.0))

    G_dlr = make_gf_dlr(G_iw)

    poles, residues, diff = approx_gf_dlr_fast(
        G_dlr, aaa_tol=tol, verbose=True, nonlinear_optimization=nonlinear_optimization)

    print(f'diff = {diff:2.2E}, tol = {tol}, n_poles = {len(poles)}')

    assert( diff < tol )


def test_gf_dlr_tol_imtime(tol=1e-8, nonlinear_optimization=False):

    m = MeshDLRImFreq(beta=10.0, statistic='Fermion', eps=1e-12, w_max=10.0)

    G_iw = Gf(mesh=m, target_shape=[])
    G_iw << inverse(iOmega_n - SemiCircular(1.0))

    G_dlr = make_gf_dlr(G_iw)

    poles, residues, diff = approx_gf_dlr_tol(
        G_dlr, tol=tol, verbose=True, nonlinear_optimization=nonlinear_optimization)

    print(f'diff = {diff:2.2E}, tol = {tol}, n_poles = {len(poles)}')

    assert( diff < tol )



def test_gf_imfreq_n_poles_and_tol(max_n_poles=6, tol=1e-14):

    """ With both stopping criteria set, AAA stops at whichever is hit first,
    here the pole budget, since the tolerance is unreachable. """

    m = MeshImFreq(beta=100.0, statistic='Fermion', n_iw=1000)

    G_iw = Gf(mesh=m, target_shape=[])
    G_iw << inverse(iOmega_n - SemiCircular(1.0))

    poles, residues, diff = approx_gf_imfreq_aaa(
        G_iw, max_n_poles=max_n_poles, aaa_tol=tol, verbose=True)

    print(f'max_n_poles = {max_n_poles}, n_poles = {len(poles)}, diff = {diff:2.2E}')

    assert( len(poles) <= max_n_poles )


def test_gf_dlr_n_poles_and_tol(max_n_poles=4, tol=1e-14):

    """ With both stopping criteria set, AAA stops at whichever is hit first,
    here the pole budget, since the tolerance is unreachable. """

    m = MeshDLRImFreq(beta=10.0, statistic='Fermion', eps=1e-12, w_max=10.0)

    G_iw = Gf(mesh=m, target_shape=[])
    G_iw << inverse(iOmega_n - SemiCircular(1.0))

    G_dlr = make_gf_dlr(G_iw)

    poles, residues, diff = approx_gf_dlr_fast(
        G_dlr, max_n_poles=max_n_poles, aaa_tol=tol, verbose=True)

    print(f'max_n_poles = {max_n_poles}, n_poles = {len(poles)}, diff = {diff:2.2E}')

    assert( len(poles) <= max_n_poles )


def test_gf_dlr_mesh_types(tol=1e-8):

    """ The DLR routines accept any DLR mesh (coefficient, Matsubara or
    imaginary time) and give the same approximation. """

    m = MeshDLRImFreq(beta=10.0, statistic='Fermion', eps=1e-12, w_max=10.0)

    G_iw = Gf(mesh=m, target_shape=[])
    G_iw << inverse(iOmega_n - SemiCircular(1.0))

    G_dlr = make_gf_dlr(G_iw)
    G_tau = make_gf_dlr_imtime(G_dlr)

    ref_fast = approx_gf_dlr_fast(G_dlr, aaa_tol=tol)
    ref_tol = approx_gf_dlr_tol(G_dlr, tol=tol)

    for G in [G_iw, G_tau]:

        poles, residues, diff = approx_gf_dlr_fast(G, aaa_tol=tol)
        np.testing.assert_array_almost_equal(poles, ref_fast[0])
        np.testing.assert_array_almost_equal(residues, ref_fast[1])

        poles, residues, diff = approx_gf_dlr_tol(G, tol=tol)
        np.testing.assert_array_almost_equal(poles, ref_tol[0])
        np.testing.assert_array_almost_equal(residues, ref_tol[1])


def test_gf_dlr_requires_dlr_mesh():

    """ The DLR routines reject Green's functions on a non-DLR mesh. """

    m = MeshImFreq(beta=100.0, statistic='Fermion', n_iw=1000)

    G_iw = Gf(mesh=m, target_shape=[])
    G_iw << inverse(iOmega_n - SemiCircular(1.0))

    with pytest.raises(ValueError):
        approx_gf_dlr_fast(G_iw, aaa_tol=1e-8)


def test_gf_missing_stopping_criterion():

    """ The AAA based routines require at least one of `max_n_poles`/`aaa_tol`. """

    m = MeshDLRImFreq(beta=10.0, statistic='Fermion', eps=1e-12, w_max=10.0)

    G_iw = Gf(mesh=m, target_shape=[])
    G_iw << inverse(iOmega_n - SemiCircular(1.0))

    G_dlr = make_gf_dlr(G_iw)

    with pytest.raises(ValueError):
        approx_gf_imfreq_aaa(G_iw)

    with pytest.raises(ValueError):
        approx_gf_dlr_fast(G_dlr)


def test_tdc_tol_sweep():

    m = MeshDLRImFreq(beta=100.0, statistic='Fermion', eps=1e-14, w_max=8.0)

    G_w = Gf(mesh=m, target_shape=[2, 2])
    G_w << inverse(iOmega_n - 0.4 - SemiCircular(1.0))

    for tol in 10.**(-np.arange(2, 12)):
        print(f"Testing TriqsDLRCompression with tol = {tol:+2.2E}")
        tdc = TriqsDLRCompression(G_w, tol=tol, nonlinear_post_optimize=False)
        tdc_pstopt = TriqsDLRCompression(G_w, tol=tol)
        tdc_nonlin = TriqsDLRCompression(G_w, tol=tol, nonlinear_optimize=True)
        print(f'tdc_lstsq  error = {tdc.error:2.2E}, aaa_steps = {tdc.aaa_steps}')
        print(f'tdc_nonlin error = {tdc_nonlin.error:2.2E}, aaa_steps = {tdc_nonlin.aaa_steps}')
        print(f'tdc_pstopt error = {tdc_pstopt.error:2.2E}, aaa_steps = {tdc_pstopt.aaa_steps}')
        assert( tdc.error < tol)
        assert( tdc_nonlin.error < tol)
        assert( tdc_pstopt.error < tol)



def _two_pole_gf(a, b, beta=2.1, w_max=2.0, eps=1e-12, ra=0.6, rb=0.4):

    """ G(z) = ra/(z - a) + rb/(z - b) on a DLR Matsubara mesh. """

    m = MeshDLRImFreq(beta=beta, statistic='Fermion', w_max=w_max, eps=eps)
    G_w = Gf(mesh=m, target_shape=[1, 1])
    Z = np.array([complex(w) for w in G_w.mesh])
    G_w.data[:, 0, 0] = ra / (Z - a) + rb / (Z - b)
    return G_w


def _imtime_l2_error(tdc):

    from adapol.sop import SumOfSimplePoles
    sop = SumOfSimplePoles(poles=np.asarray(tdc.poles), residues=np.asarray(tdc.residues))
    return (tdc.sop_comp.sop - sop).imtime_l2_norm(beta=tdc.beta)


def test_tdc_dlr_window():

    """ The compressed poles stay inside the DLR window, the reported error is the
    error of the returned poles, and the tolerance is met. The surplus pole of the
    conjugate-pair constrained AAA lands at a round-off determined location, which
    for most of these inputs is far outside the window. """

    beta, w_max, tol = 2.1, 2.0, 1e-9
    n_dropped = 0

    for a in np.arange(-1.9, 2.0, 0.4):
        for b in np.arange(a + 0.2, 2.0, 0.4):

            tdc = TriqsDLRCompression(_two_pole_gf(a, b, beta=beta, w_max=w_max), tol=tol, verbose=False)
            bw = tdc.beta * np.asarray(tdc.poles)

            assert tdc.Lambda == beta * w_max
            assert np.all(np.abs(bw) <= tdc.Lambda * (1 + tdc.window_slack)), \
                f'a = {a:+.1f}, b = {b:+.1f}: pole outside the DLR window, beta*omega = {bw}'
            assert np.all(np.abs(beta * tdc.dropped_poles) > tdc.Lambda * (1 + tdc.window_slack))
            assert len(tdc.poles) + len(tdc.dropped_poles) == len(tdc.sop_comp.poles)
            assert len(tdc.dropped_poles) == len(tdc.dropped_residues)
            assert tdc.error < tol, f'a = {a:+.1f}, b = {b:+.1f}: error {tdc.error:2.2E} >= tol'
            np.testing.assert_allclose(tdc.error, _imtime_l2_error(tdc), rtol=1e-6, atol=1e-15)

            n_dropped += len(tdc.dropped_poles)

    assert n_dropped > 0, 'vacuous: no pole was ever outside the window, so the filter never ran'


def test_tdc_dlr_window_opt_out():

    """ With restrict_to_dlr_window=False the result of the tolerance search is
    returned unchanged, including a surplus pole outside the window. """

    for a, b in [(-0.5, 0.5), (-1.4, 0.3), (-0.7, 0.8), (-0.9, 0.9)]:

        G_w = _two_pole_gf(a, b)
        tdc = TriqsDLRCompression(G_w, tol=1e-9, verbose=False)
        raw = TriqsDLRCompression(G_w, tol=1e-9, restrict_to_dlr_window=False, verbose=False)

        np.testing.assert_array_equal(raw.poles, raw.sop_comp.poles)
        np.testing.assert_array_equal(raw.residues, raw.sop_comp.residues)
        assert raw.error == raw.sop_comp.error == raw.error_before_window
        assert len(raw.dropped_poles) == 0 and raw.window_residual == 0.
        assert tdc.error_before_window == raw.error

        if len(tdc.dropped_poles) == 0:
            np.testing.assert_array_equal(tdc.poles, raw.poles)
            np.testing.assert_array_equal(tdc.residues, raw.residues)


def test_tdc_dlr_window_edge_pole_survives():

    """ A physical pole sitting exactly on the window edge, omega = +-w_max, must survive.

    The fit may place it slightly outside the window, which the window slack covers. eps
    is swept to check that the slack holds for both loose and tight DLR accuracy. """

    for beta, w_max, eps in ((2.1, 1.0, 1e-12), (1.0, 1.0, 1e-6), (2.1, 0.5, 1e-6),
                             (0.3, 1.0, 1e-8), (5.0, 2.0, 1e-10)):

        G_w = _two_pole_gf(-w_max, w_max, beta=beta, w_max=w_max, eps=eps, ra=0.5, rb=0.5)
        tdc = TriqsDLRCompression(G_w, tol=1e-9, verbose=False)

        bw = beta * np.asarray(tdc.poles)
        weight = np.abs(np.asarray(tdc.residues)).reshape(len(bw), -1).max(axis=1)
        dominant = weight > 0.5 * weight.max()

        assert np.sum(dominant) >= 2, \
            f'beta={beta}, w_max={w_max}, eps={eps:.0e}: only {np.sum(dominant)} dominant pole(s) ' \
            f'survived, the +-w_max pair must be kept, beta*omega/Lambda = {bw / tdc.Lambda}'
        assert np.all(np.abs(bw) <= tdc.Lambda * (1 + tdc.window_slack))
        assert tdc.error < 1e-9


def test_tdc_dlr_window_harmless_drop_does_not_raise():

    """ Dropping a surplus pole whose refit is as close to the DLR expansion as the
    unfiltered fit must not raise. Here the pointwise error of the fit is about 20x its
    L2 error, so a guard comparing the pointwise change to the L2 error misfires. """

    G_w = _two_pole_gf(-0.4, 0.1, beta=20., w_max=2., eps=1e-10)

    for tol in [1e-9, 1e-10]:
        tdc = TriqsDLRCompression(G_w, tol=tol, verbose=False)
        assert tdc.error < tol
        assert np.all(np.abs(tdc.beta * np.asarray(tdc.poles)) <= tdc.Lambda * (1 + tdc.window_slack))


class _TightWindowCompression(TriqsDLRCompression):

    """ Shrinks the DLR window by a factor, so that physical poles fall outside it. """

    shrink = 0.5

    @property
    def window_slack(self):
        return self.shrink - 1.


def test_tdc_dlr_window_load_bearing_pole_raises():

    """ Dropping a pole that carries weight must raise rather than return a wrong fit. """

    G_w = _two_pole_gf(-0.5, 1.5)  # beta*omega = 3.15 for the pole at 1.5, 0.75 Lambda

    with pytest.raises(RuntimeError, match='carried real weight'):
        _TightWindowCompression(G_w, tol=1e-9, verbose=False)


def test_tdc_dlr_window_all_poles_outside_raises():

    """ A window that excludes every pole cannot be repaired and must raise. """

    class _NoWindowCompression(_TightWindowCompression):
        shrink = 1e-6

    with pytest.raises(RuntimeError, match='every one of the'):
        _NoWindowCompression(_two_pole_gf(-0.5, 0.5), tol=1e-9, verbose=False)


def test_tdc_requires_dlr_mesh():

    m = MeshImFreq(beta=100.0, statistic='Fermion', n_iw=1000)
    G_iw = Gf(mesh=m, target_shape=[])
    G_iw << inverse(iOmega_n - SemiCircular(1.0))

    with pytest.raises(ValueError):
        TriqsDLRCompression(G_iw, tol=1e-8, verbose=False)


if __name__ == "__main__":
    
    test_tdc_tol_sweep()
    test_tdc_dlr_window()
    test_tdc_dlr_window_opt_out()
    test_tdc_dlr_window_edge_pole_survives()
    test_tdc_dlr_window_harmless_drop_does_not_raise()
    test_tdc_dlr_window_load_bearing_pole_raises()
    test_tdc_dlr_window_all_poles_outside_raises()
    test_tdc_requires_dlr_mesh()

    test_gf_dlr_mesh_types()
    test_gf_dlr_requires_dlr_mesh()
    test_gf_missing_stopping_criterion()

    for max_n_poles in range(1, 20):
        test_gf_imfreq_n_poles(max_n_poles=max_n_poles)
        test_gf_dlr_n_poles(max_n_poles=max_n_poles, nonlinear_optimization=False)
        test_gf_dlr_n_poles(max_n_poles=max_n_poles, nonlinear_optimization=True)
        test_gf_imfreq_n_poles_and_tol(max_n_poles=max_n_poles)
        test_gf_dlr_n_poles_and_tol(max_n_poles=max_n_poles)

    for tol in 10.**(-np.arange(2, 13)):
        test_gf_imfreq_tol(tol=tol)
        test_gf_dlr_tol(tol=tol, nonlinear_optimization=False)
        test_gf_dlr_tol(tol=tol, nonlinear_optimization=True)
        test_gf_dlr_tol_imtime(tol=tol, nonlinear_optimization=False)
        test_gf_dlr_tol_imtime(tol=tol, nonlinear_optimization=True)