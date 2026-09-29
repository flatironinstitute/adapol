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


def test_gf_dlr_tol_sweep():

    m = MeshDLRImFreq(beta=100.0, statistic='Fermion', eps=1e-14, w_max=8.0)

    G_w = Gf(mesh=m, target_shape=[2, 2])
    G_w << inverse(iOmega_n - 0.4 - SemiCircular(1.0))

    for tol in 10.**(-np.arange(2, 12)):
        print(f"Testing approx_gf_dlr_tol with tol = {tol:+2.2E}")
        _, _, err = approx_gf_dlr_tol(G_w, tol=tol)
        _, _, err_nonlin = approx_gf_dlr_tol(G_w, tol=tol, nonlinear_optimization=True)
        print(f'lstsq  error = {err:2.2E}')
        print(f'nonlin error = {err_nonlin:2.2E}')
        assert( err < tol)
        assert( err_nonlin < tol)


def _two_pole_gf(a, b, beta=2.1, w_max=2.0, eps=1e-12, ra=0.6, rb=0.4):

    """ G(z) = ra/(z - a) + rb/(z - b) on a DLR Matsubara mesh. """

    m = MeshDLRImFreq(beta=beta, statistic='Fermion', w_max=w_max, eps=eps)
    G_w = Gf(mesh=m, target_shape=[1, 1])
    Z = np.array([complex(w) for w in G_w.mesh])
    G_w.data[:, 0, 0] = ra / (Z - a) + rb / (Z - b)
    return G_w


def _window(G_w):

    """ The DLR window Lambda = beta*w_max of the mesh of `G_w` and its relative slack. """

    from adapol.triqs import _dlr_window_slack
    return G_w.mesh.beta * G_w.mesh.w_max, _dlr_window_slack(G_w.mesh.eps)


def _imtime_l2_error(G_w, poles, residues):

    """ Imaginary time L2 error of the sum of simple poles against the DLR expansion of `G_w`. """

    from adapol.sop import SumOfSimplePoles
    G_c = make_gf_dlr(G_w)
    beta = G_c.mesh.beta
    dlr = SumOfSimplePoles(poles=np.array([float(w) for w in G_c.mesh]) / beta, residues=G_c.data.copy())
    sop = SumOfSimplePoles(poles=np.asarray(poles), residues=np.asarray(residues))
    return (dlr - sop).imtime_l2_norm(beta=beta)


def test_gf_dlr_tol_dlr_window():

    """ The compressed poles stay inside the DLR window, the reported error is the
    error of the returned poles, and the tolerance is met. The surplus pole of the
    conjugate-pair constrained AAA lands at a round-off determined location, which
    for most of these inputs is far outside the window. """

    beta, w_max, tol = 2.1, 2.0, 1e-9
    n_dropped = 0

    for a in np.arange(-1.9, 2.0, 0.4):
        for b in np.arange(a + 0.2, 2.0, 0.4):

            G_w = _two_pole_gf(a, b, beta=beta, w_max=w_max)
            Lambda, slack = _window(G_w)
            poles, residues, err = approx_gf_dlr_tol(G_w, tol=tol)
            raw_poles, _, _ = approx_gf_dlr_tol(G_w, tol=tol, restrict_to_dlr_window=False)
            bw = beta * np.asarray(poles)

            assert Lambda == beta * w_max
            assert np.all(np.abs(bw) <= Lambda * (1 + slack)), \
                f'a = {a:+.1f}, b = {b:+.1f}: pole outside the DLR window, beta*omega = {bw}'
            assert len(poles) == len(residues)
            assert err < tol, f'a = {a:+.1f}, b = {b:+.1f}: error {err:2.2E} >= tol'
            np.testing.assert_allclose(err, _imtime_l2_error(G_w, poles, residues), rtol=1e-6, atol=1e-15)

            n_outside = int(np.sum(np.abs(beta * np.asarray(raw_poles)) > Lambda * (1 + slack)))
            assert len(poles) == len(raw_poles) - n_outside
            n_dropped += n_outside

    assert n_dropped > 0, 'vacuous: no pole was ever outside the window, so the filter never ran'


def test_gf_dlr_tol_dlr_window_opt_out():

    """ With restrict_to_dlr_window=False the result of the tolerance search is
    returned unchanged, and without out-of-window poles the restriction is a no-op. """

    from adapol.sop_compr import SumOfPolesCompression
    from adapol.triqs import _gf_dlr_to_data

    n_outside = 0

    for a, b in [(-0.5, 0.5), (-1.4, 0.3), (-0.7, 0.8), (-0.9, 0.9)]:

        G_w = _two_pole_gf(a, b)
        Lambda, slack = _window(G_w)
        poles, residues, err = approx_gf_dlr_tol(G_w, tol=1e-9)
        raw_poles, raw_residues, raw_err = approx_gf_dlr_tol(G_w, tol=1e-9, restrict_to_dlr_window=False)

        p, R, beta, Z = _gf_dlr_to_data(G_w)
        sc = SumOfPolesCompression(p, R, beta, Z=Z, tol=1e-9, verbose=False)
        np.testing.assert_array_equal(raw_poles, sc.poles)
        np.testing.assert_array_equal(raw_residues, sc.residues)
        assert raw_err == sc.error

        outside = np.abs(beta * raw_poles) > Lambda * (1 + slack)
        n_outside += int(np.sum(outside))

        if not np.any(outside):
            np.testing.assert_array_equal(poles, raw_poles)
            np.testing.assert_array_equal(residues, raw_residues)
            assert err == raw_err

    assert n_outside > 0, 'vacuous: no raw fit had a pole outside the window'


def test_gf_dlr_tol_dlr_window_edge_pole_survives():

    """ A physical pole sitting exactly on the window edge, omega = +-w_max, must survive.

    The fit may place it slightly outside the window, which the window slack covers. eps
    is swept to check that the slack holds for both loose and tight DLR accuracy. """

    for beta, w_max, eps in ((2.1, 1.0, 1e-12), (1.0, 1.0, 1e-6), (2.1, 0.5, 1e-6),
                             (0.3, 1.0, 1e-8), (5.0, 2.0, 1e-10)):

        G_w = _two_pole_gf(-w_max, w_max, beta=beta, w_max=w_max, eps=eps, ra=0.5, rb=0.5)
        Lambda, slack = _window(G_w)
        poles, residues, err = approx_gf_dlr_tol(G_w, tol=1e-9)

        bw = beta * np.asarray(poles)
        weight = np.abs(np.asarray(residues)).reshape(len(bw), -1).max(axis=1)
        dominant = weight > 0.5 * weight.max()

        assert np.sum(dominant) >= 2, \
            f'beta={beta}, w_max={w_max}, eps={eps:.0e}: only {np.sum(dominant)} dominant pole(s) ' \
            f'survived, the +-w_max pair must be kept, beta*omega/Lambda = {bw / Lambda}'
        assert np.all(np.abs(bw) <= Lambda * (1 + slack))
        assert err < 1e-9


def test_gf_dlr_tol_dlr_window_harmless_drop_does_not_raise():

    """ Dropping a surplus pole whose refit is as close to the DLR expansion as the
    unfiltered fit must not raise. Here the pointwise error of the fit is about 20x its
    L2 error, so a guard comparing the pointwise change to the L2 error misfires. """

    G_w = _two_pole_gf(-0.4, 0.1, beta=20., w_max=2., eps=1e-10)
    Lambda, slack = _window(G_w)

    for tol in [1e-9, 1e-10]:
        poles, residues, err = approx_gf_dlr_tol(G_w, tol=tol)
        assert err < tol
        assert np.all(np.abs(G_w.mesh.beta * np.asarray(poles)) <= Lambda * (1 + slack))


def test_gf_dlr_tol_dlr_window_load_bearing_pole_raises():

    """ Dropping a pole that carries weight must raise rather than return a wrong fit.
    The window is shrunk to half its size, so that a physical pole falls outside it. """

    import adapol.triqs

    G_w = _two_pole_gf(-0.5, 1.5)  # beta*omega = 3.15 for the pole at 1.5, 0.75 Lambda

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(adapol.triqs, '_dlr_window_slack', lambda eps: -0.5)
        with pytest.raises(RuntimeError, match='carried real weight'):
            approx_gf_dlr_tol(G_w, tol=1e-9)


def test_gf_dlr_tol_dlr_window_all_poles_outside_raises():

    """ A window that excludes every pole cannot be repaired and must raise. """

    import adapol.triqs

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(adapol.triqs, '_dlr_window_slack', lambda eps: 1e-6 - 1.)
        with pytest.raises(RuntimeError, match='every one of the'):
            approx_gf_dlr_tol(_two_pole_gf(-0.5, 0.5), tol=1e-9)


def test_gf_dlr_tol_requires_dlr_mesh():

    m = MeshImFreq(beta=100.0, statistic='Fermion', n_iw=1000)
    G_iw = Gf(mesh=m, target_shape=[])
    G_iw << inverse(iOmega_n - SemiCircular(1.0))

    with pytest.raises(ValueError):
        approx_gf_dlr_tol(G_iw, tol=1e-8)


if __name__ == "__main__":
    
    test_gf_dlr_tol_sweep()
    test_gf_dlr_tol_dlr_window()
    test_gf_dlr_tol_dlr_window_opt_out()
    test_gf_dlr_tol_dlr_window_edge_pole_survives()
    test_gf_dlr_tol_dlr_window_harmless_drop_does_not_raise()
    test_gf_dlr_tol_dlr_window_load_bearing_pole_raises()
    test_gf_dlr_tol_dlr_window_all_poles_outside_raises()
    test_gf_dlr_tol_requires_dlr_mesh()

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