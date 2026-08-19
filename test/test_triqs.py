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


if __name__ == "__main__":
    
    test_tdc_tol_sweep()

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