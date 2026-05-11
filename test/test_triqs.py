""" Test Adapol's TRIQS API.

Author: Hugo U. R. Strand (2026)"""

import numpy as np


from adapol.sop import SumOfSimplePoles


from adapol.triqs import approximate_gf_imfreq_with_max_n_poles
from adapol.triqs import approximate_gf_imfreq_with_fixed_error_tolerance

from adapol.triqs import approximate_gf_dlr_with_max_n_poles
from adapol.triqs import approximate_gf_dlr_with_fixed_error_tolerance


try:
    from triqs.gfs import Gf, MeshImFreq, MeshDLRImFreq, inverse, iOmega_n, SemiCircular, \
        make_gf_dlr
except ImportError:
    raise ImportError(
        "It seems like you are running tests with the triqs interface "
        "but failed to import the triqs package (https://triqs.github.io/triqs/latest/). "
        "Please ensure that it is installed to run the entire test suite."
    )


def test_gf_imfreq_n_poles(max_n_poles=5):

    m = MeshImFreq(beta=100.0, statistic='Fermion', n_iw=1000)

    G_iw = Gf(mesh=m, target_shape=[])
    G_iw << inverse(iOmega_n - SemiCircular(1.0))

    poles, residues = approximate_gf_imfreq_with_max_n_poles(
        G_iw, max_n_poles=max_n_poles, verbose=True)

    sop = SumOfSimplePoles(poles, residues)
    print(f'max_n_poles = {max_n_poles}, n_poles = {len(poles)}')
    assert( len(poles) <= max_n_poles )

    iw_n = np.array([complex(w) for w in m])
    G_iw_approx = sop(iw_n)

    diff = np.max(np.abs(G_iw.data - G_iw_approx))
    print(f'Max difference between G_iw and its approximation = {diff:2.2E}')


def test_gf_imfreq_tol(tol=1e-8):

    m = MeshImFreq(beta=100.0, statistic='Fermion', n_iw=1000)

    G_iw = Gf(mesh=m, target_shape=[])
    G_iw << inverse(iOmega_n - SemiCircular(1.0))

    poles, residues = approximate_gf_imfreq_with_fixed_error_tolerance(
        G_iw, tol=tol, verbose=True)

    sop = SumOfSimplePoles(poles, residues)

    iw_n = np.array([complex(w) for w in m])
    G_iw_approx = sop(iw_n)

    diff = np.max(np.abs(G_iw.data - G_iw_approx))
    print(f'diff = {diff:2.2E}, tol = {tol}, n_poles = {len(poles)}')

    assert( diff < tol )


def test_gf_dlr_n_poles(max_n_poles=5, nonlinear_optimization=False):

    m = MeshDLRImFreq(beta=10.0, statistic='Fermion', eps=1e-12, w_max=10.0)

    G_iw = Gf(mesh=m, target_shape=[])
    G_iw << inverse(iOmega_n - SemiCircular(1.0))

    G_dlr = make_gf_dlr(G_iw)

    poles, residues = approximate_gf_dlr_with_max_n_poles(
        G_dlr, max_n_poles=max_n_poles, verbose=True, nonlinear_optimization=nonlinear_optimization)

    sop = SumOfSimplePoles(poles, residues)
    print(f'max_n_poles = {max_n_poles}, n_poles = {len(poles)}')
    assert( len(poles) <= max_n_poles )

    iw_n = np.array([complex(w) for w in m])
    G_iw_approx = sop(iw_n)

    diff = np.max(np.abs(G_iw.data - G_iw_approx))
    print(f'diff = {diff:2.2E}, n_poles = {len(poles)}')


def test_gf_dlr_tol(tol=1e-8, nonlinear_optimization=False):

    m = MeshDLRImFreq(beta=10.0, statistic='Fermion', eps=1e-12, w_max=10.0)

    G_iw = Gf(mesh=m, target_shape=[])
    G_iw << inverse(iOmega_n - SemiCircular(1.0))

    G_dlr = make_gf_dlr(G_iw)

    poles, residues = approximate_gf_dlr_with_fixed_error_tolerance(
        G_dlr, tol=tol, verbose=True, nonlinear_optimization=nonlinear_optimization)

    sop = SumOfSimplePoles(poles, residues)

    iw_n = np.array([complex(w) for w in m])
    G_iw_approx = sop(iw_n)

    diff = np.max(np.abs(G_iw.data - G_iw_approx))
    print(f'diff = {diff:2.2E}, tol = {tol}, n_poles = {len(poles)}')

    assert( diff < tol )


if __name__ == "__main__":
    
    for max_n_poles in range(1, 20):
        test_gf_imfreq_n_poles(max_n_poles=max_n_poles)
        test_gf_dlr_n_poles(max_n_poles=max_n_poles, nonlinear_optimization=False)
        test_gf_dlr_n_poles(max_n_poles=max_n_poles, nonlinear_optimization=True)

    exit()

    for tol in 10.**(-np.arange(2, 14)):
        test_gf_imfreq_tol(tol=tol)
        test_gf_dlr_tol(tol=tol, nonlinear_optimization=False)
        test_gf_dlr_tol(tol=tol, nonlinear_optimization=True)        