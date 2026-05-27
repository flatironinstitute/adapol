""" Test Adapol's TRIQS API.

Author: Hugo U. R. Strand (2026)"""

import pytest

pytest.importorskip(
    "triqs",
    reason="Triqs is not installed. Skipping test_triqs_xca. "
           "Please ensure that it is installed to run the entire test suite."
)


import numpy as np


from triqs.gfs import Gf, MeshImFreq, MeshDLRImFreq, inverse, \
    iOmega_n, SemiCircular, make_gf_dlr


from adapol.sop import SumOfSimplePoles

from adapol.triqs import TriqsDLRCompression

from adapol.triqs import approximate_gf_imfreq_with_max_n_poles
from adapol.triqs import approximate_gf_imfreq_with_fixed_error_tolerance

from adapol.triqs import approximate_gf_dlr_with_max_n_poles
from adapol.triqs import approximate_gf_dlr_with_fixed_error_tolerance
from adapol.triqs import approximate_gf_dlr_with_fixed_error_tolerance_in_imaginary_time


def test_gf_imfreq_n_poles(max_n_poles=5):

    m = MeshImFreq(beta=100.0, statistic='Fermion', n_iw=1000)

    G_iw = Gf(mesh=m, target_shape=[])
    G_iw << inverse(iOmega_n - SemiCircular(1.0))

    poles, residues, diff = approximate_gf_imfreq_with_max_n_poles(
        G_iw, max_n_poles=max_n_poles, verbose=True)

    print(f'max_n_poles = {max_n_poles}, n_poles = {len(poles)}')
    assert( len(poles) <= max_n_poles )
    print(f'Max difference between G_iw and its approximation = {diff:2.2E}')


def test_gf_imfreq_tol(tol=1e-8):

    m = MeshImFreq(beta=100.0, statistic='Fermion', n_iw=1000)

    G_iw = Gf(mesh=m, target_shape=[])
    G_iw << inverse(iOmega_n - SemiCircular(1.0))

    poles, residues, diff = approximate_gf_imfreq_with_fixed_error_tolerance(
        G_iw, tol=tol, verbose=True)

    print(f'diff = {diff:2.2E}, tol = {tol}, n_poles = {len(poles)}')

    assert( diff < tol )


def test_gf_dlr_n_poles(max_n_poles=5, nonlinear_optimization=False):

    m = MeshDLRImFreq(beta=10.0, statistic='Fermion', eps=1e-12, w_max=10.0)

    G_iw = Gf(mesh=m, target_shape=[])
    G_iw << inverse(iOmega_n - SemiCircular(1.0))

    G_dlr = make_gf_dlr(G_iw)

    poles, residues, diff = approximate_gf_dlr_with_max_n_poles(
        G_dlr, max_n_poles=max_n_poles, verbose=True, nonlinear_optimization=nonlinear_optimization)

    print(f'max_n_poles = {max_n_poles}, n_poles = {len(poles)}')
    assert( len(poles) <= max_n_poles )
    print(f'diff = {diff:2.2E}, n_poles = {len(poles)}')


def test_gf_dlr_tol(tol=1e-8, nonlinear_optimization=False):

    m = MeshDLRImFreq(beta=10.0, statistic='Fermion', eps=1e-12, w_max=10.0)

    G_iw = Gf(mesh=m, target_shape=[])
    G_iw << inverse(iOmega_n - SemiCircular(1.0))

    G_dlr = make_gf_dlr(G_iw)

    poles, residues, diff = approximate_gf_dlr_with_fixed_error_tolerance(
        G_dlr, tol=tol, verbose=True, nonlinear_optimization=nonlinear_optimization)

    print(f'diff = {diff:2.2E}, tol = {tol}, n_poles = {len(poles)}')

    assert( diff < tol )


def test_gf_dlr_tol_imtime(tol=1e-8, nonlinear_optimization=False):

    m = MeshDLRImFreq(beta=10.0, statistic='Fermion', eps=1e-12, w_max=10.0)

    G_iw = Gf(mesh=m, target_shape=[])
    G_iw << inverse(iOmega_n - SemiCircular(1.0))

    G_dlr = make_gf_dlr(G_iw)

    poles, residues, diff = approximate_gf_dlr_with_fixed_error_tolerance_in_imaginary_time(
        G_dlr, tol=tol, verbose=True, nonlinear_optimization=nonlinear_optimization)

    print(f'diff = {diff:2.2E}, tol = {tol}, n_poles = {len(poles)}')

    assert( diff < tol )


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

    for max_n_poles in range(1, 20):
        test_gf_imfreq_n_poles(max_n_poles=max_n_poles)
        test_gf_dlr_n_poles(max_n_poles=max_n_poles, nonlinear_optimization=False)
        test_gf_dlr_n_poles(max_n_poles=max_n_poles, nonlinear_optimization=True)

    for tol in 10.**(-np.arange(2, 13)):
        test_gf_imfreq_tol(tol=tol)
        test_gf_dlr_tol(tol=tol, nonlinear_optimization=False)
        test_gf_dlr_tol(tol=tol, nonlinear_optimization=True)
        test_gf_dlr_tol_imtime(tol=tol, nonlinear_optimization=False)
        test_gf_dlr_tol_imtime(tol=tol, nonlinear_optimization=True)