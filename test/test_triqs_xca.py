""" Test the TriqsDLRCompression function. 

Author: Hugo U. R. Strand (2026)"""


import numpy as np


from triqs.gf import Gf, MeshDLRImFreq, inverse, iOmega_n, SemiCircular


from adapol.triqs_xca import TriqsDLRCompression


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