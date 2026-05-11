
import itertools

import numpy as np

# Import barycentric interpolation and pole fitting functions
from adapol.aaa_bra import BarycentricRationalApproximation, aaa_bra
from adapol.aaa import aaa_matrix_real

from triqs.gf import Gf, MeshImFreq, MeshDLR, MeshDLRImFreq, inverse, iOmega_n, SemiCircular, make_gf_dlr, make_gf_imtime

from adapol.aaa_bra_triqs import TriqsDLRCompression


def test_aaa_bra():

    tss = [[], [1, 1], [2, 2], [3, 3]]

    for ts in tss:
        aaa_bra_runner(target_shape=ts)


def aaa_bra_runner(target_shape):

    print('-'*72)
    print(f'test_aaa_bra with target_shape = {target_shape}')
    print('-'*72)
    
    beta = 100.0
    poles = np.array([-2.0 + 0.5j, 1.0 + 1.j])
    residues = np.array([1.0, 0.5])

    m = MeshDLRImFreq(beta=beta, statistic='Fermion', eps=1e-14, w_max=10.0)
    G_w = Gf(mesh=m, target_shape=target_shape)
    G_w << sum([inverse(iOmega_n - pole) * residue for pole, residue in zip(poles, residues)])

    Z = np.array([complex(w) for w in m])
    F = G_w.data.copy()

    tol = 1e-12
    bra = aaa_bra(Z, F, tol=tol)

    F_bra = bra(Z)
    residual = np.max(np.abs(F - F_bra))
    print(f'Max residual = {residual:2.2E} with tol = {tol:2.2E}')

    assert( residual < tol )

    if target_shape != []:
        assert(len(target_shape) == 2)
        assert(target_shape[0] == target_shape[1])
        I = np.eye(target_shape[0])
        residues = residues[:, None, None] * I[None, ...]

    bra_poles, bra_residues = bra.poles_and_residues()

    np.testing.assert_array_almost_equal(bra_poles, poles)
    np.testing.assert_array_almost_equal(bra_residues, residues)


def test_aaa_bra_constrained():

    npoless = [2, 3]
    tss = [[], [1, 1], [2, 2], [3, 3]]

    for npoles, ts in itertools.product(npoless, tss):
        aaa_bra_constrained_runner(target_shape=ts, npoles=npoles)


def aaa_bra_constrained_runner(target_shape, npoles):

    print('-'*72)
    print(f'test_aaa_bra_constrained with target_shape = {target_shape}, npoles = {npoles}')
    print('-'*72)
    
    beta = 100.0

    if npoles == 3:
        # Three poles works well for the constrained AAA
        poles = np.array([-2.0, 1.0, 3.0])
        residues = np.array([0.1, 1.0, 0.5])
    elif npoles == 2:
        # Two poles works less well for the constrained AAA
        # due to the Frossart doublet removal taking pairs of poles
        poles = np.array([-2.0, 1.0])
        residues = np.array([0.1, 1.0])
    else:
        raise NotImplementedError(f"test_aaa_bra_constrained is only implemented for npoles = 2 or 3, but got npoles = {npoles}.")

    m = MeshDLRImFreq(beta=beta, statistic='Fermion', eps=1e-14, w_max=10.0)
    G_w = Gf(mesh=m, target_shape=target_shape)
    G_w << sum([inverse(iOmega_n - pole) * residue for pole, residue in zip(poles, residues)])

    Z = np.array([complex(w) for w in m])
    F = G_w.data.copy()

    tol = 1e-12
    bra = aaa_bra(Z, F, tol=tol, constrained=True)

    F_bra = bra(Z)
    residual = np.max(np.abs(F - F_bra))
    print(f'Max residual = {residual:2.2E} with tol = {tol:2.2E}')

    assert( residual < tol )

    if target_shape != []:
        assert(len(target_shape) == 2)
        assert(target_shape[0] == target_shape[1])
        I = np.eye(target_shape[0])
        residues = residues[:, None, None] * I[None, ...]

    bra_poles, bra_residues = bra.poles_and_residues()

    # Remove poles with small residues, before comparison
    tol = 1e-12
    if target_shape == []:
        ridxs = np.nonzero(np.abs(bra_residues) < tol)
    elif len(target_shape) == 2:
        ridxs = np.nonzero(np.max(np.abs(bra_residues), axis=(1, 2)) < tol)

    bra_poles = np.delete(bra_poles, ridxs, axis=0)
    bra_residues = np.delete(bra_residues, ridxs, axis=0)

    np.testing.assert_array_almost_equal(bra_poles, poles)
    np.testing.assert_array_almost_equal(bra_residues, residues)


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
    test_aaa_bra()
    test_aaa_bra_constrained()
    test_tdc_tol_sweep()

