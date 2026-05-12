""" Test the Barycentric Rational Approximation (BRA) and the AAA algorithm. 

Author: Hugo U. R. Strand (2026)"""


import itertools
import numpy as np


from adapol.aaa import aaa
from adapol.bra import BarycentricRationalApproximation


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

    N = 1000
    Z = 1.j * np.pi / beta * (2 * np.arange(-N, N) + 1)
    C_Zp = 1./(Z[:, None] - poles[None, :])

    F = np.einsum('Zp,p->Z', C_Zp, residues)

    if len(target_shape) == 2:
        F = np.einsum('Z,...->Z...', F, np.eye(target_shape[0]))
    else:
        assert(target_shape == [])

    tol = 1e-12
    bra = aaa(Z, F, tol=tol)

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

    N = 1000
    Z = 1.j * np.pi / beta * (2 * np.arange(-N, N) + 1)
    C_Zp = 1./(Z[:, None] - poles[None, :])

    F = np.einsum('Zp,p->Z', C_Zp, residues)

    if len(target_shape) == 2:
        F = np.einsum('Z,...->Z...', F, np.eye(target_shape[0]))
    else:
        assert(target_shape == [])

    tol = 1e-12
    bra = aaa(Z, F, tol=tol, constrained=True)

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


if __name__ == "__main__":
    test_aaa_bra()
    test_aaa_bra_constrained()

