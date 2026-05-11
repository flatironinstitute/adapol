

import numpy as np

from adapol.sop import SumOfSimplePoles

from adapol.adapol import approximate_frequency_data_with_n_poles
from adapol.adapol import approximate_frequency_data_with_fixed_error_tolerance

from adapol.adapol import approximate_sum_of_simple_poles_with_n_poles
from adapol.adapol import approximate_sum_of_simple_poles_with_fixed_error_tolerance

def test_freq_n_poles():
    
    poles = np.array([0.5, -1.2])
    residues = np.array([1., 2.])

    Z = 1.j * np.array([0.1, 0.2, 0.3, 0.4])

    C_zp = 1. / (Z[:, None] - poles[None, :])
    F = np.sum(C_zp * residues[None, :], axis=1)

    n_poles = 2
    poles, residues = approximate_frequency_data_with_n_poles(F, Z, n_poles, verbose=True)

    sop = SumOfSimplePoles(poles=poles, residues=residues)
    F_approx = sop(Z)

    diff = np.max(np.abs(F - F_approx))
    print(f'Max abs diff = {diff:2.2E}')

    print(f'Poles = {poles}')
    print(f'Residues = {residues}')
    print(f'n_poles = {n_poles}, len(poles) = {len(poles)}')

    np.testing.assert_array_almost_equal(F_approx, F)


def test_freq_n_poles_semi_circular():
    pass # todo


def test_freq_tol():
    
    poles = np.array([0.5, -1.2, -0.3])
    residues = np.array([1., 2., 3.])

    Z = 1.j * np.arange(-4.0, 4.0, 0.01)

    C_zp = 1. / (Z[:, None] - poles[None, :])
    F = np.sum(C_zp * residues[None, :], axis=1)

    tol = 1e-12
    poles, residues = approximate_frequency_data_with_fixed_error_tolerance(
        F, Z, tol, verbose=True)

    sop = SumOfSimplePoles(poles=poles, residues=residues)
    
    print(f'Poles = {poles}')
    print(f'Residues = {residues}')

    F_approx = sop(Z)

    diff = np.max(np.abs(F - F_approx))
    print(f'Max abs diff = {diff:2.2E}')

    assert( diff < tol )


def test_sop_n_poles():

    beta = 2.3
    poles = np.array([0.5, -1.2])
    residues = np.array([1., 2.])

    poles_fit, residues_fit = approximate_sum_of_simple_poles_with_n_poles(
        poles, residues, n_poles=2, beta=beta, verbose=True)

    sop = SumOfSimplePoles(poles=poles, residues=residues)    
    sop_fit = SumOfSimplePoles(poles=poles_fit, residues=residues_fit)

    diff = (sop - sop_fit).imtime_l2_norm(beta=beta)
    print(f'L2 norm of difference in imaginary time = {diff:2.2E}')

    assert( diff < 1e-12 )


def test_sop_tol():

    tol = 1e-12
    beta = 2.3
    poles = np.array([0.5, -1.2])
    residues = np.array([1., 2.])

    poles_fit, residues_fit = approximate_sum_of_simple_poles_with_fixed_error_tolerance(
        poles, residues, tol=tol, beta=beta, verbose=True)

    sop = SumOfSimplePoles(poles=poles, residues=residues)    
    sop_fit = SumOfSimplePoles(poles=poles_fit, residues=residues_fit)

    diff = (sop - sop_fit).imtime_l2_norm(beta=beta)
    print(f'L2 norm of difference in imaginary time = {diff:2.2E}')

    assert( diff < tol )


if __name__ == '__main__':

    test_freq_n_poles()
    test_freq_tol()
    test_sop_n_poles()
    test_sop_tol()