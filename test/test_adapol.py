

import numpy as np


from adapol.sop import SumOfSimplePoles

from adapol.adapol import approximate_frequency_data_with_max_n_poles
from adapol.adapol import approximate_frequency_data_with_fixed_error_tolerance

from adapol.adapol import approximate_sum_of_simple_poles_with_max_n_poles
from adapol.adapol import approximate_sum_of_simple_poles_with_fixed_error_tolerance
from adapol.adapol import approximate_sum_of_simple_poles_with_fixed_error_tolerance_in_imaginary_time


def test_freq_n_poles():
    """Test ``approximate_frequency_data_with_max_n_poles``.

    Fits Matsubara samples generated from a known sum-of-simple-poles under
    a maximum-pole budget and checks the fit reproduces the samples.
    """

    print()
    print('=' * 72)
    print('test_freq_n_poles')
    print('-' * 72)
    print('Fitting Matsubara samples from a known sum-of-simple-poles using a')
    print('maximum-pole budget, and checking the fit reproduces the samples.')
    print('=' * 72)
    print()

    poles = np.array([0.5, -1.2])
    residues = np.array([1., 2.])

    Z = 1.j * np.array([0.1, 0.2, 0.3, 0.4])

    C_zp = 1. / (Z[:, None] - poles[None, :])
    F = np.sum(C_zp * residues[None, :], axis=1)

    max_n_poles = 4
    poles, residues, max_abs_diff = approximate_frequency_data_with_max_n_poles(F, Z, max_n_poles, verbose=True)

    print(f'Max abs diff = {max_abs_diff:2.2E}')

    print(f'Poles = {poles}')
    print(f'Residues = {residues}')
    print(f'max_n_poles = {max_n_poles}, len(poles) = {len(poles)}')

    assert( max_abs_diff < 1e-12 )
    assert( len(poles) <= max_n_poles )


def test_freq_tol():
    """Test ``approximate_frequency_data_with_fixed_error_tolerance``.

    Fits Matsubara samples generated from a known sum-of-simple-poles under
    a fixed error tolerance and checks the max sample-wise error meets it.
    """

    print()
    print('=' * 72)
    print('test_freq_tol')
    print('-' * 72)
    print('Fitting Matsubara samples from a known sum-of-simple-poles using a')
    print('fixed error tolerance, and checking the max sample-wise error meets it.')
    print('=' * 72)
    print()

    poles = np.array([0.5, -1.2, -0.3])
    residues = np.array([1., 2., 3.])

    Z = 1.j * np.arange(-4.0, 4.0, 0.01)

    C_zp = 1. / (Z[:, None] - poles[None, :])
    F = np.sum(C_zp * residues[None, :], axis=1)

    tol = 1e-12
    poles, residues, max_abs_diff = approximate_frequency_data_with_fixed_error_tolerance(
        F, Z, tol, verbose=True)
    
    print(f'Poles = {poles}')
    print(f'Residues = {residues}')
    print(f'Max abs diff = {max_abs_diff:2.2E}')

    assert( max_abs_diff < tol )


def test_sop_n_poles():
    """Test ``approximate_sum_of_simple_poles_with_max_n_poles``.

    Compresses an existing ``SumOfSimplePoles`` under a maximum-pole budget
    and checks the imaginary-time L2 norm of the difference is small.
    """

    print()
    print('=' * 72)
    print('test_sop_n_poles')
    print('-' * 72)
    print('Compressing an existing SumOfSimplePoles using a maximum-pole budget,')
    print('and checking the imaginary-time L2 norm of the difference is small.')
    print('=' * 72)
    print()

    beta = 2.3
    poles = np.array([0.5, -1.2])
    residues = np.array([1., 2.])

    poles_fit, residues_fit, diff = \
        approximate_sum_of_simple_poles_with_max_n_poles(
        poles, residues, max_n_poles=4, beta=beta, verbose=True)

    print(f'L2 norm of difference in imaginary time = {diff:2.2E}')

    assert( diff < 1e-12 )


def test_sop_tol():
    """Test ``approximate_sum_of_simple_poles_with_fixed_error_tolerance``.

    Compresses an existing ``SumOfSimplePoles`` under a fixed error tolerance
    and checks the imaginary-time L2 norm of the difference meets it.
    """

    print()
    print('=' * 72)
    print('test_sop_tol')
    print('-' * 72)
    print('Compressing an existing SumOfSimplePoles using a fixed error tolerance,')
    print('and checking the imaginary-time L2 norm of the difference meets it.')
    print('=' * 72)
    print()

    tol = 1e-12
    beta = 2.3
    poles = np.array([0.5, -1.2])
    residues = np.array([1., 2.])

    poles_fit, residues_fit, diff = \
        approximate_sum_of_simple_poles_with_fixed_error_tolerance(
        poles, residues, tol=tol, beta=beta, verbose=True)

    print(f'L2 norm of difference in imaginary time = {diff:2.2E}')

    assert( diff < tol )


def test_sop_tol_imtime():
    """Test ``approximate_sum_of_simple_poles_with_fixed_error_tolerance``.

    Compresses an existing ``SumOfSimplePoles`` under a fixed (imtime) error tolerance
    and checks the imaginary-time L2 norm of the difference meets it.
    """

    print()
    print('=' * 72)
    print('test_sop_tol_imtime')
    print('-' * 72)
    print('Compressing an existing SumOfSimplePoles using a fixed (imtime) error tolerance,')
    print('and checking the imaginary-time L2 norm of the difference meets it.')
    print('=' * 72)
    print()

    tol = 1e-12
    beta = 2.3
    poles = np.array([0.5, -1.2])
    residues = np.array([1., 2.])

    poles_fit, residues_fit, diff = \
        approximate_sum_of_simple_poles_with_fixed_error_tolerance_in_imaginary_time(
        poles, residues, tol=tol, beta=beta, verbose=True)

    print(f'L2 norm of difference in imaginary time = {diff:2.2E}')

    assert( diff < tol )


if __name__ == '__main__':

    test_freq_n_poles()
    test_freq_tol()
    test_sop_n_poles()
    test_sop_tol()
    test_sop_tol_imtime()