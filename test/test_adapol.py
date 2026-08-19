

import numpy as np

from adapol.adapol import approx_freq_aaa
from adapol.adapol import approx_sop_fast
from adapol.adapol import approx_sop_tol


def test_freq_n_poles():
    """Test ``approx_freq_aaa`` with a maximum-pole budget.

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
    poles, residues, max_abs_diff = approx_freq_aaa(F, Z, max_n_poles=max_n_poles, verbose=True)

    print(f'Max abs diff = {max_abs_diff:2.2E}')

    print(f'Poles = {poles}')
    print(f'Residues = {residues}')
    print(f'max_n_poles = {max_n_poles}, len(poles) = {len(poles)}')

    assert( max_abs_diff < 1e-12 )
    assert( len(poles) <= max_n_poles )


def test_freq_tol():
    """Test ``approx_freq_aaa`` with a fixed error tolerance.

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
    poles, residues, max_abs_diff = approx_freq_aaa(
        F, Z, aaa_tol=tol, verbose=True)
    
    print(f'Poles = {poles}')
    print(f'Residues = {residues}')
    print(f'Max abs diff = {max_abs_diff:2.2E}')

    assert( max_abs_diff < tol )


def test_sop_n_poles():
    """Test ``approx_sop_fast`` with a maximum-pole budget.

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
        approx_sop_fast(
        poles, residues, max_n_poles=4, beta=beta, verbose=True)

    print(f'L2 norm of difference in imaginary time = {diff:2.2E}')

    assert( diff < 1e-12 )


def test_sop_tol():
    """Test ``approx_sop_fast`` with a fixed error tolerance.

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
        approx_sop_fast(
        poles, residues, aaa_tol=tol, beta=beta, verbose=True)

    print(f'L2 norm of difference in imaginary time = {diff:2.2E}')

    assert( diff < tol )


def test_sop_tol_imtime():
    """Test ``approx_sop_tol``.

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
        approx_sop_tol(
        poles, residues, tol=tol, beta=beta, verbose=True)

    print(f'L2 norm of difference in imaginary time = {diff:2.2E}')

    assert( diff < tol )


def test_freq_max_n_poles_and_tol():
    """Test ``approx_freq_aaa`` with both ``max_n_poles`` and ``tol`` set.

    AAA should stop as soon as either the tolerance is reached or the maximum
    pole budget is exhausted, whichever happens first.
    """

    print()
    print('=' * 72)
    print('test_freq_max_n_poles_and_tol')
    print('-' * 72)
    print('Fitting Matsubara samples with both a maximum-pole budget and a fixed')
    print('error tolerance, checking AAA stops at whichever limit is hit first.')
    print('=' * 72)
    print()

    poles = np.array([0.5, -1.2, -0.3])
    residues = np.array([1., 2., 3.])

    Z = 1.j * np.arange(-4.0, 4.0, 0.01)

    C_zp = 1. / (Z[:, None] - poles[None, :])
    F = np.sum(C_zp * residues[None, :], axis=1)

    # tol is loose, so the max_n_poles budget should bind first.
    max_n_poles = 2
    tol = 1e-2
    poles_fit, residues_fit, max_abs_diff = approx_freq_aaa(
        F, Z, max_n_poles=max_n_poles, aaa_tol=tol, verbose=True)

    print(f'Poles = {poles_fit}')
    print(f'Max abs diff = {max_abs_diff:2.2E}')

    assert( len(poles_fit) <= max_n_poles )


def test_sop_max_n_poles_and_tol():
    """Test ``approx_sop_fast`` with both ``max_n_poles`` and ``tol`` set.

    AAA should stop as soon as either the tolerance is reached or the maximum
    pole budget is exhausted, whichever happens first.
    """

    print()
    print('=' * 72)
    print('test_sop_max_n_poles_and_tol')
    print('-' * 72)
    print('Compressing a SumOfSimplePoles with both a maximum-pole budget and a')
    print('fixed error tolerance, checking AAA stops at whichever limit is hit first.')
    print('=' * 72)
    print()

    tol = 1e-12
    beta = 2.3
    poles = np.array([0.5, -1.2])
    residues = np.array([1., 2.])

    # tol is tight enough to be reached within the generous pole budget.
    poles_fit, residues_fit, diff = approx_sop_fast(
        poles, residues, max_n_poles=8, aaa_tol=tol, beta=beta, verbose=True)

    print(f'L2 norm of difference in imaginary time = {diff:2.2E}')

    assert( diff < tol )


def test_sop_no_criterion_raises():
    """``approx_sop_fast`` requires at least one of ``max_n_poles``/``tol``."""

    import pytest
    poles = np.array([0.5, -1.2])
    residues = np.array([1., 2.])
    with pytest.raises(ValueError):
        approx_sop_fast(poles, residues, beta=2.3)


def test_freq_no_criterion_raises():
    """``approx_freq_aaa`` requires at least one of ``max_n_poles``/``tol``."""

    import pytest
    Z = 1.j * np.array([0.1, 0.2, 0.3, 0.4])
    F = 1. / (Z - 0.5)
    with pytest.raises(ValueError):
        approx_freq_aaa(F, Z)


if __name__ == '__main__':

    test_freq_n_poles()
    test_freq_tol()
    test_sop_n_poles()
    test_sop_tol()
    test_sop_tol_imtime()
    test_freq_max_n_poles_and_tol()
    test_sop_max_n_poles_and_tol()
    test_sop_no_criterion_raises()
    test_freq_no_criterion_raises()