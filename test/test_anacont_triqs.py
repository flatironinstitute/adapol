
import pytest
import numpy as np

from adapol.anacont import anacont_triqs

@pytest.mark.triqs
def test_anacont_triqs_semi_circular_sweep_accuracy():
    """Test polefitting_dlr_triqs with a Gf on MeshDLR."""
    try:
        from triqs.gf import Gf, MeshDLRImFreq, SemiCircular, inverse, iOmega_n
    except ImportError:
        raise ImportError(
            "It seems like you are running tests with the triqs interface "
            "but failed to import the triqs package (https://triqs.github.io/triqs/latest/). "
            "Please ensure that it is installed, or run \"pytest -m 'not triqs'\" to disable "
            "the tests for triqs."
        )

    m = MeshDLRImFreq(beta=100.0, statistic='Fermion', eps=1e-12, w_max=10.0)
    Delta_iw = Gf(mesh=m, target_shape=[])

    Delta_iw << inverse(iOmega_n - SemiCircular(1.0))

    for tol in 10.**(-np.arange(1, 14)):

        print(f"Testing anacont_triqs with tol = {tol:+2.2E}")
        interp, fit_error, poles, pole_weights = anacont_triqs(Delta_iw, tol=tol, debug=True, verbose=True)
        assert( tol > fit_error )
        print(f'n_poles = {len(poles)}')


@pytest.mark.triqs
def test_anacont_triqs_semi_circular_sweep_prefactor():
    """Test polefitting_dlr_triqs with a Gf on MeshDLR."""
    try:
        from triqs.gf import Gf, MeshDLRImFreq, SemiCircular, inverse, iOmega_n
    except ImportError:
        raise ImportError(
            "It seems like you are running tests with the triqs interface "
            "but failed to import the triqs package (https://triqs.github.io/triqs/latest/). "
            "Please ensure that it is installed, or run \"pytest -m 'not triqs'\" to disable "
            "the tests for triqs."
        )

    m = MeshDLRImFreq(beta=100.0, statistic='Fermion', eps=1e-12, w_max=10.0)
    Delta_iw = Gf(mesh=m, target_shape=[])

    Delta_iw << inverse(iOmega_n - SemiCircular(1.0))

    tol = 1e-10
    for prefactor in 10.**np.arange(3, -4, -1):
        print(f"Testing anacont_triqs with prefactor = {prefactor:+2.2E}")
        interp, fit_error, poles, pole_weights = anacont_triqs(prefactor * Delta_iw, tol=tol, debug=True, verbose=True)
        assert( tol > fit_error )
        print(f'n_poles = {len(poles)}')


@pytest.mark.triqs
def test_anacont_triqs_break_aaa_matrix_real():
    """Test polefitting_dlr_triqs with a Gf on MeshDLR."""
    try:
        from triqs.gf import Gf, MeshDLRImFreq, SemiCircular, inverse, iOmega_n
    except ImportError:
        raise ImportError(
            "It seems like you are running tests with the triqs interface "
            "but failed to import the triqs package (https://triqs.github.io/triqs/latest/). "
            "Please ensure that it is installed, or run \"pytest -m 'not triqs'\" to disable "
            "the tests for triqs."
        )

    m = MeshDLRImFreq(beta=100.0, statistic='Fermion', eps=1e-9, w_max=1.0)
    Delta_iw = Gf(mesh=m, target_shape=[])

    Delta_iw << inverse(iOmega_n - SemiCircular(1.0))

    tol = 1e-10
    interp, fit_error, poles, pole_weights = anacont_triqs(Delta_iw, tol=tol, debug=True, verbose=True)
    assert( tol > fit_error )
    print(f'n_poles = {len(poles)}')


if __name__ == "__main__":
    #test_anacont_triqs_semi_circular_sweep_accuracy()
    #test_anacont_triqs_semi_circular_sweep_prefactor()
    test_anacont_triqs_break_aaa_matrix_real()