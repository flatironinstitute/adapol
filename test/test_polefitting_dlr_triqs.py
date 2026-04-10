import pytest
import numpy as np
from adapol.fit_utils_dlr import polefitting_dlr_triqs


def _make_delta_iw_dlr(beta, Norb, Np, w_max):
    """Build a Delta(iw) on MeshDLRImFreq from known discrete poles."""
    from triqs.gf import Gf, MeshDLR, MeshDLRImFreq

    np.random.seed(0)
    pol = np.random.randn(Np)
    pol = pol / np.max(np.abs(pol))
    weight = np.random.randn(Np, Norb, Norb) + 1j * np.random.randn(Np, Norb, Norb)
    for i in range(Np):
        weight[i] = weight[i] @ weight[i].conj().T

    dlr_mesh = MeshDLR(beta=beta, statistic='Fermion', w_max=w_max, eps=1e-14)
    dlr_iw_mesh = MeshDLRImFreq(dlr_mesh)
    delta_iw = Gf(mesh=dlr_iw_mesh, target_shape=[Norb, Norb])

    iwn_vec = np.array([iw.value for iw in dlr_iw_mesh.values()])
    for n in range(len(iwn_vec)):
        for p in range(Np):
            delta_iw.data[n] += weight[p] / (iwn_vec[n] - pol[p])

    return delta_iw, pol, weight


@pytest.mark.triqs
def test_polefitting_dlr_triqs_gf():
    """Test polefitting_dlr_triqs with a Gf on MeshDLR."""
    try:
        from triqs.gf import Gf, MeshDLR, MeshDLRImFreq, MeshImFreq, make_gf_dlr
    except ImportError:
        raise ImportError(
            "It seems like you are running tests with the triqs interface "
            "but failed to import the triqs package (https://triqs.github.io/triqs/latest/). "
            "Please ensure that it is installed, or run \"pytest -m 'not triqs'\" to disable "
            "the tests for triqs."
        )

    eps = 1e-6
    Norb = 2
    Np = 4
    beta = 20.0
    w_max = 5.0

    # Construct delta_iw on MeshDLRImFreq, then convert to MeshDLR
    delta_iw, pol_true, weight_true = _make_delta_iw_dlr(beta, Norb, Np, w_max)
    delta_dlr = make_gf_dlr(delta_iw)

    weight, pol, error = polefitting_dlr_triqs(delta_dlr, eps=eps, Np_max=50)

    # Validate by reconstructing on a dense Matsubara grid
    N = 105
    Z = 1j * np.arange(-2*N-1, 2*N+2, 2) * np.pi / beta
    Delta_exact = np.zeros((len(Z), Norb, Norb), dtype=complex)
    for n in range(len(Z)):
        for p in range(Np):
            Delta_exact[n] += weight_true[p] / (Z[n] - pol_true[p])

    Delta_recon = np.einsum('ij,jab->iab', 1 / (Z[:, None] - pol), weight)
    recon_err = np.max(np.abs(Delta_exact + Delta_recon))
    assert recon_err < eps, f"Reconstruction error {recon_err} exceeds tolerance {eps}"


@pytest.mark.triqs
def test_polefitting_dlr_triqs_imfreq():
    """Test polefitting_dlr_triqs with a Gf on MeshDLRImFreq (auto-converted to MeshDLR)."""
    try:
        from triqs.gf import Gf, MeshDLR, MeshDLRImFreq, MeshImFreq, make_gf_dlr
    except ImportError:
        raise ImportError(
            "It seems like you are running tests with the triqs interface "
            "but failed to import the triqs package (https://triqs.github.io/triqs/latest/). "
            "Please ensure that it is installed, or run \"pytest -m 'not triqs'\" to disable "
            "the tests for triqs."
        )

    eps = 1e-6
    Norb = 2
    Np = 4
    beta = 20.0
    w_max = 5.0

    # Construct delta_iw on MeshDLRImFreq and pass it directly
    delta_iw, pol_true, weight_true = _make_delta_iw_dlr(beta, Norb, Np, w_max)

    weight, pol, error = polefitting_dlr_triqs(delta_iw, eps=eps, Np_max=50)

    # Validate by reconstructing on a dense Matsubara grid
    N = 105
    Z = 1j * np.arange(-2*N-1, 2*N+2, 2) * np.pi / beta
    Delta_exact = np.zeros((len(Z), Norb, Norb), dtype=complex)
    for n in range(len(Z)):
        for p in range(Np):
            Delta_exact[n] += weight_true[p] / (Z[n] - pol_true[p])

    Delta_recon = np.einsum('ij,jab->iab', 1 / (Z[:, None] - pol), weight)
    recon_err = np.max(np.abs(Delta_exact + Delta_recon))
    assert recon_err < eps, f"Reconstruction error {recon_err} exceeds tolerance {eps}"


@pytest.mark.triqs
def test_polefitting_dlr_triqs_blockgf():
    """Test polefitting_dlr_triqs with a BlockGf."""
    try:
        from triqs.gf import Gf, BlockGf, MeshDLR, MeshDLRImFreq, make_gf_dlr
    except ImportError:
        raise ImportError(
            "It seems like you are running tests with the triqs interface "
            "but failed to import the triqs package (https://triqs.github.io/triqs/latest/). "
            "Please ensure that it is installed, or run \"pytest -m 'not triqs'\" to disable "
            "the tests for triqs."
        )

    eps = 1e-6
    Norb = 2
    Np = 4
    beta = 20.0
    w_max = 5.0

    # Construct delta_iw on MeshDLRImFreq, then convert to MeshDLR for BlockGf
    delta_iw, pol_true, weight_true = _make_delta_iw_dlr(beta, Norb, Np, w_max)
    delta_dlr = make_gf_dlr(delta_iw)

    delta_blk = BlockGf(name_list=['up', 'down'], block_list=[delta_dlr, delta_dlr], make_copies=True)

    weight_list, pol_list, error_list = polefitting_dlr_triqs(delta_blk, eps=eps, Np_max=50)

    assert len(weight_list) == 2
    assert len(pol_list) == 2
    assert len(error_list) == 2

    # Validate each block
    N = 105
    Z = 1j * np.arange(-2*N-1, 2*N+2, 2) * np.pi / beta
    Delta_exact = np.zeros((len(Z), Norb, Norb), dtype=complex)
    for n in range(len(Z)):
        for p in range(Np):
            Delta_exact[n] += weight_true[p] / (Z[n] - pol_true[p])

    for weight, pol in zip(weight_list, pol_list):
        Delta_recon = np.einsum('ij,jab->iab', 1 / (Z[:, None] - pol), weight)
        recon_err = np.max(np.abs(Delta_exact + Delta_recon))
        assert recon_err < eps, f"Reconstruction error {recon_err} exceeds tolerance {eps}"


@pytest.mark.triqs
def test_polefitting_dlr_triqs_semi_circular_sweep_accuracy():
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

    m = MeshDLRImFreq(beta=10.0, statistic='Fermion', eps=1e-12, w_max=4.0)
    Delta_iw = Gf(mesh=m, target_shape=[1, 1])

    Delta_iw << inverse(iOmega_n - SemiCircular(1.0))

    for tol in 10.**(-np.arange(1, 10)):
        print(f"Testing polefitting_dlr_triqs with tol = {tol:+2.2E}")
        weights, poles, fit_error = polefitting_dlr_triqs(Delta_iw, eps=tol, verbose=True)
        assert( tol > fit_error )
        print(f'n_poles = {len(poles)}')


@pytest.mark.triqs
def test_polefitting_dlr_triqs_semi_circular_sweep_prefactor(tol=1e-6):
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

    m = MeshDLRImFreq(beta=10.0, statistic='Fermion', eps=1e-12, w_max=4.0)
    Delta_iw = Gf(mesh=m, target_shape=[])

    Delta_iw << inverse(iOmega_n - SemiCircular(1.0))

    for prefactor in 10.**np.arange(3, -4, -1):
        print(f"Testing polefitting_dlr_triqs with prefactor = {prefactor:+2.2E}")
        weights, poles, fit_error = polefitting_dlr_triqs(prefactor * Delta_iw, eps=tol, verbose=True, Np_max=100)
        print(fit_error)
        assert( tol > fit_error )
        print(f'n_poles = {len(poles)}')


if __name__ == "__main__":
    test_polefitting_dlr_triqs_semi_circular_sweep_accuracy()
    test_polefitting_dlr_triqs_semi_circular_sweep_prefactor()