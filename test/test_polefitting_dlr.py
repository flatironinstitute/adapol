import numpy as np
from adapol.fit_utils_dlr import polefitting_dlr


def test_polefitting_dlr():
    """Test polefitting_dlr with random DLR inputs."""
    eps = 1e-6
    N2 = 20
    Norb = 3
    w_dlr = np.random.randn(N2)
    Delta_dlr = np.random.randn(N2, Norb, Norb) + 1j * np.random.randn(N2, Norb, Norb)
    
    for i in range(N2):
        Delta_dlr[i] = Delta_dlr[i] @ Delta_dlr[i].conj().T
    
    beta = 10.0
    w_dlr = w_dlr / np.max(np.abs(w_dlr)) * 1.4 * beta
    Z = 1j * np.arange(-4001, 4002, 2) * np.pi / beta

    iw_z = 1 / (Z[:, None] - w_dlr/beta)
    Deltaiw = np.einsum('ij,jab->iab', iw_z, Delta_dlr)
    
    weight, x, error = polefitting_dlr(  Delta_dlr, w_dlr, beta, Np_max=50, eps=eps, statistics="Fermion")
    
    assert error < eps, f"Error {error} exceeds tolerance {eps}"


