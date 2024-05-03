import pytest
import numpy as np
import scipy
from matsubara import hybfit


def make_G_with_random_discrete_pole(Np, Z):
    np.random.seed(0)
    pol = np.random.randn(Np)
    pol = pol / np.max(np.abs(pol))
    vec = scipy.stats.ortho_group.rvs(dim=Np)
    weight = np.array(
        [vec[:, i, None] * np.transpose(np.conj(vec[:, i])) for i in range(Np)]
    )

    pol_t = np.reshape(pol, [pol.size, 1])
    M = 1 / ( Z - pol_t)
    M = M.transpose()
    if len(weight.shape) == 1:
        weight = weight / sum(weight)
        G = M @ weight
    else:
        Np = weight.shape[0]
        Norb = weight.shape[1]
        Nw = len(Z)
        G = (M @ (weight.reshape(Np, Norb * Norb))).reshape(Nw, Norb, Norb)
    return pol, vec, weight, G


def tst_discrete(Np):
    beta = 20
    N = 105
    Z = 1j *(np.linspace(-N, N, N + 1)) * np.pi / beta
    tol = 1e-6
    pol_true, vec_true, weight_true, Delta = make_G_with_random_discrete_pole(Np, Z)

    # bath_energy, bath_hyb = ImFreq_obj.bathfitting_tol(tol = tol, maxiter = 50, disp = False, cleanflag = True)
    bathenergy, bathhyb, final_error, func = hybfit(
        Delta, Z, tol=tol, maxiter=50
    )
    assert final_error < tol 
    


@pytest.mark.parametrize("Np", [2, 3, 4, 5, 6, 7])
def test_discrete(Np):
    tst_discrete(Np)


@pytest.mark.triqs
def test_discrete_triqs():
    assert 1 < 2
