import pytest # noqa
import numpy as np
import scipy
from adapol.hybfit import hybfit, hybfit_triqs



def make_Delta_with_random_discrete_pole(Np, Z, statistics="Fermion"):
    np.random.seed(0)
    pol = np.random.randn(Np)
    pol = pol / np.max(np.abs(pol))
    vec = scipy.stats.ortho_group.rvs(dim=Np)
    weight = np.array(
        [vec[:, i, None] * np.transpose(np.conj(vec[:, i])) for i in range(Np)]
    )

    pol_t = np.reshape(pol, [pol.size, 1])
    M = 1 / ( Z - pol_t)
    if statistics == "Boson":
        M = pol_t / ( Z - pol_t)
        M[:, Z==0] = -1.0
    M = M.transpose()
    if len(weight.shape) == 1:
        weight = weight / sum(weight)
        Delta = M @ weight
    else:
        Np = weight.shape[0]
        Norb = weight.shape[1]
        Nw = len(Z)
        Delta = (M @ (weight.reshape(Np, Norb * Norb))).reshape(Nw, Norb, Norb)
    return pol, vec, weight, Delta


def tst_discrete(Np):
    beta = 20
    N = 105
    Z = 1j *(np.linspace(-N, N, N + 1)) * np.pi / beta
    tol = 1e-6
    pol_true, vec_true, weight_true, Delta = make_Delta_with_random_discrete_pole(Np, Z)

    bathenergy, bathhyb, final_error, func = hybfit(
        Delta, Z, tol=tol, maxiter=50
    )
    assert final_error < tol 

def tst_discrete_boson(Np):
    beta = 20
    N = 105
    Z = 1j * np.arange(-N, N+1) * np.pi / beta
    tol = 1e-6
    pol_true, vec_true, weight_true, Delta = make_Delta_with_random_discrete_pole(Np, Z, statistics="Boson")

    bathenergy, bathhyb, final_error, func = hybfit(
        Delta, Z, tol=tol, maxiter=50, statistics="Boson"
    )
    f_reconstruct =  func(Z)  
    assert final_error < tol  
    print(np.max(np.abs(Delta - f_reconstruct)))
    assert np.max(np.abs(Delta - f_reconstruct)) < tol*100

def tst_discrete_triqs(Np):
    try:
        from triqs.gf import Gf, BlockGf, MeshImFreq
    except ImportError:
        raise ImportError("It seems like you are running tests with the triqs interface "
                          "but failed to import the triqs package ((https://triqs.github.io/triqs/latest/). "
                          "Please ensure that it is installed, or run \"pytest -m 'not triqs'\" to disable "
                          "the tests for triqs.")

    beta = 20
    N = 105
    Z = 1j * (np.linspace(-N, N, N + 1)) * np.pi / beta
    tol = 1e-6
    pol_true, vec_true, weight_true, Delta = make_Delta_with_random_discrete_pole(Np, Z)

    norb = Delta.shape[-1]
    iw_mesh = MeshImFreq(beta=beta, S='Fermion', n_iw=(N+1)//2)
    delta_iw = Gf(mesh=iw_mesh, target_shape=[norb, norb])
    delta_iw.data[:] = Delta

    # Gf interface
    bathhyb, bathenergy, delta_fit, final_error = hybfit_triqs(delta_iw, tol=tol, maxiter=50, debug=True)
    assert final_error < tol

    # BlockGf interface
    delta_blk = BlockGf(name_list=['up', 'down'], block_list=[delta_iw, delta_iw], make_copies=True)
    bathhyb, bathenergy, delta_fit, final_error = hybfit_triqs(delta_blk, tol=tol, maxiter=50, debug=True)
    assert final_error[0] < tol and final_error[1] < tol


@pytest.mark.parametrize("Np", [2, 3, 4, 5, 6, 7])
def test_discrete(Np):
    tst_discrete(Np)

@pytest.mark.parametrize("Np", [2, 3, 4, 5, 6, 7])
def test_discrete_boson(Np):
    tst_discrete_boson(Np)

@pytest.mark.triqs
@pytest.mark.parametrize("Np", [2, 3, 4, 5, 6, 7])
def test_discrete_triqs(Np):
    tst_discrete_triqs(Np)
