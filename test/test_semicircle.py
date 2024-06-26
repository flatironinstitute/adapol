import pytest # noqa
import numpy as np
import scipy
from aaadapol import hybfit, hybfit_triqs


def Kw(w, v):
    return 1 / ( v - w)


def semicircular(x):
    return 2 * np.sqrt(1 - x**2) / np.pi


def make_G_with_cont_spec(N1, Z, rho, a=-1.0, b=1.0, eps=1e-12):
    np.random.seed(0)
    H = np.random.rand(N1, N1) + 1j * np.random.rand(N1, N1)
    H = H + np.conj(H.T)

    G = np.zeros((Z.shape[0], H.shape[0], H.shape[1]), dtype=np.complex128)
    en, vn = np.linalg.eig(H)
    en = en / np.max(np.abs(en))  # np.random.rand(en.shape[0])
    for i in range(en.shape[0]):
        for n in range(len(Z)):

            def f(w):
                return Kw(w - en[i], Z[n]) * rho(w)

            # f = lambda w: Kw(w-en[i],Z[n])*rho(w)
            gn = scipy.integrate.quad(
                f, a, b, epsabs=eps, epsrel=eps, complex_func=True
            )[0]
            G[n, :, :] = G[n, :, :] + gn * vn[:, i] * np.conj(
                np.transpose(vn[None, :, i])
            )

    return H, G


beta = 20
N = 55
Z = 1j *(np.linspace(-N, N, N + 1)) * np.pi / beta

dim = 3
H, Delta = make_G_with_cont_spec(dim, Z, semicircular)


def fit_cont(tol):
    bathenergy, bathhyb, final_error, func = hybfit(
        Delta, Z, tol=tol, maxiter=500
    )

    assert final_error < tol


def fit_cont_triqs(tol):
    try:
        from triqs.gf import Gf, BlockGf, MeshImFreq
    except ImportError:
        raise ImportError("It seems like you are running tests with the triqs interface "
                          "but failed to import the triqs package ((https://triqs.github.io/triqs/latest/). "
                          "Please ensure that it is installed, or run \"pytest -m 'not triqs'\" to disable "
                          "the tests for triqs.")

    norb = Delta.shape[-1]
    iw_mesh = MeshImFreq(beta=beta, S='Fermion', n_iw=(N+1)//2)
    delta_iw = Gf(mesh=iw_mesh, target_shape=[norb, norb])
    delta_iw.data[:] = Delta

    # Gf interface
    V, eps, delta_fit, final_error = hybfit_triqs(delta_iw, tol=tol, maxiter=500, debug=True)
    assert final_error < tol

    # BlockGf interface
    delta_blk = BlockGf(name_list=['up', 'down'], block_list=[delta_iw, delta_iw], make_copies=True)
    V, eps, delta_fit, final_error = hybfit_triqs(delta_blk, tol=tol, maxiter=500, debug=True)
    assert final_error[0] < tol and final_error[1] < tol


def test_cont():
    fit_cont(5e-4)


@pytest.mark.triqs
def test_cont_triqs():
    fit_cont_triqs(5e-4)

