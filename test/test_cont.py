import numpy as np
import sys

sys.path.insert(0, "../")
import scipy
from matsubara import hybfit, check_weight_psd


def Kw(w, v):
    return 1 / ( v - w)


def semicircular(x):
    return 2 * np.sqrt(1 - x**2) / np.pi


def make_G_with_cont_spec(N1, Z, rho, a=-1.0, b=1.0, eps=1e-12):
    N1 = 3
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
    bathenergy, bathhyb, final_error, func, pol, weight = hybfit(
        Delta, Z, tol=tol, maxiter=500
    )

    assert final_error < tol
    assert check_weight_psd(weight)


def test_cont():
    fit_cont(5e-4)
