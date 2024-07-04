"""
This code implements a specific variant of the AAA algorithm.

"""

import numpy as np
import scipy.linalg


def aaa_matrix_real(F, Z, tol=1e-13, mmax=100):
    """ """
    Z = np.asanyarray(Z).ravel()

    # only use input z that are on iR_+. Will map them to iR_- by taking conjugate of function value.
    half_index = np.imag(Z) > 0
    Z_half = Z[half_index]
    F_half = F[half_index, :, :]
    M_half = len(Z_half)

    Z = np.append(Z_half, np.conjugate(Z_half))

    Norb = F.shape[1]
    F_other_half = np.zeros_like(F_half)
    for i in range(M_half):
        F_other_half[i, :, :] = np.conjugate(np.transpose(F_half[i, :, :]))
    F = np.concatenate((F_half, F_other_half), axis=0)

    M = M_half * 2

    F_mat = np.reshape(F, (M, Norb * Norb))

    J = list(range(M))
    zj = np.empty(0, dtype=Z.dtype)
    fj = np.empty((0, Norb * Norb), dtype=F.dtype)
    C = np.empty([M, 0], dtype=F.dtype)
    errors = []

    reltol = tol * np.linalg.norm(F_mat, np.inf)

    R = np.mean(F_mat) * np.ones_like(F_mat)

    mlist = range(2, mmax + 1, 2)

    for m in mlist:
        # find largest residual
        jj = np.argmax(np.sum(abs(F_mat - R) ** 2, 1))
        zj = np.append(zj, (Z[jj],))
        fj = np.concatenate((fj, F_mat[jj : jj + 1, :]), axis=0)
        J.remove(jj)

        # Cauchy matrix containing the basis functions as columns

        jj2 = (jj + M_half) % M
        zj = np.append(zj, (Z[jj2],))
        fj = np.concatenate((fj, F_mat[jj2 : jj2 + 1, :]), axis=0)

        J.remove(jj2)

        C = 1.0 / (Z[J, None] - zj[None, :])

        # Loewner matrix
        Apart = np.zeros(((M - m) * Norb * Norb, m), dtype=F.dtype)
        for i in range(Norb * Norb):
            Fhere = F_mat[:, i]
            fjhere = fj[:, i]

            Apart[range(0 + i, i + (M - m) * Norb * Norb, Norb * Norb), :] = (
                Fhere[J, None] - fjhere[None, :]
            ) * C

        Awidth = np.size(Apart, 1)
        Apart_l = Apart[:, range(0, Awidth, 2)]
        Apart_r = Apart[:, range(1, Awidth, 2)]
        Anew = np.concatenate((Apart_l + Apart_r, (Apart_l - Apart_r) * 1j), axis=1)
        Anew = np.concatenate((np.real(Anew), np.imag(Anew)), axis=0)

        # compute weights as right singular vector for smallest singular value
        _, _, Vh = np.linalg.svd(Anew, full_matrices=False)

        wj_r = Vh[-1, :]

        wj_r = np.reshape(wj_r, (2, int(m / 2)))
        wj_c = np.zeros((2, int(m / 2)), dtype=np.complex128)
        wj_c[0, :] = wj_r[0, :] + 1j * wj_r[1, :]
        wj_c[1, :] = wj_r[0, :] - 1j * wj_r[1, :]

        wj = np.asanyarray(wj_c.T).ravel()

        # approximation: numerator / denominator

        D = C.dot(wj)

        # update residual
        R = F_mat.copy()

        for i in range(Norb * Norb):
            fjhere = fj[:, i]
            N = C.dot(wj * fjhere)  # needs to change N and R
            R[J, i] = N / D

        # check for convergence
        errors.append(np.linalg.norm(F_mat - R, np.inf))
        if errors[-1] <= reltol:
            break

    fj = fj.reshape(m, Norb, Norb)
    pol = aaa_pol(zj, wj)
    return pol, zj, fj, wj


def aaa_pol(zj, wj):
    """Return the poles and residues of the rational function."""

    m = len(wj)

    # compute poles
    B = np.eye(m + 1)
    B[0, 0] = 0
    E = np.block([[0, wj], [np.ones((m, 1)), np.diag(zj)]])
    evals = scipy.linalg.eigvals(E, B)
    pol = np.real_if_close(evals[np.isfinite(evals)])

    return pol
