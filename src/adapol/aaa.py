"""
This code implements a specific variant of the AAA algorithm.

The major modifications compared to the original AAA algorithm are:
1. The input functions are matrix-valued; 
2. The interpolation points are on the imaginary axis, i.e., Z = i * R_+;
3. An symmetry is imposed on the interpolation points, whichh is used to obtain real-valued poles.

The implementation follows the convention in the original AAA paper [1], as well as its matlab [2] and python [3] implementations.

References:

    1. The AAA Algorithm for Rational Approximation, Yuji Nakatsukasa, Olivier Sete, and Lloyd N. Trefethen, SIAM Journal on Scientific Computing 2018 40:3, A1494-A1522https://doi.org/10.1137/16M1106122
    2. http://www.chebfun.org
    3. https://github.com/c-f-h/baryrat/blob/master/baryrat.py
"""

import numpy as np
import scipy.linalg


def aaa_matrix_real(F, Z, tol=1e-13, mmax=100):
    # only use input z that are on iR_+. Will map them to iR_- by taking conjugate of function value.
    half_index = np.imag(Z) > 0
    Z_half = Z[half_index]
    F_half = F[half_index, :, :]
    N_half = len(Z_half)

    Z = np.append(Z_half, np.conjugate(Z_half))

    Norb = F.shape[1]
    F_other_half = np.zeros_like(F_half)
    for i in range(N_half):
        F_other_half[i, :, :] = np.conjugate(np.transpose(F_half[i, :, :]))
    F = np.concatenate((F_half, F_other_half), axis=0)

    N = N_half * 2

    F_mat = np.reshape(F, (N, Norb * Norb))

    I = [i for i in range(N)]
    z_interp = []
    f_interp = np.empty((0, Norb * Norb), dtype=F.dtype)

    F_mat_fit = np.sum(F_mat) * np.ones((N, Norb * Norb), dtype=F_mat.dtype) / (N * Norb * Norb)
    
    for n in  range(2, mmax + 1, 2):
        jj = np.argmax(np.sum(abs(F_mat - F_mat_fit) ** 2, 1))
        z_interp.append(Z[jj])
        f_interp = np.concatenate((f_interp, F_mat[jj : jj + 1, :]), axis=0)
        I.remove(jj)

        jj2 = (jj + N_half) % N
        z_interp.append(Z[jj2])
        f_interp = np.concatenate((f_interp, F_mat[jj2 : jj2 + 1, :]), axis=0)

        I.remove(jj2)

        Cauchy_mat = 1.0 / (Z[I, None] - np.array(z_interp)[None, :])

        Apart = np.zeros(((N - n) * Norb * Norb, n), dtype=F.dtype)
        for i in range(Norb * Norb):
            Fhere = F_mat[:, i]
            fhere = f_interp[:, i]

            Apart[range(0 + i, i + (N - n) * Norb * Norb, Norb * Norb), :] = (Fhere[I, None] - fhere[None, :]) * Cauchy_mat

        Apart_l = Apart[:, range(0, n, 2)]
        Apart_r = Apart[:, range(1, n, 2)]
        Anew = np.concatenate((Apart_l + Apart_r, (Apart_l - Apart_r) * 1j), axis=1)
        Anew = np.concatenate((np.real(Anew), np.imag(Anew)), axis=0)
        
        _, _, Vh = scipy.linalg.svd(Anew, full_matrices=False)

        w_r = Vh[-1, :]

        w_r = np.reshape(w_r, (2, int(n / 2)))
        w_c = np.zeros((2, int(n / 2)), dtype=np.complex128)
        w_c[0, :] = w_r[0, :] + 1j * w_r[1, :]
        w_c[1, :] = w_r[0, :] - 1j * w_r[1, :]

        weight = w_c.T.flatten()
        F_mat_fit = F_mat * 1.0

        for i in range(Norb * Norb):
            F_mat_fit[I, i] = (Cauchy_mat @ (weight * f_interp[:, i])) / (Cauchy_mat @ weight)

        if np.max(np.abs(F_mat_fit - F_mat)) <= tol:
            break

    f_interp = f_interp.reshape(n, Norb, Norb)
    z_interp = np.array(z_interp)
    pol = find_pol(z_interp, weight)
    return pol, z_interp, f_interp, weight


def find_pol(z, w):
    m = len(w)
    mat1, mat2 = np.zeros((m + 1, m + 1), dtype=np.complex128), np.zeros((m + 1, m + 1), dtype=np.complex128)
    for i in range(m):
        mat2[0, i + 1] = w[i]
        mat2[i + 1, 0] = 1.0
        mat2[i + 1, i + 1] = z[i]
        mat1[i + 1, i + 1] = 1.0
    pol = scipy.linalg.eigvals(mat2, mat1)
    return np.real_if_close(pol[np.isfinite(pol)])
