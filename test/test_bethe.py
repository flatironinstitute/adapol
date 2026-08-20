""" Test the adapol.approx_sop_tol function with a discretized semicircle density of states.

Author: Hugo U. R. Strand (2026)"""

import numpy as np

from adapol.adapol import approx_sop_tol


def test_real_frequency_discretized_semicircle():

    beta = 100.0
    eps = 1e-2
    tol = 1e-12

    w = np.linspace(-2+eps, 2-eps, 1000)  # real-frequency grid
    dw = w[1] - w[0]
    rho = np.sqrt(np.maximum(4 - w**2, 0.0)) / (2 * np.pi)  # semicircle density

    poles, residues, error = approx_sop_tol(w, rho * dw, tol=tol, beta=beta, verbose=True)

    print(f'poles = {poles}')
    print(f'residues = {residues}')
    print(f'error = {error}')

    assert(len(poles) < 30)      # 200 input poles compressed to a handful
    assert(float(error) < tol)  # final imaginary-time error is below the tolerance


if __name__ == '__main__':
    test_real_frequency_discretized_semicircle()