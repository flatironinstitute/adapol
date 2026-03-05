"""Test that the dyadic panel Gauss-Legendre quadrature correctly integrates
sums of the kernel K(tau, omega) = exp(-tau*omega) / (1 + exp(-omega))
on [0, 1] to near machine precision."""

import numpy as np
from adapol.fit_utils_dlr import exp_quadrature, kernel, erroreval_dlr, get_weight_dlr


def analytic_integral(omega):
    """Exact integral of K(tau, omega) over tau in [0, 1].

    int_0^1 exp(-tau*omega)/(1+exp(-omega)) dtau
      = (1 - exp(-omega)) / (omega * (1 + exp(-omega)))
      = tanh(omega/2) / omega        for omega != 0
      = 1/2                           for omega == 0
    """
    omega = np.asarray(omega, dtype=float)
    result = np.where(
        np.abs(omega) < 1e-12,
        0.5,
        np.tanh(omega / 2.0) / omega,
    )
    return result


def test_panel_quadrature_sum_of_exponentials():
    """Integrate a weighted sum of four kernels K(tau, omega_k) with large
    frequencies, where dyadic refinement is essential."""

    omegas = np.array([-2.0, 47.0, -200.0, 500.0])
    coeffs = np.array([0.7, -1.0, -0.5, 0.25])

    omega_max = np.max(np.abs(omegas))
    nodes, weights = exp_quadrature(omega_max)

    K_vals = kernel(nodes, omegas)

    numerical = 0.0
    for k in range(len(omegas)):
        numerical += coeffs[k] * np.dot(weights, K_vals[:, k])

    exact = np.dot(coeffs, analytic_integral(omegas))

    rel_error = np.abs(numerical - exact) / np.abs(exact)
    print(f"\nSum of 4 kernels, omegas = {omegas.tolist()}, coeffs = {coeffs.tolist()}")
    print(f"  omega_max = {omega_max}, quadrature nodes = {len(nodes)}")
    print(f"  exact = {exact:.16e}, numerical = {numerical:.16e}, rel error = {rel_error:.2e}")

    assert rel_error < 1e-14, f"Relative error {rel_error:.4e} exceeds tolerance"


def test_panel_quadrature_individual_kernels():
    """Test each kernel individually across a range of omega values."""

    omega_values = [0.1, 1.0, 10.0, 50.0, 100.0, -0.1, -1.0, -10.0, -50.0, -100.0]

    print(f"\nIndividual kernels K(tau, omega) integrated over [0, 1]:")
    for omega in omega_values:
        omega_arr = np.array([omega])
        omega_max = np.abs(omega)
        nodes, weights = exp_quadrature(max(omega_max, 1.0))
        K_vals = kernel(nodes, omega_arr).flatten()

        numerical = np.dot(weights, K_vals)
        exact = float(analytic_integral(omega))

        rel_error = np.abs(numerical - exact) / max(np.abs(exact), 1e-300)
        print(f"  omega = {omega:8.1f}: {len(nodes):4d} nodes, rel error = {rel_error:.2e}")

        assert rel_error < 1e-14, f"omega={omega}: relative error {rel_error:.4e} exceeds tolerance"

def test_erroreval_gradient():
    """Test gradient computation in fit_utils_dlr.erroreval using finite difference validation."""
    
    N1 = 20
    N2 = 10
    pol = np.random.randn(N1) 

    Norb = 3
    weights = np.random.randn(N1, Norb, Norb) + 1j * np.random.randn(N1, Norb, Norb)
    w_dlr = np.random.randn(N2)
    Delta_dlr = np.random.randn(N2, Norb, Norb) + 1j * np.random.randn(N2, Norb, Norb)
    for i in range(N2):
        Delta_dlr[i] = Delta_dlr[i] @ Delta_dlr[i].conj().T
    # Delta_dlr = Delta_dlr * 0.0
    beta = 100.0
    w_dlr = w_dlr / np.max(np.abs(w_dlr)) * 1.4 * beta

    gradient1 = erroreval_dlr(pol,  w_dlr,Delta_dlr,   beta, weights=weights )[1]

    # Finite difference validation of gradient
    eps = 1e-6
    grad_fd = np.zeros_like(pol)
    for i in range(pol.size):
        pol_p = pol.copy()
        pol_m = pol.copy()
        pol_p[i] += eps
        pol_m[i] -= eps
        err_p = erroreval_dlr(pol_p,  w_dlr,Delta_dlr,   beta, weights=weights)[0]
        err_m = erroreval_dlr(pol_m,  w_dlr,Delta_dlr,   beta, weights=weights)[0]
        grad_fd[i] = (err_p - err_m) / (2 * eps)

    assert np.allclose(gradient1, grad_fd), f"Gradients do not match: {np.linalg.norm(gradient1 - grad_fd)}"

def test_get_weight_dlr():
    """Test that get_weight_dlr returns expected weights for a simple case."""
    N1 = 20
    N2 = 10
    pol = np.random.randn(N1) 

    Norb = 3
    weights = np.random.randn(N1, Norb, Norb) + 1j * np.random.randn(N1, Norb, Norb)
    w_dlr = np.random.randn(N2)
    Delta_dlr = np.random.randn(N2, Norb, Norb) + 1j * np.random.randn(N2, Norb, Norb)
    for i in range(N2):
        Delta_dlr[i] = Delta_dlr[i] @ Delta_dlr[i].conj().T
    # Delta_dlr = Delta_dlr * 0.0
    beta = 100.0
    w_dlr = w_dlr / np.max(np.abs(w_dlr)) * 1.4 * beta
    # Check that weights are positive and of reasonable magnitude
    weights_dlr = get_weight_dlr(w_dlr/beta, w_dlr, Delta_dlr, beta)[0]
    assert np.allclose(erroreval_dlr(w_dlr/beta,  w_dlr,Delta_dlr,   beta, weights=weights_dlr)[0], 0), "Error should be close to zero"