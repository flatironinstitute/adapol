"""Test that the dyadic panel Gauss-Legendre quadrature correctly integrates
sums of the kernel K(tau, omega) = exp(-tau*omega) / (1 + exp(-omega))
on [0, 1] to near machine precision."""

import numpy as np
from adapol.fit_utils_xca import exp_quadrature, kernel


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
