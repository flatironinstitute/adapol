"""
Test for the sum of simple poles (SOP) representation of functions in complex frequency space,
and their evaluation and optimization in imaginary time.

Author: Hugo U. R. Strand (2026)
"""


import numpy as np


from adapol.sop import SumOfSimplePoles


def test_sop():

    p = np.array([-1., 0.5, 2.])
    R = np.array([1., 2., 3.])
    sop = SumOfSimplePoles(poles=p, residues=R)

    Z = np.array([0.1, 0.2, 0.3, 0.4])
    F = sop(Z)
    F_manual = np.sum(R / (Z[:, None] - p[None, :]), axis=1)
    
    np.testing.assert_array_almost_equal(F, F_manual)

    sop.fit_residues_to_freq_samples(Z, F)

    np.testing.assert_array_almost_equal(sop.R, R)


def test_sop_imtime():

    """ Test evaluation and the L2 norm in imaginary time. """

    beta = 2.3
    tau = np.linspace(0, beta, num=5)

    sop = SumOfSimplePoles(poles=np.array([0.]), residues=np.array([1.]))

    f_tau = sop.eval_imtime(tau, beta)
    print(f'f_tau = {f_tau}')
    np.testing.assert_array_almost_equal(f_tau, -0.5 * np.ones_like(tau))

    norm = sop.imtime_l2_norm(beta=beta)
    norm_ref = np.sqrt((-0.5)**2 * beta)
    print(f'norm = {norm} (ref = {norm_ref})')
    assert( np.isclose(norm, norm_ref) )

    w = 1.2
    sop = SumOfSimplePoles(poles=np.array([w]), residues=np.array([1.]))

    f_tau = sop.eval_imtime(tau, beta)
    f_tau_ref = -np.exp(-tau*w) / (1 + np.exp(-beta*w))
    print(f'f_tau = {f_tau}')
    np.testing.assert_array_almost_equal(f_tau, f_tau_ref)

    norm = sop.imtime_l2_norm(beta=beta)
    norm_ref = np.sqrt(-(np.exp(-2*beta*w) - 1) / (2*w*(1 + np.exp(-beta*w))**2))
    print(f'norm = {norm}, (ref = {norm_ref})')
    assert( np.isclose(norm, norm_ref))


def test_sop_imtime_optimization():

    beta = 2.3

    poles = np.array([1.2])
    sop = SumOfSimplePoles(poles=poles, residues=np.array([1.]))
    sop_opt = sop.best_imtime_lstsq_l2_norm_approximation_using_poles(poles=poles, beta=beta)
    np.testing.assert_array_almost_equal(sop_opt.R, sop.R)

    poles = np.array([1.2, -1., -3.])
    sop = SumOfSimplePoles(poles=poles, residues=np.array([0.5, 0.3, 0.4]))
    sop_opt = sop.best_imtime_lstsq_l2_norm_approximation_using_poles(poles=poles, beta=beta)
    np.testing.assert_array_almost_equal(sop_opt.R, sop.R)

    # -- Try to fit with fewer poles than the original function --

    eps = 1e-6
    
    poles = np.array([1. + eps, 1. - eps, 0.1])
    residues = np.array([0.5, 0.25, 1.3])

    poles_fit = np.array([1., 0.1])
    res_expected = np.array([0.75, 1.3])

    sop = SumOfSimplePoles(poles=poles, residues=residues)
    sop_opt = sop.best_imtime_lstsq_l2_norm_approximation_using_poles(poles=poles_fit, beta=beta)
    print(f'sop_opt.R = {sop_opt.R}, res_expected = {res_expected}')

    np.testing.assert_array_almost_equal(sop_opt.R, res_expected)

    # -- Try non-linear optimization of the poles --

    poles = np.array([-1., 1., 0.1])
    residues = np.array([[0.5], [0.25], [1.3]])

    poles_guess = np.array([-0.9, 1.1, 0.2])

    sop = SumOfSimplePoles(poles=poles, residues=residues)
    sop_opt = sop.best_imtime_non_linear_lstsq_l2_norm_approximation_using_pole_guess(
        poles=poles_guess, beta=beta, verbose=True)

    print(f'sop_opt.p = {sop_opt.p}, poles_expected = {poles}')
    print(f'sop_opt.R = {sop_opt.R.flatten()}, res_expected = {residues.flatten()}')

    np.testing.assert_array_almost_equal(sop_opt.R, residues)

    # -- Try non-linear optimization of the poles with fewer poles than the original function --

    eps = 1e-6    
    poles = np.array([1. + eps, 1. - eps, 0.1])
    residues = np.array([[0.5], [0.25], [1.3]])

    #poles_guess = np.array([1.0, 0.1])
    poles_guess = np.array([1.1, 0.1])

    poles_expected = np.array([1., 0.1])
    res_expected = np.array([[0.75], [1.3]])

    sop = SumOfSimplePoles(poles=poles, residues=residues)
    sop_opt = sop.best_imtime_non_linear_lstsq_l2_norm_approximation_using_pole_guess(
        poles=poles_guess, beta=beta, verbose=True)
    
    print(f'sop_opt.p = {sop_opt.p}, poles_expected = {poles_expected}')
    print(f'sop_opt.R = {sop_opt.R.flatten()}, res_expected = {res_expected.flatten()}')

    np.testing.assert_array_almost_equal(sop_opt.R, res_expected)
    np.testing.assert_array_almost_equal(sop_opt.p, poles_expected)


def test_sop_imtime_pole_grad():

    beta = 2.3
    
    poles = np.array([1.])
    residues = np.array([[1.]])

    #poles = np.array([-1., 1., 0.1])
    #residues = np.array([[0.5], [0.25], [1.3]])

    sop = SumOfSimplePoles(poles=poles, residues=residues)

    #poles_guess = np.array([1.0, 0.1])
    poles_guess = np.array([2.0])
    sop_approx = sop.best_imtime_lstsq_l2_norm_approximation_using_poles(poles=poles_guess, beta=beta)
    print(f'sop_approx.p = {sop_approx.p}, sop_approx.R = {sop_approx.R.flatten()}')

    sop_diff = sop - sop_approx
    norm_ref = sop_diff.imtime_l2_norm(beta=beta)
    print(f'norm_ref = {norm_ref}')
    # -- Manual norm calc

    #itq = sop.get_imtime_quadrature(beta=beta)
    itq = sop_diff.get_imtime_quadrature(beta=beta)
    #print(f'itq.tau_i = {itq.tau_i}')

    f_i = sop_diff.eval_imtime(itq.tau_i, beta)
    norm_0 = np.sqrt(itq.integrate(np.abs(f_i)**2))
    print(f'norm_0 = {norm_0}')

    np.testing.assert_array_almost_equal(itq.w_i, itq.sqrt_w_i**2)

    wf_i = itq.sqrt_w_i[:, None] * f_i
    norm_1 = np.sqrt(beta * np.sum(np.abs(wf_i)**2))
    print(f'norm_1 = {norm_1}')

    M_ip = itq.sqrt_w_i[:, None] * itq.kernel_matrix(sop_diff.p)
    wf_i_manual = M_ip @ sop_diff.R
    norm_2 = np.sqrt(beta) * np.linalg.norm(wf_i_manual)

    print(f'norm_2 = {norm_2}')

    np.testing.assert_array_almost_equal(wf_i, wf_i_manual)

    # -- Norm and gradient calc

    norm, grad = itq.l2_norm_gradient_with_respect_to_poles(sop, poles_guess)

    print(f'norm = {norm}, norm_ref = {norm_ref}')
    assert( np.isclose(norm, norm_ref) )

    def grad_func(poles):
        norm, grad = itq.l2_norm_gradient_with_respect_to_poles(sop, poles_guess)
        return grad

    def func(poles):
        sop_approx = sop.best_imtime_lstsq_l2_norm_approximation_using_poles(poles=poles, beta=beta)
        sop_diff = sop - sop_approx
        norm = sop_diff.imtime_l2_norm(beta=beta)
        return norm
    
    from scipy.optimize import check_grad

    grad_err = check_grad(func, grad_func, poles_guess)

    print(f'grad_err = {grad_err}')


if __name__ == "__main__":
    #test_sop()
    #test_sop_imtime()
    #test_sop_imtime_optimization()
    test_sop_imtime_pole_grad()