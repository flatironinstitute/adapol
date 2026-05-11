
import itertools

import numpy as np

# Import barycentric interpolation and pole fitting functions
from adapol.aaa_bra import BarycentricRationalApproximation, aaa_bra
from adapol.aaa import aaa_matrix_real

from triqs.gf import Gf, MeshImFreq, MeshDLR, MeshDLRImFreq, inverse, iOmega_n, SemiCircular, make_gf_dlr, make_gf_imtime

from adapol.aaa_bra_triqs import TriqsDLRCompression


def test_aaa_bra():

    tss = [[], [1, 1], [2, 2], [3, 3]]

    for ts in tss:
        aaa_bra_runner(target_shape=ts)


def aaa_bra_runner(target_shape):

    print('-'*72)
    print(f'test_aaa_bra with target_shape = {target_shape}')
    print('-'*72)
    
    beta = 100.0
    poles = np.array([-2.0 + 0.5j, 1.0 + 1.j])
    residues = np.array([1.0, 0.5])

    m = MeshDLRImFreq(beta=beta, statistic='Fermion', eps=1e-14, w_max=10.0)
    G_w = Gf(mesh=m, target_shape=target_shape)
    G_w << sum([inverse(iOmega_n - pole) * residue for pole, residue in zip(poles, residues)])

    Z = np.array([complex(w) for w in m])
    F = G_w.data.copy()

    tol = 1e-12
    bra = aaa_bra(Z, F, tol=tol)

    F_bra = bra(Z)
    residual = np.max(np.abs(F - F_bra))
    print(f'Max residual = {residual:2.2E} with tol = {tol:2.2E}')

    assert( residual < tol )

    if target_shape != []:
        assert(len(target_shape) == 2)
        assert(target_shape[0] == target_shape[1])
        I = np.eye(target_shape[0])
        residues = residues[:, None, None] * I[None, ...]

    bra_poles, bra_residues = bra.poles_and_residues()

    np.testing.assert_array_almost_equal(bra_poles, poles)
    np.testing.assert_array_almost_equal(bra_residues, residues)


def test_aaa_bra_constrained():

    npoless = [2, 3]
    tss = [[], [1, 1], [2, 2], [3, 3]]

    for npoles, ts in itertools.product(npoless, tss):
        aaa_bra_constrained_runner(target_shape=ts, npoles=npoles)


def aaa_bra_constrained_runner(target_shape, npoles):

    print('-'*72)
    print(f'test_aaa_bra_constrained with target_shape = {target_shape}, npoles = {npoles}')
    print('-'*72)
    
    beta = 100.0

    if npoles == 3:
        # Three poles works well for the constrained AAA
        poles = np.array([-2.0, 1.0, 3.0])
        residues = np.array([0.1, 1.0, 0.5])
    elif npoles == 2:
        # Two poles works less well for the constrained AAA
        # due to the Frossart doublet removal taking pairs of poles
        poles = np.array([-2.0, 1.0])
        residues = np.array([0.1, 1.0])
    else:
        raise NotImplementedError(f"test_aaa_bra_constrained is only implemented for npoles = 2 or 3, but got npoles = {npoles}.")

    m = MeshDLRImFreq(beta=beta, statistic='Fermion', eps=1e-14, w_max=10.0)
    G_w = Gf(mesh=m, target_shape=target_shape)
    G_w << sum([inverse(iOmega_n - pole) * residue for pole, residue in zip(poles, residues)])

    Z = np.array([complex(w) for w in m])
    F = G_w.data.copy()

    tol = 1e-12
    bra = aaa_bra(Z, F, tol=tol, constrained=True)

    F_bra = bra(Z)
    residual = np.max(np.abs(F - F_bra))
    print(f'Max residual = {residual:2.2E} with tol = {tol:2.2E}')

    assert( residual < tol )

    if target_shape != []:
        assert(len(target_shape) == 2)
        assert(target_shape[0] == target_shape[1])
        I = np.eye(target_shape[0])
        residues = residues[:, None, None] * I[None, ...]

    bra_poles, bra_residues = bra.poles_and_residues()

    # Remove poles with small residues, before comparison
    tol = 1e-12
    if target_shape == []:
        ridxs = np.nonzero(np.abs(bra_residues) < tol)
    elif len(target_shape) == 2:
        ridxs = np.nonzero(np.max(np.abs(bra_residues), axis=(1, 2)) < tol)

    bra_poles = np.delete(bra_poles, ridxs, axis=0)
    bra_residues = np.delete(bra_residues, ridxs, axis=0)

    np.testing.assert_array_almost_equal(bra_poles, poles)
    np.testing.assert_array_almost_equal(bra_residues, residues)


def test_tdc_tol_sweep():

    m = MeshDLRImFreq(beta=100.0, statistic='Fermion', eps=1e-14, w_max=8.0)

    G_w = Gf(mesh=m, target_shape=[2, 2])
    G_w << inverse(iOmega_n - 0.4 - SemiCircular(1.0))

    for tol in 10.**(-np.arange(2, 12)):
        print(f"Testing TriqsDLRCompression with tol = {tol:+2.2E}")
        tdc = TriqsDLRCompression(G_w, tol=tol, nonlinear_post_optimize=False)
        tdc_pstopt = TriqsDLRCompression(G_w, tol=tol)
        tdc_nonlin = TriqsDLRCompression(G_w, tol=tol, nonlinear_optimize=True)
        print(f'tdc_lstsq  error = {tdc.error:2.2E}, aaa_steps = {tdc.aaa_steps}')
        print(f'tdc_nonlin error = {tdc_nonlin.error:2.2E}, aaa_steps = {tdc_nonlin.aaa_steps}')
        print(f'tdc_pstopt error = {tdc_pstopt.error:2.2E}, aaa_steps = {tdc_pstopt.aaa_steps}')
        assert( tdc.error < tol)
        assert( tdc_nonlin.error < tol)
        assert( tdc_pstopt.error < tol)


def deprecated():

    G_c = make_gf_dlr(G_w)
    dlr_freq = np.array([float(w) for w in G_c.mesh])
    G_dlr_coeff = G_c.data.copy()

    G_tau = make_gf_imtime(G_w, n_tau=400)

    Z = np.array([complex(w) for w in m])
    F = G_w.data.copy()

    z = 1.j * np.linspace(Z.imag.min()*1.2, Z.imag.max()*1.2, num=400)

    tol = 1e-12
    #max_steps = 4
    max_steps = len(Z) // 2 - 1

    print('-'*72)
    bra = aaa_bra(Z, F, max_steps=2*max_steps, tol=tol)
    poles, residues = bra.poles_and_residues()
    f = bra(z)

    #bra = aaa_bra(Z, F, tol=tol)
    print('-'*72)
    cbra = aaa_bra(Z, F, max_steps=max_steps, tol=tol, constrained=True)
    cpoles, cresidues = cbra.poles_and_residues()
    fc = cbra(z)
    Fc = cbra(Z)
    print('-'*72)

    from adapol.fit_utils_dlr import erroreval_dlr, kernel, get_weight_dlr, exp_quadrature

    # L2 error in imaginary time
    err_direct, grad =  erroreval_dlr(cpoles.real, dlr_freq, G_dlr_coeff.reshape(-1, 1, 1), m.beta, weights=-cresidues.reshape(-1, 1, 1))

    # L2 error in imaginary time after least squares optimization of weights
    cresidues_opt, _ = get_weight_dlr(cpoles.real, dlr_freq, G_dlr_coeff.reshape(-1, 1, 1), m.beta)
    err_lstsq, grad =  erroreval_dlr(cpoles.real, dlr_freq, G_dlr_coeff.reshape(-1, 1, 1), m.beta, weights=cresidues_opt)

    # -- Non-linear optimization of pole locations --

    tau_nodes, tau_weights = exp_quadrature(max(2 * np.max(np.abs(np.concatenate([cpoles.real * m.beta, dlr_freq]))), 1.0))

    def func(poles):
        err, jac = erroreval_dlr(poles, dlr_freq, G_dlr_coeff.reshape(-1, 1, 1), m.beta,
                                tau_nodes=tau_nodes, tau_weights=tau_weights) 
        return err, jac

    from scipy.optimize import minimize as scipy_minimize

    res = scipy_minimize(
        func, cpoles.real, 
        method='L-BFGS-B', 
        jac=True,
        tol=1e-14,
        )

    cresidues_nonlin_opt, _ = get_weight_dlr(res.x, dlr_freq, G_dlr_coeff.reshape(-1, 1, 1), m.beta)

    print(res)

    print(f'Error of constrained AAA poles with DLR coefficients = {err_direct:2.2E} in imaginary time.')
    print(f'Error of constrained AAA poles with DLR coefficients = {err_lstsq:2.2E} in imaginary time. (after least squares optimization)')
    print(f'Error of constrained AAA poles with DLR coefficients = {res.fun:2.2E} in imaginary time. (after non-linear optimization)')

    #exit()

    tau = np.array([float(t) for t in G_tau.mesh])
    G_tau_AAA = G_tau.copy()
    G_tau_AAA.data[:] = -np.sum(cresidues[None, :] * kernel(tau/m.beta, cpoles.real*m.beta), axis=1)

    G_tau_AAA_opt = G_tau.copy()
    G_tau_AAA_opt.data[:] = np.sum(cresidues_opt.flatten()[None, :] * kernel(tau/m.beta, cpoles.real*m.beta), axis=1)

    G_tau_AAA_nonlin_opt = G_tau.copy()
    G_tau_AAA_nonlin_opt.data[:] = np.sum(cresidues_nonlin_opt.flatten()[None, :] * kernel(tau/m.beta, res.x*m.beta), axis=1)

    #print(f'AAA:  poles = {poles}')
    #print(f'AAA: residues = {residues}')
    print(f'AAA: cpoles = {cpoles}')
    print(f'AAA: cresidues = {np.abs(cresidues)}')

    zh_poles, z_interp, f_interp, zh_weight = \
        aaa_matrix_real(F.reshape(-1, 1, 1), Z, mmax=max_steps*2, tol=tol)

    zhbra = BarycentricRationalApproximation(z=z_interp, f=f_interp.flatten(), w=zh_weight)
    fzh = zhbra(z)
    Fzh = zhbra(Z)

    R_c = np.max(np.abs(F - cbra(Z)))
    R_zh = np.max(np.abs(F - zhbra(Z)))

    print(f'AAA: Max residual of constrained AAA = {R_c:2.2E}')
    print(f'AAA: Max residual of ZH AAA = {R_zh:2.2E}')

    print(f'AAA: ZH poles = {zh_poles}')
    print('-'*72)

    print(f'cbra.z = {cbra.z}')
    print(f'zh z   = {z_interp[::2]}')
    #print(f'zh z   = {z_interp}')

    print('-'*72)
    print(f'cbra.f = {cbra.f}')
    print(f'zh f   = {f_interp[::2].flatten()}')
    #print(f'zh f   = {f_interp.flatten()}')

    print('-'*72)
    print(f'cbra.w = {cbra.w}')
    print(f'zh w   = {zh_weight[::2]}')
    #print(f'zh w   = {zh_weight}')

    #print(f'F(Z) = {F}')
    #print(f'Fc(Z) = {Fc}')
    #print(f'Fzh(Z) = {Fzh}')

    #import matplotlib.pyplot as plt
    from triqs.plot.mpl_interface import oplot, plt, oplotr, oploti

    subp = [5, 1, 1]

    plt.figure(figsize=(6, 12))

    plt.subplot(*subp); subp[-1] += 1
    oplot(G_tau, label='Exact')
    oplot(G_tau_AAA, label='AAA')
    oplot(G_tau_AAA_opt, label='AAA (Opt)')
    oplot(G_tau_AAA_nonlin_opt, label='AAA (Non-linear Opt)')

    plt.subplot(*subp); subp[-1] += 1

    G_tau_AAA_err = G_tau.copy()
    G_tau_AAA_err.data[:] = np.abs(G_tau.data - G_tau_AAA.data)

    G_tau_AAA_opt_err = G_tau.copy()
    G_tau_AAA_opt_err.data[:] = np.abs(G_tau.data - G_tau_AAA_opt.data)

    G_tau_AAA_nonlin_opt_err = G_tau.copy()
    G_tau_AAA_nonlin_opt_err.data[:] = np.abs(G_tau.data - G_tau_AAA_nonlin_opt.data)

    #oplot(G_tau - G_tau_AAA, label='Error')
    #oplot(G_tau - G_tau_AAA_opt, label='Error (Opt)')

    oplotr(G_tau_AAA_err, label='Error')
    oplotr(G_tau_AAA_opt_err, label='Error (Opt)')
    oplotr(G_tau_AAA_nonlin_opt_err, label='Error (Non-linear Opt)')

    plt.semilogy([], [])

    plt.subplot(*subp); subp[-1] += 1
    plt.plot(Z.imag, F.real, '+', label='Re')
    plt.plot(Z.imag, F.imag, 'x', label='Im')

    #plt.plot(z.imag, f.real, '-', label='Re (interpolated)')
    #plt.plot(z.imag, f.imag, '-', label='Im (interpolated)')

    plt.plot(z.imag, fc.real, '-.', label='Re (conj interp)')
    plt.plot(z.imag, fc.imag, '-.', label='Im (conj interp)')

    plt.plot(Z.imag, Fc.real, '<', label='Re (conj interp)')
    plt.plot(Z.imag, Fc.imag, '>', label='Im (conj interp)')

    plt.plot(z.imag, fzh.real, ':', label='Re (ZH interp)')
    plt.plot(z.imag, fzh.imag, ':', label='Im (ZH interp)')

    plt.plot(Z.imag, Fzh.real, 's', label='Re (ZH interp)')
    plt.plot(Z.imag, Fzh.imag, 'o', label='Im (ZH interp)')


    plt.xlabel(r'$\omega_n$')
    plt.ylabel(r'$G(i\omega_n)$')
    plt.legend(loc='best')

    plt.subplot(*subp); subp[-1] += 1
    plt.plot(Z.imag, np.abs(F - Fc), '+', label='Residual of constrained AAA')
    plt.plot(Z.imag, np.abs(F - Fzh), 'x', label='Residual of ZH AAA')
    plt.xlabel(r'$\omega_n$')
    plt.ylabel(r'$\Delta G(i\omega_n)$')
    plt.legend(loc='best')

    plt.subplot(*subp); subp[-1] += 1
    #plt.plot(poles.real, poles.imag, 'x', label='Poles')
    plt.plot(cpoles.real, cpoles.imag, '+', label='Constrained Poles')
    plt.plot(zh_poles.real, zh_poles.imag, 'o', alpha=0.5, label='ZH Poles')
    plt.legend(loc='best')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_aaa_bra()
    test_aaa_bra_constrained()
    test_tdc_tol_sweep()

