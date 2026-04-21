
import itertools

from adapol.fit_utils_dlr import get_weight_dlr, merge_degenerate_poles
import numpy as np

# Import barycentric interpolation and pole fitting functions
from adapol.aaa_bra import BarycentricRationalApproximation, aaa_bra
from adapol.aaa import aaa_matrix_real

from triqs.gf import Gf, MeshImFreq, MeshDLR, MeshDLRImFreq, inverse, iOmega_n, SemiCircular, make_gf_dlr, make_gf_imtime

from adapol.aaa_bra_triqs import TriqsDLRCompression


class SumOfSimplePoles:

    """ Rational function represented as a sum of simple poles, 

    .. math::
        s(z) = \\sum_k R_k / (z - p_k)
         
    where :math:`p_k` are the poles and :math:`R_k` are the residues. """
    
    def __init__(self, poles, residues):
        self.p = poles
        self.R = residues


    def __call__(self, z):
        C = 1. / (z[:, None] - self.p[None, :])
        return np.einsum('zp,p...->z...', C, self.R)


    def fit_residues(self, Z, F):
        """ Fit the residues using least squares, 
        by solving the linear system :math:`C R = F`, 
        where :math:`C_{jk} = 1/(Z_j - p_k)`. """
        
        n = len(Z)
        C = 1. / (Z[:, None] - self.p[None, :])
        residues, _, _, _ = np.linalg.lstsq(C, F.reshape(n, -1), rcond=None)
        self.R = residues.reshape([len(self.p)] + list(F.shape[1:]))


def plot_pole_evolution(max_steps=20, aaa_tol=1e-19):
    
    beta = 128.0
    m_dlr = MeshDLRImFreq(beta=beta, statistic='Fermion', eps=1e-14, w_max=8.0)
    m = MeshImFreq(beta=beta, statistic='Fermion', n_iw=1000)

    G_w = Gf(mesh=m, target_shape=[1, 1])
    G_w << inverse(iOmega_n - 0.4 - SemiCircular(1.0))

    G_w_dlr = Gf(mesh=m_dlr, target_shape=[1, 1])
    G_w_dlr << inverse(iOmega_n - 0.4 - SemiCircular(1.0))

    G_c = make_gf_dlr(G_w_dlr)
    dlr_freq = np.array([float(w) for w in G_c.mesh])
    G_dlr_coeff = G_c.data.copy()

    Z = np.array([complex(w) for w in m])
    F = G_w.data.copy()

    z = 1.j * np.linspace(Z.imag.min()*1.2, Z.imag.max()*1.2, num=400)

    import matplotlib.pyplot as plt

    subp_org = [2, 1, 1]
    plt.figure(figsize=(6, 8))

    residue_dz = 1e-5

    for max_steps in np.arange(2, max_steps + 1):

        cbra = aaa_bra(Z, F, max_steps=max_steps, tol=aaa_tol, constrained=True, cleanup=False)
        cpoles, cresidues = cbra.poles_and_residues(residue_dz=residue_dz)
        cresidues_opt, _ = get_weight_dlr(cpoles.real, dlr_freq, G_dlr_coeff, m.beta)
        cresidues_opt *= -1.

        cssp = SumOfSimplePoles(cpoles, cresidues)
        #cssp = SumOfSimplePoles(cpoles.real, cresidues)
        #cssp = SumOfSimplePoles(cpoles.real, cresidues_opt)

        cssp.fit_residues(Z, F)

        R_c = np.max(np.abs(F - cbra(Z)))
        R_cssp = np.max(np.abs(F - cssp(Z)))

        cbra_clean = aaa_bra(Z, F, max_steps=max_steps, tol=aaa_tol, constrained=True, cleanup=True)
        cpoles_clean, cresidues_clean = cbra_clean.poles_and_residues(residue_dz=residue_dz)
        cresidues_clean_opt, _ = get_weight_dlr(cpoles_clean.real, dlr_freq, G_dlr_coeff, m.beta)
        cresidues_clean_opt *= -1.

        #cssp_clean = SumOfSimplePoles(cpoles_clean, cresidues_clean)
        #cssp_clean = SumOfSimplePoles(cpoles_clean.real, cresidues_clean)
        cssp_clean = SumOfSimplePoles(cpoles_clean.real, cresidues_clean_opt)

        #cssp_clean.fit_residues(Z, F)

        R_c_clean = np.max(np.abs(F - cbra_clean(Z)))
        R_cssp_clean = np.max(np.abs(F - cssp_clean(Z)))

        zh_poles, z_interp, f_interp, zh_weight = \
            aaa_matrix_real(F, Z, mmax=max_steps*2, tol=aaa_tol)
        zh_poles_merged = merge_degenerate_poles(zh_poles.real, verbose=True)
        zhbra = BarycentricRationalApproximation(z=z_interp, f=f_interp, w=zh_weight)

        #zh_residues_opt, _ = get_weight_dlr(zh_poles.real, dlr_freq, G_dlr_coeff, m.beta)
        #zh_residues_opt *= -1.
        #zhssp = SumOfSimplePoles(zh_poles, zh_residues_opt)

        zh_residues_opt, _ = get_weight_dlr(zh_poles_merged, dlr_freq, G_dlr_coeff, m.beta)
        zh_residues_opt *= -1.
        zhssp = SumOfSimplePoles(zh_poles_merged, zh_residues_opt)

        #zhbra_poles, zhbra_residues = zhbra.poles_and_residues()
        #zhssp = SumOfSimplePoles(zhbra_poles, zhbra_residues)

        R_zh = np.max(np.abs(F - zhbra(Z)))
        R_zh_ssp = np.max(np.abs(F - zhssp(Z)))
    
        #zhbra = BarycentricRationalApproximation(z=z_interp, f=f_interp, w=zh_weight)

        print(f'max_steps = {max_steps}, poles = {cpoles}')

        subp = subp_org.copy()

        plt.subplot(*subp); subp[-1] += 1
        plt.plot(max_steps + 0*zh_poles.real, np.abs(zh_poles.imag), '+', color='r')
        plt.plot(max_steps + 0*cpoles.real, np.abs(cpoles.imag), 'x', color='b')
        plt.plot(max_steps + 0*cpoles_clean.real, np.abs(cpoles_clean.imag), 'o', color='g', alpha=0.5)

        plt.subplot(*subp); subp[-1] += 1
        plt.plot(max_steps, R_zh, '+', color='r')
        plt.plot(max_steps, R_zh_ssp, 's', color='r', alpha=0.5)

        plt.plot(max_steps, R_c, 'x', color='b')
        plt.plot(max_steps, R_cssp, '<', color='b', alpha=0.5)

        plt.plot(max_steps, R_c_clean, 'o', color='g', alpha=0.5)
        plt.plot(max_steps, R_cssp_clean, '>', color='g', alpha=0.5)


    subp = subp_org.copy()
    plt.subplot(*subp); subp[-1] += 1
    plt.plot([], [], '+', color='r', label='ZH AAA')
    plt.plot([], [], 'x', color='b', label='Constrained AAA')
    plt.plot([], [], 'o', color='g', label='Constrained AAA + cleanup')

    plt.xlabel('AAA steps')
    plt.ylabel('Imaginary part of poles')
    plt.grid(True)
    plt.yscale('log')
    plt.legend(loc='best')

    plt.subplot(*subp); subp[-1] += 1

    plt.plot([], [], '+', color='r', label='ZH Bary.Rat.')
    plt.plot([], [], 's', color='r', alpha=0.5, label='ZH Smpl.Pol. + Pol.Merg. + L2-Lstsq')

    plt.plot([], [], 'x', color='b', label='Constr.Bary.Rat')
    plt.plot([], [], '<', color='b', alpha=0.5, label='Constr.Smpl.Pol. + Z-Lstsq')

    plt.plot([], [], 'o', color='g', alpha=0.5, label='Constr.Bary.Rat. w. Cleanup')
    plt.plot([], [], '>', color='g', alpha=0.5, label='Constr.Smpl.Pol. w. Cleanup + L2-Lstsq')


    plt.xlabel('AAA steps')
    plt.ylabel('Max residual')
    plt.grid(True)
    plt.yscale('log')
    plt.legend(loc='best', fontsize=8)

    plt.tight_layout()
    plt.savefig('figure_AAA_pole_evolution.pdf')

def plot_poles_aaa(aaa_tol=1e-13, max_steps=None):

    #m = MeshDLRImFreq(beta=100.0, statistic='Fermion', eps=1e-14, w_max=20.0)
    m = MeshImFreq(beta=100.0, statistic='Fermion', n_iw=1000)

    G_w = Gf(mesh=m, target_shape=[1, 1])
    G_w << inverse(iOmega_n - 0.4 - SemiCircular(1.0))

    Z = np.array([complex(w) for w in m])
    F = G_w.data.copy()

    z = 1.j * np.linspace(Z.imag.min()*1.2, Z.imag.max()*1.2, num=400)

    if max_steps is None:
        max_steps = len(Z) // 2 - 1

    # Unconstrained AAA
    #bra = aaa_bra(Z, F, max_steps=2*max_steps, tol=tol)
    #poles, residues = bra.poles_and_residues()
    #f = bra(z)

    # Constrained AAA (running without Froissart doubles cleanup)
    cbra = aaa_bra(Z, F, max_steps=max_steps, tol=aaa_tol, constrained=True, cleanup=False)
    bra = cbra.barycentric_rational_interpolant()

    cpoles, cresidues = cbra.poles_and_residues()

    fc = cbra(z)
    Fc = cbra(Z)
    print('-'*72)

    # AAA from Zhen
    zh_poles, z_interp, f_interp, zh_weight = \
        aaa_matrix_real(F, Z, mmax=max_steps*2, tol=aaa_tol)
    
    zhbra = BarycentricRationalApproximation(z=z_interp, f=f_interp, w=zh_weight)
    fzh = zhbra(z)
    Fzh = zhbra(Z)

    R_c = np.max(np.abs(F - cbra(Z)))
    R_zh = np.max(np.abs(F - zhbra(Z)))

    # -- Compare parameters of the two rational barycentric interpolations --
    z_diff = np.max(np.abs(bra.z - zhbra.z))
    f_diff = np.max(np.abs(bra.f.flatten() - zhbra.f.flatten()))
    w_diff = np.max(np.abs(bra.w/bra.w[0] - zhbra.w/zhbra.w[0])) # NB! w has a arbitrary phase, so we normalize by the first weight to compare

    print('='*72)
    print(f'z_diff = {z_diff:2.2E}, f_diff = {f_diff:2.2E}, w_diff = {w_diff:2.2E}')
    print('='*72)

    print(f'AAA: Max residual of constrained AAA = {R_c:2.2E}')
    print(f'AAA: Max residual of ZH AAA = {R_zh:2.2E}')

    print('-'*72)
    print(f'AAA: cpoles = {cpoles}')
    print(f'AAA: ZH poles = {zh_poles}')
    print('-'*72)

    print(f'cbra.z = {cbra.z}')
    print(f'bra.z  = {bra.z}')
    print(f'zh z   = {zhbra.z}')
    #print(f'zh z   = {z_interp}')

    print('-'*72)
    print(f'cbra.f = {cbra.f.flatten()}')
    print(f'bra.f  = {bra.f.flatten()}')
    print(f'zh f   = {zhbra.f.flatten()}')
    #print(f'zh f   = {f_interp.flatten()}')

    print('-'*72)
    print(f'cbra.w = {cbra.w}')
    print(f'bra.w  = {bra.w}')
    print(f'zh w   = {zhbra.w}')
    #print(f'zh w   = {zh_weight}')

    #print(f'F(Z) = {F}')
    #print(f'Fc(Z) = {Fc}')
    #print(f'Fzh(Z) = {Fzh}')

    #import matplotlib.pyplot as plt
    from triqs.plot.mpl_interface import oplot, plt, oplotr, oploti

    subp = [3, 1, 1]

    plt.figure(figsize=(6, 12))

    F = F.flatten()
    fc = fc.flatten()
    Fc = Fc.flatten()
    fzh = fzh.flatten()
    Fzh = Fzh.flatten()

    plt.subplot(*subp); subp[-1] += 1
    plt.title(r'AAA tol = '+ f'{aaa_tol:2.2E}')

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


def plot_poles_aaa_and_optimization(aaa_tol=1e-13):

    m = MeshDLRImFreq(beta=100.0, statistic='Fermion', eps=1e-14, w_max=20.0)

    G_w = Gf(mesh=m, target_shape=[1, 1])
    G_w << inverse(iOmega_n - 0.4 - SemiCircular(1.0))

    G_c = make_gf_dlr(G_w)
    dlr_freq = np.array([float(w) for w in G_c.mesh])
    G_dlr_coeff = G_c.data.copy()

    G_tau = make_gf_imtime(G_w, n_tau=400)

    Z = np.array([complex(w) for w in m])
    F = G_w.data.copy()

    z = 1.j * np.linspace(Z.imag.min()*1.2, Z.imag.max()*1.2, num=400)

    #max_steps = 4
    max_steps = len(Z) // 2 - 1

    # Unconstrained AAA
    #bra = aaa_bra(Z, F, max_steps=2*max_steps, tol=tol)
    #poles, residues = bra.poles_and_residues()
    #f = bra(z)

    # Constrained AAA (running without Froissart doubles cleanup)
    cbra = aaa_bra(Z, F, max_steps=max_steps, tol=aaa_tol, constrained=True, cleanup=False)
    cpoles, cresidues = cbra.poles_and_residues()
    fc = cbra(z)
    Fc = cbra(Z)
    print('-'*72)

    from adapol.fit_utils_dlr import erroreval_dlr, kernel, get_weight_dlr, exp_quadrature

    # L2 error in imaginary time
    err_direct, grad =  erroreval_dlr(cpoles.real, dlr_freq, G_dlr_coeff, m.beta, weights=-cresidues.reshape(-1, 1, 1))

    # L2 error in imaginary time after least squares optimization of weights
    cresidues_opt, _ = get_weight_dlr(cpoles.real, dlr_freq, G_dlr_coeff, m.beta)
    err_lstsq, grad =  erroreval_dlr(cpoles.real, dlr_freq, G_dlr_coeff, m.beta, weights=cresidues_opt)

    # -- Non-linear optimization of pole locations --

    tau_nodes, tau_weights = exp_quadrature(max(2 * np.max(np.abs(np.concatenate([cpoles.real * m.beta, dlr_freq]))), 1.0))

    def func(poles):
        err, jac = erroreval_dlr(poles, dlr_freq, G_dlr_coeff, m.beta,
                                tau_nodes=tau_nodes, tau_weights=tau_weights) 
        return err, jac

    from scipy.optimize import minimize as scipy_minimize

    res = scipy_minimize(
        func, cpoles.real, 
        method='L-BFGS-B', 
        jac=True,
        tol=1e-14,
        )

    cresidues_nonlin_opt, _ = get_weight_dlr(res.x, dlr_freq, G_dlr_coeff, m.beta)

    print(res)

    print(f'Error of constrained AAA poles with DLR coefficients = {err_direct:2.2E} in imaginary time.')
    print(f'Error of constrained AAA poles with DLR coefficients = {err_lstsq:2.2E} in imaginary time. (after least squares optimization)')
    print(f'Error of constrained AAA poles with DLR coefficients = {res.fun:2.2E} in imaginary time. (after non-linear optimization)')


    tau = np.array([float(t) for t in G_tau.mesh])
    G_tau_AAA = G_tau.copy()
    G_tau_AAA.data[:] = -np.einsum('k...,ik->i...', cresidues, kernel(tau/m.beta, cpoles.real*m.beta))

    G_tau_AAA_opt = G_tau.copy()
    G_tau_AAA_opt.data[:] = np.einsum('k...,ik->i...', cresidues_opt, kernel(tau/m.beta, cpoles.real*m.beta))

    G_tau_AAA_nonlin_opt = G_tau.copy()
    G_tau_AAA_nonlin_opt.data[:] = np.einsum('k...,ik->i...', cresidues_nonlin_opt, kernel(tau/m.beta, cpoles.real*m.beta))

    print(f'AAA: cpoles = {cpoles}')
    print(f'AAA: cresidues = {np.abs(cresidues).flatten()}')

    # AAA from Zhen

    zh_poles, z_interp, f_interp, zh_weight = \
        aaa_matrix_real(F, Z, mmax=max_steps*2, tol=aaa_tol)
    
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

    F = F.flatten()
    fc = fc.flatten()
    Fc = Fc.flatten()
    fzh = fzh.flatten()
    Fzh = Fzh.flatten()

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

    plot_pole_evolution()
    #plot_poles_aaa(aaa_tol=1e-19, max_steps=20)

    #plot_poles_aaa(aaa_tol=1e-12)
    #plot_poles_aaa(aaa_tol=5e-13)
    #plot_poles_aaa(aaa_tol=1e-13)
    #plot_poles_aaa(aaa_tol=1e-15)
    from triqs.plot.mpl_interface import oplot, plt, oplotr, oploti
    plt.show()
