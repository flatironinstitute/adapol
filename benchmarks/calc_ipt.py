
"""
Purpose
-------

Implement iterated perturbation theory (IPT) using the Triqs library, 
and test it on the single-band Hubbard model at half-filling.

The generated Green's functions are used as non-trivial test cases for the 
AAA-BRA pole compression in the imaginary time domain.

Finding
-------

In general we find that when the TriqsDLRCompression fails it is related 
to convergence issues in the IPT self-consistency not in the AAA-BRA compression itself.

Once the IPT self-consistency is stable the TriqsDLRCompression successfully 
compresses the Green's functions generated in each self-consistent step.

Side note
---------

However, the convergence of the Triqs/cppdlr based IPT solver itself is somewhat unstable, 
and we need to enforce hermiticity and particle-hole symmetry to achieve convergence.

A reference implementation using the PyDLR library is more stable and does not require these constraints to converge.

Author: Hugo U. R. Strand, 2026
"""

import numpy as np

import pydlr
from triqs.gfs import MeshReFreq
from triqs.gfs import make_gf_imfreq, make_gf_imtime
from triqs.gfs import make_gf_dlr_imtime, make_gf_dlr, make_gf_dlr_imfreq
from triqs.gfs import Gf, MeshDLRImFreq, SemiCircular, inverse, iOmega_n

from adapol.triqs import approx_gf_dlr_tol


def solve_ipt_and_adapol(
        beta=1., U=2., maxiter=100, tol=1e-12, 
        mix=0.75, run_adapol=True, tol_adapol=1e-9):

    """ Solve the single-band Hubbard model at half-filling using iterated perturbation theory (IPT)
    
    in order to generate non-tivial Green's functions for testing the AAA-BRA pole compression.
    
    The compression is performed at each iteration of the IPT self-consistency loop
    but the result is not used in the self consistent DMFT-IPT iteration.
     
    When the compression fails (to reach the requested accuracy) we stop and return the corresponding Green's function."""

    m = MeshDLRImFreq(beta=beta, statistic='Fermion', eps=1e-12, w_max=np.max((4*U, 4)))
    print(m)

    G_w = Gf(mesh=m, target_shape=[])
    S_w = G_w.copy()
    G0_w = G_w.copy()

    S_tau = make_gf_dlr_imtime(S_w)
    m_tau = S_tau.mesh

    G_w << SemiCircular(2.0)

    for iter in range(1, maxiter+1):
        
        G0_w << inverse(iOmega_n - G_w)

        G0_tau = make_gf_dlr_imtime(G0_w)
        G0_dlr = make_gf_dlr(G0_tau)
        for t in m_tau:
            S_tau[t] = (U**2) * G0_tau[t] * G0_tau[t] * G0_dlr(-t)
        
        S_tau.data[:].imag = 0. # Enforce hermicity to stabilize convergence ??

        S_w = make_gf_dlr_imfreq(S_tau)

        G_w_new = G_w.copy()
        G_w_new << inverse(inverse(G0_w) - S_w)

        diff = np.max(np.abs(G_w_new.data - G_w.data))

        # Perform pole compression using TriqsDLRCompression

        if run_adapol:
            try:
                poles, pole_weights, fit_error = \
                    approx_gf_dlr_tol(
                    G_w, tol=tol_adapol, verbose=False)

            except ValueError as e:
                print(f"Warning: AAA-BRA compression failed with error: {e}")
                break
            print(f"Iter {iter:3d}: max|G_w_new - G_w| = {diff:.2e}, U = {U}, beta = {beta}, n_poles = {len(poles)}, fit_error = {fit_error:.2e}")
        else:
            print(f"Iter {iter:3d}: max|G_w_new - G_w| = {diff:.2e}, U = {U}, beta = {beta}")

        if diff < tol:
            print(f"Convergence achieved after {iter} iterations.")
            break

        G_w << mix * G_w_new + (1 - mix) * G_w

        G_w.data[:].real = 0 # Enforce particle-hole symmetry to stabilize convergence ??

        if iter == maxiter:
            print(f"Warning: Maximum number of iterations ({maxiter}) reached without convergence.")
            break

    return G_w, diff, iter


def solve_ipt_pydlr(
        beta=1., U=2., maxiter=100, tol=1e-12, 
        mix=0.75, matsubara_dyson=False):
    
    """ Solve the single-band Hubbard model at half-filling using iterated perturbation theory (IPT) and the PyDLR library. 
    
    This produces a reference solution to the Triqs based solver above. 
    
    Unfortunately we find that the pydlr based solver is stable without imposing the hermiticity and particle-hole symmetry 
    constraints that we need to impose in the Triqs based solver to achieve convergence. This suggests that the pydlr based 
    solver is more stable than the Triqs based one. We should investigate this further. 
    
    One of the differences is the solving of the Dyson equation in DLR coefficient space. (Instead of in Matsubara frequency)
    The solution in Matsubara frequency is also different in the selection of the imaginary frequency grid, since
    pydlr uses a non-uniform initial grid for selecting the DLR Matsubara nodes.
    """
    
    from pydlr.pydlr import dlr
    d = dlr(lamb=beta*np.max([4*U, 4]), eps=1e-12)
    print(f'd.rank = {d.rank}, d.eps = {d.eps}, d.lamb = {d.lamb}, w_max = {d.lamb/beta}')
    
    from pydlr import kernel
    from scipy.integrate import quad

    def eval_semi_circ_tau(tau, beta, h, t):
        I = lambda x : -2 / np.pi / t**2 * kernel(np.array([tau])/beta, beta*np.array([x]))[0,0]
        g, res = quad(I, -t+h, t+h, weight='alg', wvar=(0.5, 0.5))
        return g

    eval_semi_circ_tau = np.vectorize(eval_semi_circ_tau)
    tau_l = d.get_tau(beta)
    h=0.0
    t=1.0
    G_l = np.array(eval_semi_circ_tau(tau_l, beta, h, t).reshape(-1, 1, 1), dtype=complex)
    G_x = d.dlr_from_tau(G_l)

    Sigma_l = np.zeros_like(G_l)
    H = np.array([[0.0]], dtype=complex)

    def sigma_x_ipt(g_x, J, d, beta):

        tau_l = d.get_tau(beta)
        tau_l_rev = beta - tau_l

        g_l = d.tau_from_dlr(g_x)
        g_l_rev = d.eval_dlr_tau(g_x, tau_l_rev, beta)

        sigma_l = J**2 * g_l**2 * g_l_rev

        return sigma_l

    for iter in range(1, maxiter+1):

        if matsubara_dyson:
            G_q = d.matsubara_from_dlr(G_x, beta)
            G0_q = d.dyson_matsubara(H, G_q, beta)
            G0_x = d.dlr_from_matsubara(G0_q, beta)
            pass
        else:
            G0_x = d.dyson_dlr(H, G_x, beta)
 
        Sigma_l = sigma_x_ipt(G0_x, U, d, beta)

        if matsubara_dyson:
            Sigma_l.imag[:] = 0.            
            Sigma_x = d.dlr_from_tau(Sigma_l)
            Sigma_q = d.matsubara_from_dlr(Sigma_x, beta)
            G_q_new = d.dyson_matsubara(H, Sigma_q + G_q, beta)
            
            G_q_new.real[:] = 0. # Enforce particle-hole symmetry to stabilize convergence ??

            G_x_new = d.dlr_from_matsubara(G_q_new, beta)
            G_l_new = d.tau_from_dlr(G_x_new)
        else:
            Sigma_x = d.dlr_from_tau(Sigma_l)
            G_x_new = d.dyson_dlr(H, Sigma_x + G_x, beta)
            G_l_new = d.tau_from_dlr(G_x_new)

        diff = np.max(np.abs(G_l_new - G_l))

        print(f"Iter {iter:3d}: max|G_l_new - G_l| = {diff:.2e}, U = {U}, beta = {beta}")

        if diff < tol:
            print(f"Convergence achieved after {iter} iterations.")
            break

        G_l = mix * G_l_new + (1 - mix) * G_l
        G_x = d.dlr_from_tau(G_l)

        if iter == maxiter:
            print(f"Warning: Maximum number of iterations ({maxiter}) reached without convergence.")
            break

    return G_l, diff, iter, d


if __name__ == "__main__":
    
    betas = [
        40., 80., 160., 320., 
        #640., 1280., 2560., 5120.
        ]

    G_ws = []
    diffs = []
    iters = []
    
    G_w_pys = []
    diffs_py = []
    iters_py = []
    
    diffs_py_vs_triqs = []

    for beta in betas:

        G_w, diff, iter = solve_ipt_and_adapol(
            beta=beta, U=5., tol=1e-8)
        
        G_ws.append(G_w)
        diffs.append(diff)
        iters.append(iter)

        G_l, diff_py, iter_py, d = solve_ipt_pydlr(
            beta=beta, U=5., tol=1e-8, matsubara_dyson=False)
        
        diffs_py.append(diff_py)
        iters_py.append(iter_py)

        # Eval pydlr result on Matsubara grid of Triqs and c.f.
        G_x = d.dlr_from_tau(G_l)
        w = np.array([complex(w) for w in G_w.mesh])
        G_w_py = d.eval_dlr_freq(G_x, w, beta).flatten()
        diff_py_vs_triqs = np.max(np.abs(G_w_py - G_w.data))
        print(f'diff_py_vs_triqs = {diff_py_vs_triqs:.2e}')

        diffs_py_vs_triqs.append(diff_py_vs_triqs)
        G_w_pys.append((w, G_w_py))


    from triqs.plot.mpl_interface import oplot, plt, oplotr, oploti
    plt.figure(figsize=(6, 8))

    for G_w, (w_py, G_w_py), beta in zip(G_ws, G_w_pys, betas):
        subp = [4, 1, 1]

        plt.subplot(*subp); subp[-1] += 1
        oplot(G_w.real, label=rf'$\beta$={beta}')
        plt.plot(w_py.imag, G_w_py.real, '+-')
        plt.xlabel(r'$\omega_n$')
        plt.ylabel(r'$\mathrm{Re} G(i\omega_n)$')
        plt.legend(fontsize=7)
        
        plt.subplot(*subp); subp[-1] += 1
        oplot(G_w.imag, label=rf'$\beta$={beta}')
        plt.plot(w_py.imag, G_w_py.imag, '+-')
        plt.xlabel(r'$\omega_n$')
        plt.ylabel(r'$\mathrm{Im} G(i\omega_n)$')
        plt.legend(fontsize=7)

    plt.subplot(*subp); subp[-1] += 1
    plt.plot(betas, diffs, 'o-', label='Triqs')
    plt.plot(betas, diffs_py, 's-', label='PyDLR')
    plt.plot(betas, diffs_py_vs_triqs, '^-', label='PyDLR vs Triqs')
    plt.xscale('log')
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$\max|G_w^{(n+1)} - G_w^{(n)}|$')
    plt.legend()

    plt.subplot(*subp); subp[-1] += 1
    plt.plot(betas, iters, 'o-', label='Triqs')
    plt.plot(betas, iters_py, 's-', label='PyDLR')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\beta$')
    plt.ylabel('DMFT iters')
    plt.legend()

    plt.tight_layout()
    plt.show()
    