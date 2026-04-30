
# Implement iterated perturbation theory (IPT) using the Triqs library, and test it on the single-band Hubbard model at half-filling.

import numpy as np

from triqs.gf import MeshReFreq
from triqs.gf import make_gf_imfreq
from triqs.gf import make_gf_dlr_imtime, make_gf_dlr, make_gf_dlr_imfreq
from triqs.gf import Gf, MeshDLRImFreq, SemiCircular, inverse, iOmega_n

def solve_ipt_and_adapol(beta=1., U=2., maxiter=40, tol=1e-12):

    m = MeshDLRImFreq(beta=beta, statistic='Fermion', eps=1e-14, w_max=np.max((2*U, 4)))

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

        #S_tau << (U**2) * G0_tau * G0_tau * G0_tau # Unstable convergence

        S_w = make_gf_dlr_imfreq(S_tau)

        G_w_new = G_w.copy()
        G_w_new << inverse(inverse(G0_w) - S_w)

        diff = np.max(np.abs(G_w_new.data - G_w.data))

        print(f"Iter {iter:3d}: max|G_w_new - G_w| = {diff:.2e}")
        if diff < tol:
            print(f"Convergence achieved after {iter} iterations.")
            break

        G_w << G_w_new

        tol_adapol = 1e-10
        if False:
            from adapol.fit_utils_dlr import polefitting_dlr_triqs
            pole_weights, poles, fit_error = polefitting_dlr_triqs(G_w, eps=tol_adapol, statistics="Fermion", verbose=True)
            pole_weights *= -1. # FIXME! Why is this necessary? Is there a sign convention issue in polefitting_dlr?
            
        from adapol.aaa_bra_triqs import TriqsDLRCompression
        tdc = TriqsDLRCompression(G_w, tol=tol_adapol, verbose=True)
        poles, pole_weights, fit_error = tdc.poles, tdc.residues, tdc.error

    return G_w


if __name__ == "__main__":
    
    G_w = solve_ipt_and_adapol(beta=40., U=5.)

    G_w_lin = make_gf_imfreq(G_w, n_iw=1000)
    # Get real axis function with Pade approximation
    G_f = Gf(mesh=MeshReFreq(window = (-5.0,5.0), n_w=1000), target_shape=[])
    G_f.set_from_pade(G_w_lin, 100, 0.01)

    from triqs.plot.mpl_interface import oplot, plt, oplotr, oploti

    oplot(-G_f.imag/np.pi)

    plt.legend(loc='best')
    plt.show()

    #solve_ipt_and_adapol()