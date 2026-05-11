""" Plot the evolution of the poles found by the constrained AAA algorithm 
variants as a function of the number of AAA steps.

Author: Hugo U. R. Strand (2026)"""


import numpy as np


from triqs.gf import Gf, MeshImFreq, MeshDLRImFreq, inverse, iOmega_n, SemiCircular, make_gf_dlr


from adapol.aaa import aaa
from adapol.sop import SumOfSimplePoles


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

    sop = SumOfSimplePoles(poles=dlr_freq/beta, residues=G_dlr_coeff)

    Z = np.array([complex(w) for w in m])
    F = G_w.data.copy()

    z = 1.j * np.linspace(Z.imag.min()*1.2, Z.imag.max()*1.2, num=400)

    import matplotlib.pyplot as plt

    subp_org = [2, 1, 1]
    plt.figure(figsize=(6, 8))

    residue_dz = 1e-5

    for max_steps in np.arange(2, max_steps + 1):

        cbra = aaa(Z, F, max_steps=max_steps, tol=aaa_tol, constrained=True, cleanup=False)
        cpoles, cresidues = cbra.poles_and_residues(residue_dz=residue_dz)

        cssp = cbra.get_sop(real_poles=False)
        cssp.fit_residues_to_freq_samples(Z, F)

        R_c = np.max(np.abs(F - cbra(Z)))
        R_cssp = np.max(np.abs(F - cssp(Z)))

        cbra_clean = aaa(Z, F, max_steps=max_steps, tol=aaa_tol, constrained=True, cleanup=True)
        cpoles_clean, cresidues_clean = cbra_clean.poles_and_residues(residue_dz=residue_dz)

        cssp_clean = sop.best_imtime_lstsq_l2_norm_approximation_using_poles(cpoles_clean.real, beta)

        R_c_clean = np.max(np.abs(F - cbra_clean(Z)))
        R_cssp_clean = np.max(np.abs(F - cssp_clean(Z)))

        print(f'max_steps = {max_steps}, poles = {cpoles}')

        subp = subp_org.copy()

        plt.subplot(*subp); subp[-1] += 1
        plt.plot(max_steps + 0*cpoles.real, np.abs(cpoles.imag), 'x', color='b')
        plt.plot(max_steps + 0*cpoles_clean.real, np.abs(cpoles_clean.imag), 'o', color='g', alpha=0.5)

        plt.subplot(*subp); subp[-1] += 1

        plt.plot(max_steps, R_c, 'x', color='b')
        plt.plot(max_steps, R_cssp, '<', color='b', alpha=0.5)

        plt.plot(max_steps, R_c_clean, 'o', color='g', alpha=0.5)
        plt.plot(max_steps, R_cssp_clean, '>', color='g', alpha=0.5)


    subp = subp_org.copy()
    plt.subplot(*subp); subp[-1] += 1
    plt.plot([], [], 'x', color='b', label='Constrained AAA')
    plt.plot([], [], 'o', color='g', label='Constrained AAA + cleanup')

    plt.xlabel('AAA steps')
    plt.ylabel('Imaginary part of poles')
    plt.grid(True)
    plt.yscale('log')
    plt.legend(loc='best')

    plt.subplot(*subp); subp[-1] += 1

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


if __name__ == "__main__":

    plot_pole_evolution()

    from matplotlib import pyplot as plt
    plt.show()
