
import itertools

from adapol.fit_utils_dlr import get_weight_dlr, merge_degenerate_poles
import numpy as np

# Import barycentric interpolation and pole fitting functions
from adapol.aaa_bra import BarycentricRationalApproximation, aaa_bra
from adapol.aaa import aaa_matrix_real

from triqs.gf import Gf, MeshImFreq, MeshDLR, MeshDLRImFreq, inverse, iOmega_n, SemiCircular, make_gf_dlr, make_gf_imtime

from adapol.aaa_bra_triqs import TriqsDLRCompression


def plot_pole_locations(max_steps=12, aaa_tol=1e-19):

    beta = 128.0

    m = MeshImFreq(beta=beta, statistic='Fermion', n_iw=1000)
    G_w = Gf(mesh=m, target_shape=[1, 1])
    G_w << inverse(iOmega_n - 0.4 - SemiCircular(1.0))
    Z = np.array([complex(w) for w in m])
    F = G_w.data.copy()

    cbra = aaa_bra(Z, F, max_steps=max_steps, tol=aaa_tol, constrained=True, cleanup=False)
    poles, residues = cbra.poles_and_residues()
    
    zh_poles, z_interp, f_interp, zh_weight = aaa_matrix_real(F, Z, mmax=max_steps*2, tol=aaa_tol)

    import matplotlib.pyplot as plt
    plt.figure()

    plt.title(f'AAA steps = {max_steps}')
    l = plt.plot(poles.real, poles.imag, 'x', label='C-BRA Poles')
    l_zh = plt.plot(zh_poles.real, zh_poles.imag, '+', label='ZH-BRA Poles')

    for pole in poles:
        plt.plot([pole.real]*2, [pole.imag, 0], '-', color=l[0].get_color())

    for pole in zh_poles:
        plt.plot([pole.real]*2, [pole.imag, 0], '-', color=l_zh[0].get_color())

    plt.legend(loc='best')
    #plt.ylim([-1e-9, +1e-9])

    plt.savefig('figure_pole_locations.pdf')
    plt.show()


if __name__ == "__main__":
    plot_pole_locations()