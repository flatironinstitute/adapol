""" Plot the locations of the poles found by 
the constrained AAA algorithm for a test function.

Author: Hugo U. R. Strand (2026)"""


import numpy as np

from triqs.gfs import Gf, MeshImFreq, inverse, iOmega_n, SemiCircular

from adapol.aaa import aaa


def plot_pole_locations(max_steps=12, aaa_tol=1e-19):

    beta = 128.0

    m = MeshImFreq(beta=beta, statistic='Fermion', n_iw=1000)
    G_w = Gf(mesh=m, target_shape=[1, 1])
    G_w << inverse(iOmega_n - 0.4 - SemiCircular(1.0))
    Z = np.array([complex(w) for w in m])
    F = G_w.data.copy()

    cbra = aaa(Z, F, max_steps=max_steps, tol=aaa_tol, constrained=True, cleanup=False)
    poles, residues = cbra.poles_and_residues()
    
    import matplotlib.pyplot as plt
    plt.figure()

    plt.title(f'AAA steps = {max_steps}')
    l = plt.plot(poles.real, poles.imag, 'x', label='C-BRA Poles')

    for pole in poles:
        plt.plot([pole.real]*2, [pole.imag, 0], '-', color=l[0].get_color())

    plt.legend(loc='best')

    plt.savefig('figure_pole_locations.pdf')
    plt.show()


if __name__ == "__main__":
    plot_pole_locations()