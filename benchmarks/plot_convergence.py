""" Test the convergence of the TriqsDLRCompression class 
as a function of the tolerance parameter.

Author: Hugo U. R. Strand (2026)"""

import time
import numpy as np

from triqs.gf import Gf, MeshDLRImFreq, SemiCircular, inverse, iOmega_n

from adapol.triqs_xca import TriqsDLRCompression


class Dummy():
    def __init__(self): pass


class ListDummy():
    def __init__(self, data): self.data = data
    def __getattr__(self, key): return np.array([ getattr(d, key) for d in self.data ])
    

def test_convergence(beta=1.0):

    tdcs = []

    tols = 10.**(-np.arange(1, 14))
    #tols = [1e-8, 1e-9]

    m = MeshDLRImFreq(beta=beta, statistic='Fermion', eps=1e-14, w_max=8.0)
    G_w = Gf(mesh=m, target_shape=[2, 2])
    G_w << inverse(iOmega_n - 0.4 - SemiCircular(1.0))

    for tol in tols:
        print(f"Testing convergence with tol = {tol:+2.2E}")

        t_tdc = time.time()
        tdc = TriqsDLRCompression(G_w, tol=tol, nonlinear_post_optimize=False)
        tdc.runtime = time.time() - t_tdc
        tdc.n_poles = len(tdc.poles)
        tdcs.append(tdc)

    tdcs = ListDummy(tdcs)

    print(f'tdcs.error = {tdcs.error}')
    print(f'tdcs.runtime = {tdcs.runtime}')

    from matplotlib import pyplot as plt

    plt.figure(figsize=(3.25, 8))
    subp = [3, 1, 1]

    ax = plt.subplot(*subp); subp[-1] += 1
    plt.title(r'$\beta = '+ f'{m.beta}$')
    plt.plot(tols, tdcs.error, 's-', label='TriqsDLRCompression')
    plt.plot(tols, tols, 'k-')
    plt.loglog()
    plt.grid(True)
    #plt.axis('square')
    plt.xlabel('Tolerance')
    plt.ylabel('L2 Error (imtime)')

    plt.subplot(*subp, sharex=ax); subp[-1] += 1
    plt.plot(tols, tdcs.n_poles, 's-', label='TriqsDLRCompression')
    plt.semilogx()
    plt.grid(True)
    plt.xlabel('Tolerance')
    plt.ylabel('Number of Poles')
    plt.legend(loc='best', fontsize=9)

    plt.subplot(*subp, sharex=ax); subp[-1] += 1
    plt.plot(tols, tdcs.runtime, 's-', label='TriqsDLRCompression')
    plt.loglog()
    plt.grid(True)
    plt.xlabel('Tolerance')
    plt.ylabel('Runtime (sec)')

    plt.tight_layout()
    plt.savefig(f'figure_convergence_beta_{m.beta}.pdf')
    #plt.show()


if __name__ == "__main__":

    betas = [1.0, 2.0, 4.0, 8.0]
    #betas = [16.0, 32.0, 64.0, 128.0]

    for beta in betas:
        test_convergence(beta=beta)