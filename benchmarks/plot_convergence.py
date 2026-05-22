""" Test the convergence of the TriqsDLRCompression class 
as a function of the tolerance parameter.

Author: Hugo U. R. Strand (2026)"""

import time
import numpy as np

from triqs.gfs import Gf, MeshDLRImFreq, SemiCircular, inverse, iOmega_n, make_gf_dlr

from adapol.triqs import approximate_gf_dlr_with_fixed_error_tolerance
from adapol.triqs import approximate_gf_imfreq_with_fixed_error_tolerance

from adapol.triqs_xca import TriqsDLRCompression

from adapol.triqs import _gf_dlr_to_data
from adapol.sop import SumOfSimplePoles

class Dummy():
    def __init__(self): pass


class ListDummy():
    def __init__(self, data): self.data = data
    def __getattr__(self, key): return np.array([ getattr(d, key) for d in self.data ])
    

def test_convergence(beta=1.0):

    tdcs = []
    imfs = []
    dlrs = []

    tols = 10.**(-np.arange(1, 12))
    #tols = [1e-8, 1e-9]

    m = MeshDLRImFreq(beta=beta, statistic='Fermion', eps=1e-14, w_max=8.0)
    G_w = Gf(mesh=m, target_shape=[2, 2])
    G_w << inverse(iOmega_n - 0.4 - SemiCircular(1.0))
    G_dlr = make_gf_dlr(G_w)

    # Needed for imtime L2 norm calc
    poles, residues, beta, Z =_gf_dlr_to_data(G_dlr)
    sop = SumOfSimplePoles(poles=poles, residues=residues)

    for tol in tols:
        print(f"Testing convergence with tol = {tol:+2.2E}")

        t_tdc = time.time()
        tdc = TriqsDLRCompression(G_w, tol=tol, nonlinear_post_optimize=False)
        tdc.runtime = time.time() - t_tdc
        tdc.n_poles = len(tdc.poles)
        tdcs.append(tdc)

        t_imf = time.time()
        imf = Dummy()
        imf.poles, imf.residues = \
            approximate_gf_imfreq_with_fixed_error_tolerance(G_w, tol=tol)
        
        # Hack to compute the imtime L2 error of the imfreq approximation
        # by constructing a sum of simple poles from the imfreq approximation 
        # and comparing it to the original sum of simple poles 
        # from the DLR representation, using the imtime L2 norm.
        imf_sop = SumOfSimplePoles(poles=imf.poles, residues=imf.residues)
        imf.error = (sop - imf_sop).imtime_l2_norm(beta=beta)

        imf.runtime = time.time() - t_imf
        imf.n_poles = len(imf.poles)    
        imfs.append(imf)

        t_dlr = time.time()
        dlr = Dummy()
        dlr.poles, dlr.residues = \
            approximate_gf_imfreq_with_fixed_error_tolerance(G_w, tol=tol)
        
        # Hack to compute the imtime L2 error of the imfreq approximation
        # by constructing a sum of simple poles from the imfreq approximation 
        # and comparing it to the original sum of simple poles 
        # from the DLR representation, using the imtime L2 norm.
        dlr_sop = SumOfSimplePoles(poles=dlr.poles, residues=dlr.residues)
        dlr.error = (sop - dlr_sop).imtime_l2_norm(beta=beta)

        dlr.runtime = time.time() - t_dlr
        dlr.n_poles = len(dlr.poles)    
        dlrs.append(dlr)


    tdcs = ListDummy(tdcs)
    imfs = ListDummy(imfs)
    dlrs = ListDummy(dlrs)

    print(f'tdcs.error = {tdcs.error}')
    print(f'tdcs.runtime = {tdcs.runtime}')

    from matplotlib import pyplot as plt

    plt.figure(figsize=(3.25, 8))
    subp = [3, 1, 1]

    ax = plt.subplot(*subp); subp[-1] += 1
    plt.title(r'$\beta = '+ f'{m.beta}$')
    plt.plot(tols, tdcs.error, 's-', label='TriqsDLRCompression')
    plt.plot(tols, imfs.error, 'o-', label='ImFreq Approximation')
    plt.plot(tols, dlrs.error, '^-', label='DLR Approximation')
    plt.plot(tols, tols, 'k-')
    plt.loglog()
    plt.grid(True)
    #plt.axis('square')
    plt.xlabel('Tolerance')
    plt.ylabel('L2 Error (imtime)')

    plt.subplot(*subp, sharex=ax); subp[-1] += 1
    plt.plot(tols, tdcs.n_poles, 's-', label='TriqsDLRCompression')
    plt.plot(tols, imfs.n_poles, 'o-', label='ImFreq Approximation')
    plt.plot(tols, dlrs.n_poles, '^-', label='DLR Approximation')
    plt.semilogx()
    plt.grid(True)
    plt.xlabel('Tolerance')
    plt.ylabel('Number of Poles')
    plt.legend(loc='best', fontsize=9)

    plt.subplot(*subp, sharex=ax); subp[-1] += 1
    plt.plot(tols, tdcs.runtime, 's-', label='TriqsDLRCompression')
    plt.plot(tols, imfs.runtime, 'o-', label='ImFreq Approximation')
    plt.plot(tols, dlrs.runtime, '^-', label='DLR Approximation')
    plt.loglog()
    plt.grid(True)
    plt.xlabel('Tolerance')
    plt.ylabel('Runtime (sec)')

    plt.tight_layout()
    plt.savefig(f'figure_convergence_beta_{m.beta}.pdf')
    #plt.show()


if __name__ == "__main__":

    betas = [1.0, 2.0, 4.0, 8.0]
    betas += [16.0, 32.0, 64.0, 128.0]

    for beta in betas:
        test_convergence(beta=beta)