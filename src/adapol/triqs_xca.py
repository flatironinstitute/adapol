""" Wrapper around AAA-BRA for pole compression in the imaginary time domain, using the Triqs library.

Author: Hugo U. R. Strand, 2026
"""

import numpy as np


class TriqsDLRCompression:

    def __init__(self, G, tol=1e-14, 
                 nonlinear_optimize=False, nonlinear_post_optimize=False, 
                 max_upwind_steps=4, verbose=True):

        self.G = G
        self.tol = tol
        self.nonlinear_optimize = nonlinear_optimize
        self.nonlinear_post_optimize = nonlinear_post_optimize
        self.verbose = verbose

        from triqs.gfs import MeshDLR, make_gf_dlr

        self.G_dlr = G if type(G.mesh) == MeshDLR else make_gf_dlr(G)
        self.dlr_freq = np.array([float(w) for w in self.G_dlr.mesh])
        self.G_dlr_coeff = self.G_dlr.data.copy()
        self.beta = self.G_dlr.mesh.beta

        poles = self.dlr_freq / self.beta
        residues = self.G_dlr_coeff.copy()

        from triqs.gfs import MeshDLRImFreq, make_gf_dlr_imfreq

        self.G_w = G if type(G.mesh) == MeshDLRImFreq else make_gf_dlr_imfreq(G)
        self.Z = np.array([complex(w) for w in self.G_w.mesh])

        from .sop_compr import SumOfPolesCompression
        
        self.sop_comp = SumOfPolesCompression(
            poles=poles, residues=residues,
            Z=self.Z,
            beta=self.beta, tol=tol, 
            nonlinear_optimize=nonlinear_optimize, 
            nonlinear_post_optimize=nonlinear_post_optimize, 
            max_upwind_steps=max_upwind_steps, verbose=verbose)
        
        sc = self.sop_comp
        self.poles, self.residues, self.aaa_steps, self.error = sc.poles, sc.residues, sc.aaa_steps, sc.error