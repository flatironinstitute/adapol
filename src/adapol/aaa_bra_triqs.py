""" Wrapper around AAA-BRA for pole compression in the imaginary time domain, using the Triqs library.

Author: Hugo U. R. Strand, 2026
"""

import numpy as np

from scipy.optimize import minimize as scipy_minimize

from triqs.gf import MeshDLR, MeshDLRImFreq
from triqs.gf import make_gf_dlr, make_gf_dlr_imfreq

from adapol.fit_utils_dlr import erroreval_dlr, get_weight_dlr, exp_quadrature
from adapol.aaa_bra import aaa_bra

class TriqsDLRCompression:

    def __init__(self, G, tol=1e-14, 
                 nonlinear_optimize=False, nonlinear_post_optimize=False, 
                 max_upwind_steps=4, verbose=True):

        self.G = G
        self.tol = tol
        self.nonlinear_optimize = nonlinear_optimize
        self.nonlinear_post_optimize = nonlinear_post_optimize
        self.verbose = verbose

        self.G_dlr = G if type(G.mesh) == MeshDLR else make_gf_dlr(G)
        self.dlr_freq = np.array([float(w) for w in self.G_dlr.mesh])
        self.G_dlr_coeff = self.G_dlr.data.copy()

        if self.G_dlr_coeff.ndim == 1:
            self.G_dlr_coeff = self.G_dlr_coeff.reshape(-1, 1, 1)

        self.G_w = G if type(G.mesh) == MeshDLRImFreq else make_gf_dlr_imfreq(G)
        self.Z = np.array([complex(w) for w in self.G_w.mesh])
        self.F = self.G_w.data.copy()

        aaa_tol = tol
        aaa_max_steps = None

        for step in range(1, max_upwind_steps+1):

            poles, residues, aaa_steps, aaa_err = self.aaa_compress(tol=aaa_tol, max_steps=aaa_max_steps)
            if self.nonlinear_optimize:
                err, poles, residues = self.nonlinear_optimization_of_poles_and_weights(poles, residues)
            else:
                err, residues = self.lstsq_weight_optimization(poles)

            if err < tol:
                break

            if verbose:
                print('-'*72)
                print(f'TDC: Step {step}/{max_upwind_steps}, AAA steps = {aaa_steps}, error = {err:2.2E} with tol = {aaa_tol}.')
                print('-'*72)

            aaa_tol = None
            aaa_max_steps = aaa_steps + 1

        if verbose:
            print(f'TDC: Error {err:2.2E} for {aaa_steps} AAA steps (Error {aaa_err:2.2E} no opt) c.f. tol {tol:2.2E}.')

        if step == max_upwind_steps and err >= tol:
            raise ValueError(f"TDC: Compression failed to achieve the desired accuracy {tol:2.2E} after {max_upwind_steps} steps, with final error {err:2.2E}. Consider increasing max_upwind_steps or relaxing tol.")

        n_not_converged = 0
        n_converged = aaa_steps

        poles_conv = poles.copy()
        residues_conv = residues.copy()
        err_conv = err

        err_not_conv = float('inf')

        while(n_not_converged + 1 != n_converged):

            n_test = (n_not_converged + n_converged) // 2
            #print(f"TDC: Running AAA with max_steps = {n_test}, in interval [{n_not_converged}, {n_converged}].")
            poles, residues, aaa_steps, aaa_err = self.aaa_compress(max_steps=n_test)

            #print(f'TDC: AAA with max_steps = {n_test} gives error {aaa_err:2.2E}.')

            if self.nonlinear_optimize:
                err, poles, residues = self.nonlinear_optimization_of_poles_and_weights(poles, residues)
            else:
                err, residues = self.lstsq_weight_optimization(poles)

            if verbose:
                print(f'TDC: Error {err:2.2E} for {n_test} AAA steps (Error {aaa_err:2.2E} no opt) c.f. tol {tol:2.2E}.')

            if err < tol:
                n_converged = n_test
                poles_conv = poles.copy()
                residues_conv = residues.copy()
                err_conv = err
            else:
                n_not_converged = n_test
                err_not_conv = err

            #print(f"TDC: aaa_max_steps = {n_converged:2d} is converged with error {err_conv:2.2E} < tol {tol:2.2E}.")
            #print(f'TDC: aaa_max_steps = {n_not_converged:2d} is not converged, error {err_not_conv:2.2E} > tol {tol:2.2E}.')

        if self.nonlinear_post_optimize:
            """ Exploit that the non-linear optimization often can reduce the pole no by one.

            Try first with one pole less and then with the same number of poles, 
            and keep least no of poles that satisfy the tolerance. """
            
            n_tests = [n_converged - 1, n_converged] if n_converged > 1 else [n_converged]

            for n_test in n_tests:
                poles, residues, _, _ = self.aaa_compress(max_steps=n_test)
                err, poles, residues = self.nonlinear_optimization_of_poles_and_weights(poles, residues)
                if err < tol:
                    n_converged = n_test
                    poles_conv = poles.copy()
                    residues_conv = residues.copy()
                    err_conv = err
                    break

        if verbose:
            print(f'TDC: Compression finished with {n_converged} AAA steps and error {err_conv:2.2E}.')
        self.poles, self.residues, self.aaa_steps, self.error = poles_conv, residues_conv, n_converged, err_conv


    def aaa_compress(self, tol=None, max_steps=None, cleanup=True, cleanup_residue_tol=1e-13, cleanup_imag_tol=1e-4):

        bra = aaa_bra(
            self.Z, self.F, tol=tol, max_steps=max_steps, constrained=True,
            cleanup=cleanup, cleanup_residue_tol=cleanup_residue_tol, cleanup_imag_tol=cleanup_imag_tol,
            verbose=self.verbose)

        poles, residues = bra.poles_and_residues()
        poles = poles.real

        return poles, residues, bra.aaa_steps, bra.residual


    def imtime_l2_error(self, poles, residues):
        err, _ =  erroreval_dlr(poles, self.dlr_freq, self.G_dlr_coeff, self.G.mesh.beta, weights=-residues)
        return err


    def lstsq_weight_optimization(self, poles):
        residues, _ = get_weight_dlr(poles, self.dlr_freq, self.G_dlr_coeff, self.G.mesh.beta)
        residues *= -1.
        err, _ =  erroreval_dlr(poles, self.dlr_freq, self.G_dlr_coeff, self.G.mesh.beta, weights=-residues)
        return err, residues
    

    def nonlinear_optimization_of_poles_and_weights(self, poles, residues):

        tau_nodes, tau_weights = exp_quadrature(max(2 * np.max(np.abs(np.concatenate(
            [poles * self.G.mesh.beta, self.dlr_freq]))), 1.0))
        
        def func(poles):
            err, jac = erroreval_dlr(
                poles, self.dlr_freq, self.G_dlr_coeff, self.G.mesh.beta,
                tau_nodes=tau_nodes, tau_weights=tau_weights) 
            return err, jac

        res = scipy_minimize(
            func, poles, 
            method='L-BFGS-B', 
            jac=True,
            tol=1e-14,
            )
        
        poles = res.x
        err = res.fun

        residues, _ = get_weight_dlr(poles, self.dlr_freq, self.G_dlr_coeff, self.G.mesh.beta)
        residues *= -1.

        return err, poles, residues