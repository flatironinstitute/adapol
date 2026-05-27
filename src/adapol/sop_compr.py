""" Wrapper around AAA-BRA for sum of poles compression in the imaginary time domain.

Author: Hugo U. R. Strand, 2026
"""

import numpy as np


from .aaa import aaa
from .sop import SumOfSimplePoles
from .adapol import _equispaced_matsubara_frequecy_grid


class SumOfPolesCompression:

    def __init__(self, poles, residues, beta, Z=None, tol=1e-10, 
                 nonlinear_optimize=False, nonlinear_post_optimize=False, 
                 max_upwind_steps=4, verbose=True):
        
        self.tol = tol
        self.nonlinear_optimize = nonlinear_optimize
        self.nonlinear_post_optimize = nonlinear_post_optimize
        self.verbose = verbose

        self.beta = beta
        self.sop = SumOfSimplePoles(poles=poles, residues=residues)

        if Z is None:
            Z = _equispaced_matsubara_frequecy_grid(poles, beta)

        self.Z = Z
        self.F = self.sop(self.Z)

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
                print(f'Adapol: Step {step}/{max_upwind_steps}, AAA steps = {aaa_steps}, error = {err:2.2E} with tol = {aaa_tol}.')
                print('-'*72)

            aaa_tol = None
            aaa_max_steps = aaa_steps + 1

        if verbose:
            print(f'Adapol: Error {err:2.2E} for {aaa_steps} AAA steps (Error {aaa_err:2.2E} no opt) c.f. tol {tol:2.2E}.')

        if step == max_upwind_steps and err >= tol:
            raise ValueError(f"Adapol: Compression failed to achieve the desired accuracy {tol:2.2E} after {max_upwind_steps} steps, with final error {err:2.2E}. Consider increasing max_upwind_steps or relaxing tol.")

        n_not_converged = 0
        n_converged = aaa_steps

        poles_conv = poles.copy()
        residues_conv = residues.copy()
        err_conv = err

        err_not_conv = float('inf')

        while(n_not_converged + 1 != n_converged):

            n_test = (n_not_converged + n_converged) // 2
            #print(f"Adapol: Running AAA with max_steps = {n_test}, in interval [{n_not_converged}, {n_converged}].")
            poles, residues, aaa_steps, aaa_err = self.aaa_compress(max_steps=n_test)

            #print(f'Adapol: AAA with max_steps = {n_test} gives error {aaa_err:2.2E}.')

            if self.nonlinear_optimize:
                err, poles, residues = self.nonlinear_optimization_of_poles_and_weights(poles, residues)
            else:
                err, residues = self.lstsq_weight_optimization(poles)

            if verbose:
                print(f'Adapol: Error {err:2.2E} for {n_test} AAA steps (Error {aaa_err:2.2E} no opt) c.f. tol {tol:2.2E}.')

            if err < tol:
                n_converged = n_test
                poles_conv = poles.copy()
                residues_conv = residues.copy()
                err_conv = err
            else:
                n_not_converged = n_test
                err_not_conv = err

            #print(f"Adapol: aaa_max_steps = {n_converged:2d} is converged with error {err_conv:2.2E} < tol {tol:2.2E}.")
            #print(f'Adapol: aaa_max_steps = {n_not_converged:2d} is not converged, error {err_not_conv:2.2E} > tol {tol:2.2E}.')

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
            print(f'Adapol: Compression finished with {n_converged} AAA steps and error {err_conv:2.2E}.')
        self.poles, self.residues, self.aaa_steps, self.error = poles_conv, residues_conv, n_converged, err_conv


    def aaa_compress(self, tol=None, max_steps=None, cleanup=True, cleanup_residue_tol=1e-13, cleanup_imag_tol=1e-4):

        bra = aaa(
            self.Z, self.F, tol=tol, max_steps=max_steps, constrained=True,
            cleanup=cleanup, cleanup_residue_tol=cleanup_residue_tol, cleanup_imag_tol=cleanup_imag_tol,
            verbose=self.verbose)

        poles, residues = bra.poles_and_residues()
        poles = poles.real

        return poles, residues, bra.aaa_steps, bra.residual


    def lstsq_weight_optimization(self, poles):

        sop_opt = self.sop.best_imtime_lstsq_l2_norm_approximation_using_poles(poles, self.beta)
        residues = sop_opt.R
        err = (sop_opt - self.sop).imtime_l2_norm(self.beta)
        return err, residues
    

    def nonlinear_optimization_of_poles_and_weights(self, poles, residues):

        sop_opt = self.sop.best_imtime_non_linear_lstsq_l2_norm_approximation_using_pole_guess(
                poles=poles, beta=self.beta, verbose=False)
        
        err = sop_opt.err
        poles = sop_opt.p
        residues = sop_opt.R

        return err, poles, residues