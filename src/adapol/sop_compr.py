""" Wrapper around AAA-BRA for sum of poles compression in the imaginary time domain.

Author: Hugo U. R. Strand, 2026
"""

import numpy as np


from .aaa import aaa
from .sop import SumOfSimplePoles
from .adapol import _fermionic_matsubara_frequency_grid


class SumOfPolesCompression:

    def __init__(self, poles, residues, beta, Z=None, tol=1e-10, 
                 nonlinear_optimize=False, nonlinear_post_optimize=False, 
                 max_upwind_steps=4, verbose=True):
        
        self.tol = tol
        self.nonlinear_optimize = nonlinear_optimize
        self.verbose = verbose

        # The post optimization retries the bisection result with one pole less,
        # using the non-linear optimization. When the bisection itself already runs
        # the non-linear optimization, both retries only repeat (deterministic)
        # pipeline passes that the bisection has made, and can not change the
        # result. Hence, drop the post optimization in that case.
        self.nonlinear_post_optimize = nonlinear_post_optimize and not nonlinear_optimize

        self.beta = beta
        self.sop = SumOfSimplePoles(poles=poles, residues=residues)

        if Z is None:
            Z = _fermionic_matsubara_frequency_grid(poles, beta)

        self.Z = Z
        self.F = self.sop(self.Z)

        self._aaa_cache = {}
        self._pipeline_cache = {}

        aaa_tol = tol
        aaa_max_steps = None

        for step in range(1, max_upwind_steps+1):

            poles, residues, aaa_steps, aaa_err, err = self._pipeline(
                n_steps=aaa_max_steps, aaa_tol=aaa_tol)

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
            poles, residues, aaa_steps, aaa_err, err = self._pipeline(n_steps=n_test)

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
            # Exploit that the non-linear optimization often can reduce the pole no by one.
            #
            # Try first with one pole less and then with the same number of poles,
            # and keep least no of poles that satisfy the tolerance.

            n_tests = [n_converged - 1, n_converged] if n_converged > 1 else [n_converged]

            for n_test in n_tests:
                poles, residues, _, _, err = self._pipeline(n_steps=n_test, nonlinear=True)
                if err < tol:
                    n_converged = n_test
                    poles_conv = poles.copy()
                    residues_conv = residues.copy()
                    err_conv = err
                    break

        if verbose:
            print(f'Adapol: Compression finished with {n_converged} AAA steps and error {err_conv:2.2E}.')
        self.poles, self.residues, self.aaa_steps, self.error = poles_conv, residues_conv, n_converged, err_conv


    def _pipeline(self, n_steps=None, aaa_tol=None, nonlinear=None):

        """ Run the AAA pole step followed by the residue step.

        The number of AAA steps is fixed by `n_steps`, or, if `n_steps` is None,
        determined by the AAA error tolerance `aaa_tol`. The residue step is the
        joint non-linear optimization of poles and residues if `nonlinear` is True,
        and the linear least squares fit of the residues otherwise, defaulting to
        the `nonlinear_optimize` flag of the compression.

        Returns (poles, residues, n_steps, aaa_err, err) where `n_steps` is the
        number of AAA steps actually taken, `aaa_err` the AAA error of the pole
        step, and `err` the imaginary time L2 error of the result.

        The result is cached on the number of AAA steps and the kind of residue
        step, since the search for the smallest approximation can revisit step
        counts that already have been run. A tolerance driven pole step is cached
        on the number of steps it takes, which gives the same approximation as
        asking for that number of steps directly. """

        if nonlinear is None:
            nonlinear = self.nonlinear_optimize

        if (n_steps, nonlinear) not in self._pipeline_cache:

            poles, residues, n_steps, aaa_err = self._aaa_poles(n_steps=n_steps, aaa_tol=aaa_tol)

            if nonlinear:
                err, poles, residues = self.nonlinear_optimization_of_poles_and_weights(poles, residues)
            else:
                err, residues = self.lstsq_weight_optimization(poles)

            self._pipeline_cache[(n_steps, nonlinear)] = (poles, residues, n_steps, aaa_err, err)

        poles, residues, n_steps, aaa_err, err = self._pipeline_cache[(n_steps, nonlinear)]

        return poles.copy(), residues.copy(), n_steps, aaa_err, err


    def _aaa_poles(self, n_steps=None, aaa_tol=None):

        """ AAA pole step, cached on the number of AAA steps taken.

        Cached separately from the pipeline, since the poles do not depend on how
        the residues subsequently are determined, and the post optimization redoes
        the residue step for step counts that the search already has run. """

        if n_steps not in self._aaa_cache:
            poles, residues, n_steps, aaa_err = self.aaa_compress(tol=aaa_tol, max_steps=n_steps)
            self._aaa_cache[n_steps] = (poles, residues, n_steps, aaa_err)

        poles, residues, n_steps, aaa_err = self._aaa_cache[n_steps]

        return poles.copy(), residues.copy(), n_steps, aaa_err


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