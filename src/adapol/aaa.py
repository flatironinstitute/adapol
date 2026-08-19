""" Implementation of the AAA algorithm for barycentric rational approximation.

As well as a conjugate constrained version of the AAA algorithm, 
which is used for pole fitting of Matsubara frequency Green's functions.

Author: Hugo U. R. Strand, 2026 
"""


import numpy as np

from .bra import BarycentricRationalApproximation
from .bra import ConjugatedBarycentricRationalApproximation


def aaa(Z, F, tol=None, max_steps=None, constrained=False, 
        cleanup=True, cleanup_residue_tol=1e-13, cleanup_imag_tol=1e-4,
        verbose=True, prefix=''):

    """ Implementation of the AAA algorithm for barycentric rational approximation. 
    
    Parameters
    ----------
    Z : array_like, shape (n,)
        The support points.
    F : array_like, shape (n, ...)
        The function values at the support points.
    tol : float, optional
        The tolerance on the AAA residual, i.e. on the maximum absolute deviation
        from the data over the sample points not yet used as support points. The
        iteration stops once the residual drops below the tolerance.
    max_steps : int, optional
        The maximum number of AAA steps.
    constrained : bool, optional
        Whether to use the conjugate pair constrained version of the algorithm.
    cleanup : bool, optional
        Whether to remove Froissart doublets.
    cleanup_residue_tol : float, optional
        The tolerance for identifying small residues.
    cleanup_imag_tol : float, optional
        The tolerance for identifying poles with non-negligible imaginary part.
    verbose : bool, optional
        Whether to print progress information.
    prefix : str, optional
        String prepended to each printed line, e.g. to indent the output when
        the algorithm is run as a sub step of a larger calculation.

    Returns
    -------
    bra : BarycentricRationalApproximation, ConjugatedBarycentricRationalApproximation
        The approximating function.
    """

    assert(len(Z) == len(F))

    #max_max_steps = len(Z) // 2 - 1 if constrained else len(Z) - 1 # For data with conjugated data points.
    max_max_steps = len(Z) - 1

    if max_steps is None: 
        max_steps = max_max_steps

    assert(max_steps <= max_max_steps)

    Z = Z.copy()
    F = F.copy()
    R = F.copy()

    # Empty value vector with the same shape as F (to enable appending)
    f0 = np.array([]).reshape([0] + list(F.shape[1:]))

    if constrained:
        bra = ConjugatedBarycentricRationalApproximation(f=f0)
    else:
        bra = BarycentricRationalApproximation(f=f0)

    for step in range(1, max_steps+1):
        Z, F, R = bra.aaa_step(Z, F, R)
        residual = np.max(np.abs(R))
        if verbose:
            print(f'{prefix}AAA: Residual {residual:2.2E} using {len(bra.z)} support and {len(Z)} fitting points (step {step}/{max_steps})')

        if tol is not None and residual <= tol:
            if verbose:
                print(f"{prefix}AAA: Converged after {step} steps with residual {residual:2.2E}.")
            break

    if cleanup:
        opts = dict(tol=cleanup_residue_tol, verbose=verbose, prefix=prefix)
        if constrained:
            opts['imag_tol'] = cleanup_imag_tol

        n_removed = 1
        while(n_removed > 0):
            n_removed, Z, F = bra.remove_froissart_doublets(Z, F, **opts)

    if step == max_steps and tol is not None and residual > tol:
        print(f"{prefix}AAA: Warning! Failed to converge after {max_steps} steps. Final residual {residual:2.2E} larger than tolerance {tol:2.2E}.")   

    bra.aaa_steps = step

    return bra

