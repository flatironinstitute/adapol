""" Implementation of the AAA algorithm for barycentric rational approximation.

As well as a conjugate constrained version of the AAA algorithm, which is used for pole fitting in the imaginary time domain.

Author: Hugo U. R. Strand, 2026 
"""


import numpy as np

from scipy.linalg import eigvals as scipy_eigvals

class BarycentricRationalApproximation:

    """ Barycentric rational approximation with AAA algorithm.

    YUJI NAKATSUKASA, OLIVIER SETE, AND LLOYD N. TREFETHEN
    SIAM J. SCI. COMPUT. Vol. 40, No. 3, pp. A1494–A1522 (2018)
    DOI. 10.1137/16M1106122

    Author: Hugo U. R. Strand, 2026
    """

    def __init__(self, z=np.array([]), f=np.array([]), w=np.array([])):
        
        assert(len(z) == len(w))
        assert(len(z) == f.shape[0])

        self.z = z
        self.f = f
        self.w = w


    def fast_eval(self, Z):
        """ Evaluate the rational approximation at points Z.
        Assuming that Z does not contain any support points. """
        wC = self.w[None, :] / (Z[:, None] - self.z[None, :])
        n = np.einsum('j...,kj->k...', self.f, wC)
        d = np.sum(wC, axis=1)
        r = np.einsum('k...,k->k...', n, 1/d)
        return r


    def __call__(self, Z):
        """ Evaluate the rational approximation at points Z.
        Also handle cases when Z is a support point, and
        then return the corresponding f value. """

        ZZ = Z[:, None] - self.z[None, :]

        # Find evaluation points that are support points
        idxs = np.nonzero(ZZ == 0.) 
        ZZ[idxs] = 1. # Set zeros to unity before inversion

        wC = self.w[None, :] / ZZ

        # For evaluation point in the support set
        # only return the corresponding value
        ridxs = idxs[0]
        wC[ridxs, :] = 0.0
        wC[idxs] = 1.0
        
        n = np.einsum('j...,kj->k...', self.f, wC)
        d = np.sum(wC, axis=1)
        r = np.einsum('k...,k->k...', n, 1/d)
        return r


    def aaa_step(self, Z, F, R):
        
        # Find largest residual point
        if R.ndim == 1:
            idx = np.argmax(np.abs(R))
        else:
            axis = tuple(range(1, R.ndim))
            idx = np.argmax(np.max(np.abs(R), axis=axis))

        residual = np.max(np.abs(R[idx]))

        # Use this point as new support point, and update the interpolation set
        z_new = Z[idx]
        f_new = F[idx]

        self.z = np.append(self.z, z_new)
        self.f = np.append(self.f, [f_new], axis=0)

        #print(f'AAA: Added support point z = {z_new:2.2E}')

        # Remove support point from fitting set
        Z = np.delete(Z, idx)
        F = np.delete(F, idx, axis=0)

        self.w = self.__fit_weights(Z, F)
        
        R = F - self.fast_eval(Z) # Recompute residual

        return Z, F, R


    def __fit_weights(self, Z, F, scalar=False):

        C = 1.0 / (Z[None, :] - self.z[:, None]) # Cachy matrix, Eq. (3.7) in [1]

        A = np.einsum('k...,jk->jk...', F, C) - \
            np.einsum('jk,j...->jk...', C, self.f)

        A = A.reshape((A.shape[0], -1))

        U, S, Vh = np.linalg.svd(A, full_matrices=False)
        #print(f'AAA: Smallest singular value {S[-1]:2.2E}')
        assert( U.shape[1] > 0 )
        w = U[:, -1].conjugate()

        return w


    def poles_and_residues(self):

        n = len(self.z)

        A = np.block([
            [ np.zeros((1,1)), self.w[None, :] ],
            [ np.ones((n, 1)), np.diag(self.z) ]] )

        B = np.diag(np.concatenate(([0.0], np.ones(n))))

        poles = scipy_eigvals(A, B, overwrite_a=True)

        poles = poles[np.isfinite(poles)]

        # Calculate residues by evaluating the function at points close to the poles
        dz = 1e-5 * np.exp(2j*np.pi*np.arange(1, 5)/4)
        Z = poles[:, None] + dz[None, :]
        
        shape = [len(poles), 4] + list(self.f.shape[1:])
        residues = np.einsum(
            'pf...,f->p...', self.fast_eval(Z.flatten()).reshape(shape), dz / 4)

        return poles, residues
        

    def remove_froissart_doublets(self, Z, F, tol=None):

        if tol is None:
            tol = 1e-13

        poles, residues = self.poles_and_residues()

        if residues.ndim == 1:
            ridxs = np.nonzero(np.abs(residues) < tol)
        else:
            axis = tuple(range(1, residues.ndim))
            ridxs = np.nonzero(np.max(np.abs(residues), axis=axis) < tol)

        if len(ridxs[0]) == 0:
            #print(f'AAA: No Froissart doublets found with residues smaller than {tol:2.2E}')
            return 0, Z, F

        print(f'AAA: removing residues {residues[ridxs]} with poles {poles[ridxs]}')
        print(f'AAA: Removing {len(ridxs)} Froissart doublets with residues smaller than {tol:2.2E}')

        dists = np.abs(self.z[:, None] - poles[None, ridxs])
        pidxs = np.unique(np.argmin(dists, axis=0))

        # Put points back to the fitting set
        Z = np.concatenate((Z, self.z[pidxs]))
        F = np.concatenate((F, self.f[pidxs]))

        # Remove points in support set
        self.z = np.delete(self.z, pidxs)
        self.f = np.delete(self.f, pidxs, axis=0)

        self.w = self.__fit_weights(Z, F)

        # Recompute residual
        R = F - self.fast_eval(Z)
        residual = np.max(np.abs(R))
        print(f'AAA: After removing {len(pidxs)} support points the residual is {residual:2.2E}')

        return len(pidxs), Z, F


class ConjugatedBarycentricRationalApproximation:

    def __init__(self, z=np.array([]), f=np.array([]), w=np.array([])):
        
        assert(len(z) == len(w))
        assert(len(z) == f.shape[0])
        assert(f.ndim == 1 or f.ndim == 3) # Only support scalar and matrix valued f (with known hermitian conjugation relation)

        self.z = z
        self.f = f
        self.w = w


    def fast_eval(self, Z):

        wC = self.w[None, :] / (Z[:, None] - self.z[None, :])
        wCbar = self.w[None, :].conjugate() / (Z[:, None] - self.z[None, :].conjugate())

        if self.f.ndim == 1:
            f_bar = self.f.conjugate()
        elif self.f.ndim == 3:
            f_bar = np.transpose(self.f, (0, 2, 1)).conjugate()
        else:
            raise NotImplementedError("Only scalar and matrix valued f are supported for ConjugatedBarycentricRationalApproximation")

        #n = np.sum((self.f[None, ...] * wC + f_bar[None, ...] * wCbar   ), axis=1)
        n = np.einsum('j...,kj->k...', self.f, wC) + \
            np.einsum('j...,kj->k...', f_bar, wCbar)
        d = np.sum(wC + wCbar, axis=1)
        r = np.einsum('k...,k->k...', n, 1/d)
        return r


    def __call__(self, Z):
        
        ZZ    = Z[:, None] - self.z[None, :]
        ZZbar = Z[:, None] - self.z[None, :].conjugate()

        idxs = np.nonzero(ZZ == 0.)
        idxs_bar = np.nonzero(ZZbar == 0.)

        ZZ[idxs] = 1.
        ZZbar[idxs_bar] = 1.

        #C    = 1.0 / (Z[:, None] - self.z[None, :]) # Cachy matrix, Eq. (3.7) in [1]
        #Cbar = 1.0 / (Z[:, None] - self.z[None, :].conjugate())

        C    = 1.0 / ZZ # Cachy matrix, Eq. (3.7) in [1]
        Cbar = 1.0 / ZZbar

        wC = self.w[None, :] * C
        wCbar = self.w[None, :].conjugate() * Cbar

        #ridxs = idxs[:, 0]
        ridxs = idxs[0]
        wC[ridxs, :] = 0.0
        wCbar[ridxs, :] = 0.0
        wC[idxs] = 1.0

        #ridxs_bar = idxs_bar[:, 0]
        ridxs_bar = idxs_bar[0]
        wC[ridxs_bar, :] = 0.0
        wCbar[ridxs_bar, :] = 0.0
        wCbar[idxs_bar] = 1.0

        if self.f.ndim == 1:
            f_bar = self.f.conjugate()
        elif self.f.ndim == 3:
            f_bar = np.transpose(self.f, (0, 2, 1)).conjugate()
        else:
            raise NotImplementedError("Only scalar and matrix valued f are supported for ConjugatedBarycentricRationalApproximation")

        #n = np.sum((self.f[None, ...] * wC + f_bar[None, ...] * wCbar), axis=1)
        n = np.einsum('j...,kj->k...', self.f, wC) + \
            np.einsum('j...,kj->k...', f_bar, wCbar)
        d = np.sum(wC + wCbar, axis=1)
        r = np.einsum('k...,k->k...', n, 1/d)
        return r


    def __fit_weights(self, Z, F, scalar=False):

        # Tensor valued F & f case

        C    = 1.0 / (Z[None, :] - self.z[:, None]) # Cachy matrix, Eq. (3.7) in [1]
        Cbar = 1.0 / (Z[None, :] - self.z[:, None].conjugate())

        I = np.eye(len(self.z))

        K    = np.vstack((I, +1j*I))
        Kbar = np.vstack((I, -1j*I))

        #print(f'I.shape = {I.shape}, C.shape = {C.shape}, K.shape = {K.shape}, F.shape = {F.shape}')
        #print(f'z.shape = {self.z.shape}, f.shape = {self.f.shape}, w.shape = {self.w.shape}')

        if self.f.ndim == 1:
            fbar = self.f.conjugate()
        elif self.f.ndim == 3:
            fbar = np.transpose(self.f, (0, 2, 1)).conjugate()
        else:
            raise NotImplementedError("Only scalar and matrix valued f are supported for ConjugatedBarycentricRationalApproximation")

        A = np.einsum('xk,k...->xk...', (K @ C + Kbar @ Cbar), F) + \
            - np.einsum('xj,j...,jk->xk...', K, self.f, C) \
            - np.einsum('xj,j...,jk->xk...', Kbar, fbar, Cbar) \

        A = A.reshape((A.shape[0], -1))
        A = np.hstack((A.real, A.imag))

        #print(f'A.dtype = {A.dtype}, A.shape = {A.shape}')

        U, S, Vh = np.linalg.svd(A, full_matrices=False)
        #print(f'AAA: Smallest singular value {S[-1]:2.2E}')
        assert( U.shape[1] > 0 )
        x = U[:, -1]
        w = x @ K

        return w


    def aaa_step(self, Z, F, R, tol_conj=1e-12):
        
        # Find largest residual point
        if R.ndim == 1:
            idx = np.argmax(np.abs(R))
        elif R.ndim == 3:
            idx = np.argmax(np.max(np.abs(R), axis=(1, 2)))
        else:
            raise NotImplementedError("Only scalar and matrix valued functions are supported for ConjugatedBarycentricRationalApproximation")
        
        self.residual = np.max(np.abs(R[idx]))

        #print(f'AAA: Largest residual {residual:2.2E} at z = {Z[idx]:2.2E}, idx = {idx}')

        # Use this point as new support point, and update the interpolation set
        z_new = Z[idx]
        f_new = F[idx]

        self.z = np.append(self.z, z_new)
        self.f = np.append(self.f, [f_new], axis=0)

        #print(f'AAA: Added support point z = {z_new:2.2E}')

        # Remove support point from fitting set
        Z = np.delete(Z, idx)
        F = np.delete(F, idx, axis=0)

        # Find conjugate data point and remove it from fitting set as well
        z_conj = np.conjugate(z_new)
        diff = np.abs(Z - z_conj)
        idx_conj = np.argmin(diff)
        if diff[idx_conj] < tol_conj:
            #print(f'AAA: Removing conjugate point z = {Z[idx_conj]:2.2E}')
            Z = np.delete(Z, idx_conj)
            F = np.delete(F, idx_conj, axis=0)

        self.w = self.__fit_weights(Z, F)
        
        # Recompute residual
        R = F - self.fast_eval(Z)

        return Z, F, R    


    def poles_and_residues(self):

        ww = np.concatenate((self.w, self.w.conjugate()))
        zz = np.concatenate((self.z, self.z.conjugate()))

        n = len(zz)

        A = np.block([
            [ np.zeros((1,1)), ww[None, :] ],
            [ np.ones((n, 1)), np.diag(zz) ]] )

        B = np.diag(np.concatenate(([0.0], np.ones(n))))

        poles = scipy_eigvals(A, B, overwrite_a=True)

        poles = poles[np.isfinite(poles)]

        sidx = np.argsort(poles.real)
        poles = poles[sidx]

        # Calculate residues by evaluating the function at points close to the poles
        dz = 1e-5 * np.exp(2j*np.pi*np.arange(1, 5)/4)
        Z = poles[:, None] + dz[None, :]

        shape = [len(poles), 4] + list(self.f.shape[1:])
        residues = np.einsum(
            'pf...,f->p...', self.fast_eval(Z.flatten()).reshape(shape), dz / 4)

        return poles, residues


    def remove_froissart_doublets(self, Z, F, tol=None, imag_tol=1e-4):

        if tol is None:
            tol = 1e-13

        poles, residues = self.poles_and_residues()

        # Find small residues
        if residues.ndim == 1:
            ridxs = np.nonzero(np.abs(residues) < tol)
        elif residues.ndim == 3:
            ridxs = np.nonzero(np.max(np.abs(residues), axis=(1, 2)) < tol)
        else:
            raise NotImplementedError("Only scalar and matrix valued functions are supported for ConjugatedBarycentricRationalApproximation")

        if imag_tol is not None:
            ridxs_im = np.nonzero(np.abs(poles.imag) > imag_tol)
            ridxs = (np.unique(np.concatenate((ridxs[0], ridxs_im[0]))),)

        if len(ridxs[0]) <= 1:
            # Since the conjugated support points produce pole pairs, 
            # do not remove a single pole with small residue.
            # This is likely not a Froissart doublet.
            
            #print(f'AAA: No Froissart doublets found with residues smaller than {tol:2.2E}')
            return 0, Z, F

        print(f'AAA: Found {len(ridxs[0])} residues < {tol}.')

        zz = np.concatenate((self.z, self.z.conjugate()))
        dists = np.abs(zz[:, None] - poles[ridxs][None, :])
        pidxs = np.argmin(dists, axis=0)
        pidxs = np.unique(np.mod(pidxs, len(self.z)))

        print(f'AAA: Found {len(pidxs)} adjacent support points to remove.')

        assert( len(pidxs) > 0 )

        # Put points back to the fitting set
        Z = np.concatenate((Z, self.z[pidxs]))
        F = np.concatenate((F, self.f[pidxs]))

        # Remove points in support set
        self.z = np.delete(self.z, pidxs)
        self.f = np.delete(self.f, pidxs, axis=0)

        self.w = self.__fit_weights(Z, F)

        # Recompute residual
        R = F - self.fast_eval(Z)
        self.residual = np.max(np.abs(R))
        print(f'AAA: After removing {len(pidxs)} support points the residual is {self.residual:2.2E}')

        return len(pidxs), Z, F


def aaa_bra(Z, F, tol=None, max_steps=None, constrained=False, 
            cleanup=True, cleanup_residue_tol=1e-13, cleanup_imag_tol=1e-4,
            verbose=True):

    assert(len(Z) == len(F))

    max_max_steps = len(Z) // 2 - 1 if constrained else len(Z) - 1

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
            print(f'AAA: Error {residual:2.2E} using {len(bra.z)} support and {len(Z)} fitting points (step {step}/{max_steps})')

        if tol is not None and residual <= tol:
            if verbose:
                print(f"AAA: Converged after {step} steps with error {residual:2.2E}.")
            break

    if cleanup:
        opts = dict(tol=cleanup_residue_tol)
        if constrained:
            opts['imag_tol'] = cleanup_imag_tol

        n_removed = 1
        while(n_removed > 0):
            n_removed, Z, F = bra.remove_froissart_doublets(Z, F, **opts)

    if step == max_steps and tol is not None and residual > tol:
        print(f"AAA: Warning! Failed to converge after {max_steps} steps. Final error {residual:2.2E} larger than tolerance {tol:2.2E}.")   

    bra.aaa_steps = step

    return bra

