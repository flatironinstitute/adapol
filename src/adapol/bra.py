""" Implementation of barycentric rational approximation.

As well as a conjugate constrained version, which is used 
for pole fitting of Matsubara frequency Green's functions.

Author: Hugo U. R. Strand, 2026 
"""


import numpy as np
from scipy.linalg import eigvals as scipy_eigvals

from .sop import SumOfSimplePoles


class BarycentricRationalApproximation:

    """ Barycentric rational approximation with AAA algorithm.

    Parameters
    ----------
    z : array_like, shape (m,)
        The support points.
    f : array_like, shape (m, ...)
        The function values at the support points.
    w : array_like, shape (m,)
        The weights of the rational approximation.

    Notes
    -----

    Use the `aaa` function to construct the rational approximation 
    from a given set of points and function values using the AAA algorithm.
    
    References
    ----------

    [1] YUJI NAKATSUKASA, OLIVIER SETE, AND LLOYD N. TREFETHEN
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

        # Cauchy matrix, Eq. (3.7) in [1] multiplied by the weights w
        wC = self.w[None, :] / (Z[:, None] - self.z[None, :])

        return self.__barycentric_eval(wC)


    def __barycentric_eval(self, wC):
        """ Internal helper function for barycentric formula evaluation.
        Shared by fast_eval and __call__."""

        # Evaluate the barycentric formula, Eq. (2.5) in [1]
        n = np.einsum('j...,kj->k...', self.f, wC)
        d = np.sum(wC, axis=1)
        r = np.einsum('k...,k->k...', n, 1/d)
        return r


    def __call__(self, Z):
        """ Evaluate the rational approximation at points Z.
        Also handle cases when Z is a support point, and
        then return the corresponding f value. """

        # Denominator of elements in the Cauchy matrix, Eq. (3.7) in [1]
        ZZ = Z[:, None] - self.z[None, :]

        # Find evaluation points that are support points
        idxs = np.nonzero(ZZ == 0.) 
        ZZ[idxs] = 1. # Set zeros to unity before inversion

        # Cauchy matrix, Eq. (3.7) in [1] multiplied by the weights w
        wC = self.w[None, :] / ZZ

        # For evaluation points in the support set
        # only return the corresponding value
        ridxs = idxs[0]
        wC[ridxs, :] = 0.0
        wC[idxs] = 1.0
        
        return self.__barycentric_eval(wC)


    def aaa_step(self, Z, F, R):
        """ Perform one step of the AAA algorithm. """

        # Find index of largest residual
        if R.ndim == 1:
            idx = np.argmax(np.abs(R))
        else:
            axis = tuple(range(1, R.ndim))
            idx = np.argmax(np.max(np.abs(R), axis=axis))

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
        self.residual = np.max(np.abs(R))
        return Z, F, R


    def __fit_weights(self, Z, F, scalar=False):
        """ Fit the weights w of the rational approximation by an SVD. """

        C = 1.0 / (Z[None, :] - self.z[:, None]) # Cachy matrix, Eq. (3.7) in [1]

        # Construct Löwner matrix A Eq. (3.6) in [1], 
        # such that min_w || A @ w ||_2 gives the optimal weights w
        A = np.einsum('k...,jk->jk...', F, C) - \
            np.einsum('jk,j...->jk...', C, self.f)
        A = A.reshape((A.shape[0], -1))

        # Solve minimization problem using left singular vector with smallest singular value
        U, S, Vh = np.linalg.svd(A, full_matrices=False)
        assert( U.shape[1] > 0 )
        w = U[:, -1].conjugate()

        return w


    def poles_and_residues(self):
        """ Determine the poles and residues of the rational approximation,
        by solving a generalized eigenvalue problem with an arrowhead matrix, 
        and using the Laurent series residue relation. """

        n = len(self.z)

        # Left hand (A) and right hand (B) matrices in Eq. (3.11) in [1]
        # of the generalized eigenvalue problem, A@x = \lambda B@x

        A = np.block([
            [ np.zeros((1,1)), self.w[None, :] ],
            [ np.ones((n, 1)), np.diag(self.z) ]] )

        B = np.diag(np.concatenate(([0.0], np.ones(n))))

        # Obtain the poles as generalized eigenvalues of the pencil (A, B)
        # NB! two eigenvalues are infinite, and needs to be discarded.
        poles = scipy_eigvals(A, B, overwrite_a=True)
        poles = poles[np.isfinite(poles)] # Discard infinite eigenvalues

        # Calculate residues by evaluating the function at points close to the poles
        # See the MATLAB code in Fig. 4.1 of Ref [1].

        # Is this related to a Laurent series expansion of the rational function around the pole?
        
        # TODO: understand this better, and add more comments here.
        # Q: How sensitive is this to the choice of dz?

        dz = 1e-5 * np.exp(2j*np.pi*np.arange(1, 5)/4)
        Z = poles[:, None] + dz[None, :]
        
        shape = [len(poles), 4] + list(self.f.shape[1:])
        residues = np.einsum(
            'pf...,f->p...', self.fast_eval(Z.flatten()).reshape(shape), dz / 4)

        return poles, residues
        

    def remove_froissart_doublets(self, Z, F, tol=None, verbose=False, prefix=''):
        """ Remove Froissart doublets, i.e. poles with small residues, 
        by putting the closest support point of each pole back to the fitting set. """

        if tol is None:
            tol = 1e-13

        poles, residues = self.poles_and_residues()

        # Find indices of small residues
        if residues.ndim == 1:
            ridxs = np.nonzero(np.abs(residues) < tol)
        else:
            axis = tuple(range(1, residues.ndim))
            ridxs = np.nonzero(np.max(np.abs(residues), axis=axis) < tol)

        if len(ridxs[0]) == 0:
            #print(f'AAA: No Froissart doublets found with residues smaller than {tol:2.2E}')
            return 0, Z, F

        if verbose:
            print(f'{prefix}AAA: Found {len(ridxs[0])} residues < {tol}.')

        # Locate the closest support point to each pole with small residue
        dists = np.abs(self.z[:, None] - poles[None, ridxs])
        pidxs = np.unique(np.argmin(dists, axis=0))

        if verbose:
            print(f'{prefix}AAA: Found {len(pidxs)} adjacent support points to remove.')

        # Put points back to the fitting set
        Z = np.concatenate((Z, self.z[pidxs]))
        F = np.concatenate((F, self.f[pidxs]))

        # Remove points in support set
        self.z = np.delete(self.z, pidxs)
        self.f = np.delete(self.f, pidxs, axis=0)

        self.w = self.__fit_weights(Z, F)
        R = F - self.fast_eval(Z) # Recompute residual
        self.residual = np.max(np.abs(R))

        if verbose:
            print(f'{prefix}AAA: After removing {len(pidxs)} support points the residual is {self.residual:2.2E}')

        return len(pidxs), Z, F
    

    def get_sop(self):
        """ Return the rational approximation as a sum of simple poles. """

        poles, residues = self.poles_and_residues()
        sop = SumOfSimplePoles(poles=poles, residues=residues)
        return sop


class ConjugatedBarycentricRationalApproximation:

    """ Conjugated Barycentric rational approximation with AAA algorithm.

    Parameters
    ----------
    z : array_like, shape (m,)
        The support points.
    f : array_like, shape (m,) or (m, n, n)
        The function values at the support points.
    w : array_like, shape (m,)
        The weights of the rational approximation.

    Notes
    -----

    Use the `aaa` function to construct the rational approximation 
    from a given set of points and function values using the AAA algorithm.

    This is a specialization of the standard Barycentric Rational Approximation formula,
    to the case where all support points z, function values f, and weights w, are added in conjugate pairs

    .. math:: 
        r(z) = \\frac{n(z)}{d(z)} = 
        \\left[ \\sum_{j=1}^m \\left(
        \\frac{w_j f_j}{z - z_j} + \\frac{\\bar{w}_j f_j^\\dagger}{z - \\bar{z}_j}
        \\right) \\right] 
        \\Bigg/ 
        \\left[ \\sum_{j=1}^m \\left( 
        \\frac{w_j}{z - z_j} + \\frac{\\bar{w}_j^*}{z - \\bar{z}_j^*}
        \\right) \\right]

    This ensures that :math:`r(\\bar{z}) = r(z)^\\dagger`, 
    which is a property of Green's functions in the Matsubara frequency domain.

    The modified approximation formula was presented in Ref [2] and is a generalization of Ref. [1].

    References
    ----------

    [1] YUJI NAKATSUKASA, OLIVIER SETE, AND LLOYD N. TREFETHEN
    SIAM J. SCI. COMPUT. Vol. 40, No. 3, pp. A1494–A1522 (2018)
    DOI. 10.1137/16M1106122

    [2] Zhen Huang, Denis Golež, Hugo U. R. Strand, Jason Kaye
    SciPost Phys. 19, 121 (2025) doi: 10.21468/SciPostPhys.19.5.121

    Author: Hugo U. R. Strand, 2026
    """

    def __init__(self, z=np.array([]), f=np.array([]), w=np.array([])):
        
        assert(len(z) == len(w))
        assert(len(z) == f.shape[0])
        # Only support scalar and matrix valued f (with known hermitian conjugation relation)
        assert(f.ndim == 1 or f.ndim == 3) 

        self.z = z
        self.f = f
        self.w = w


    def fast_eval(self, Z):
        """ Evaluate the conjugate paired rational approximation points Z.
        Assuming that Z does not contain any support points. """

        Zz    = Z[:, None] - self.z[None, :]
        Zzbar = Z[:, None] - self.z[None, :].conjugate()

        wC    = self.w[None, :] / Zz
        wCbar = self.w[None, :].conjugate() / Zzbar

        return self.__barycentric_eval(wC, wCbar)
    

    def __barycentric_eval(self, wC, wCbar):
        """ Internal helper function for barycentric formula evaluation.
        Shared by fast_eval and __call__."""

        n = np.einsum('j...,kj->k...', self.f, wC) + \
            np.einsum('j...,kj->k...', self.__fbar(), wCbar)
        d = np.sum(wC + wCbar, axis=1)
        r = np.einsum('k...,k->k...', n, 1/d)
        return r


    def __fbar(self):
        """ Return the conjugate paired values of f, using the known hermitian conjugation relation."""
        if self.f.ndim == 1: return self.f.conjugate()
        elif self.f.ndim == 3: return np.transpose(self.f, (0, 2, 1)).conjugate()
        else: raise NotImplementedError("Only scalar and matrix valued f are supported.")


    def __call__(self, Z):
        """ Evaluate the conjugate paired rational approximation at points Z.
        Also handle cases when Z is a support point, and
        then return the corresponding f value. """
        
        Zz    = Z[:, None] - self.z[None, :] # Cachy matrix, Eq. (3.7) in [1]
        Zzbar = Z[:, None] - self.z[None, :].conjugate()

        # Find evaluation points that are support points
        idxs = np.nonzero(Zz == 0.)
        idxs_bar = np.nonzero(Zzbar == 0.)

        # Set zeros to unity before inversion
        Zz[idxs] = 1.
        Zzbar[idxs_bar] = 1.

        wC = self.w[None, :] / Zz
        wCbar = self.w[None, :].conjugate() / Zzbar

        # For evaluation points in the support set
        # only return the corresponding value

        ridxs = idxs[0]
        wC[ridxs, :] = 0.0
        wCbar[ridxs, :] = 0.0
        wC[idxs] = 1.0

        ridxs_bar = idxs_bar[0]
        wC[ridxs_bar, :] = 0.0
        wCbar[ridxs_bar, :] = 0.0
        wCbar[idxs_bar] = 1.0

        return self.__barycentric_eval(wC, wCbar)


    def aaa_step(self, Z, F, R, tol_conj=1e-12):
        """ Perform one step of the AAA algorithm,
        with the additional constraint that support points are added in conjugate pairs. """
        
        # Find index of largest residual
        if R.ndim == 1: idx = np.argmax(np.abs(R))
        elif R.ndim == 3: idx = np.argmax(np.max(np.abs(R), axis=(1, 2)))
        else: raise NotImplementedError("Only scalar and matrix valued functions are supported.")

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
        R = F - self.fast_eval(Z) # Recompute residual
        self.residual = np.max(np.abs(R))
        return Z, F, R    


    def __fit_weights(self, Z, F, scalar=False):
        """ Fit the weights w of the conjugate paired rational approximation by an SVD.

        Note
        ----

        From the barycentric interpolation formula assuming conjugated support point pairs

        .. math::
            r(z) = \\frac{n(z)}{d(z)} =
            \\left[ \\sum_j \\frac{f_j w_j}{z - z_j} + \\frac{f^\\dagger_j \\bar{w}_j}{z - \\bar{z}_j} \\right]
            \\Bigg/
            \\left[ \\sum_j \\frac{w_j}{z - z_j} + \\frac{\bar{w}_j}{z - \\bar{z}_j} \\right]

        The residual vector :math:`R_k` of the weight minimization as

        .. math::
            R_k \\equiv f(Z_k) d(Z_k) - n(Z_k) =
            f(Z_k) \\sum_j \\left[ \\frac{w_j}{z - z_j} + \\frac{\\bar{w}_j}{z - \\bar{z}_j} \\right]
            -
            \\sum_j \\left[ \\frac{f_j w_j}{z - z_j} + \\frac{f^\\dagger_j \\bar{w}_j}{z - \\bar{z}_j} \\right]
            =
            \\sum_j \\left[ w_j C_{jk} f(Z_k) - \\bar{w}_j \\bar{C}_{jk} f(Z_k)
            - w_j f_j C_{jk} - \\bar{w}_j f^\\dagger_j \\bar{C}_{jk} \\right]
            =
            \\mathbf{x} \\left( K C S_k + \\bar{K} \\bar{C} S_k
            - K S_j C - \\bar{K} S_j^{(\\dagger)} \\bar{C} \\right)
            = \\mathbf{x} A

        where we have introduced the diagonal matrices :math:`S_k = \\textrm{diag}[f(Z_k)]`,
        :math:`S_j = \\textrm{diag}[f_j]`, and :math:`S^{(\\dagger)}_j = \\textrm{diag}[f^\\dagger_j]`.
        The matrices :math:`K = [ \\mathbf{1} | i\\mathbf{1} ]^T` and
        :math:`\bar{K} = [ \\mathbf{1} | -i\\mathbf{1} ]^T`  are transforms from the real-valued vector
        :math:`\\mathbf{x}` to the complex weights :math:`w`,
        i.e. $w_i = \\mathbf{x} K$ and $\\bar{w}_i = \\mathbf{x} \\bar{K}$.

        Finally the matrices :math:`C` and :math:`\\bar{C}` are the two Cauchy matrices

        .. math::
            C_{jk} \\equiv \\frac{1}{Z_k - z_j} \\, , \\quad \\bar{C}_{jk} \\equiv \\frac{1}{Z_k - \\bar{z}_j} \\, .

        The optimal weights :math:`w` are obtained as the left singular vector :math:`\\mathbf{u}` of
        :math:`A` with the smallest singular value, :math:`w = \\mathbf{u} K`.

        """

        C    = 1.0 / (Z[None, :] - self.z[:, None]) # Cachy matrix, Eq. (3.7) in [1]
        Cbar = 1.0 / (Z[None, :] - self.z[:, None].conjugate())

        # Transform matrices K, Kbar 
        # from vector with separated real and imaginary parts x = [w.real, w.imag] 
        # to complex valued vector w = x @ K, w.conjugate = x @ Kbar
        I = np.eye(len(self.z))
        K    = np.vstack((I, +1j*I))
        Kbar = np.vstack((I, -1j*I))

        # Construct fitting matrix A, such that min_x || A @ x ||_2 gives the optimal weights w = x @ K
        A = np.einsum('xk,k...->xk...', (K @ C + Kbar @ Cbar), F) + \
            - np.einsum('xj,j...,jk->xk...', K, self.f, C) \
            - np.einsum('xj,j...,jk->xk...', Kbar, self.__fbar(), Cbar) \

        A = A.reshape((A.shape[0], -1))
        A = np.hstack((A.real, A.imag))

        # Solve minimization problem using left singular vector with smallest singular value
        U, S, Vh = np.linalg.svd(A, full_matrices=False)
        assert( U.shape[1] > 0 )
        x = U[:, -1]
        w = x @ K

        return w


    def poles_and_residues(self, residue_dz=1e-5, scale_and_balance=False):
        """ Determine the poles and residues of the rational approximation,
        by solving a generalized eigenvalue problem with an arrowhead matrix, 
        and using the Laurent series residue relation. """

        #wwbar = np.concatenate((self.w, self.w.conjugate()))
        #zzbar = np.concatenate((self.z, self.z.conjugate()))

        wwbar = self.__interleave_first_axis(self.w, self.w.conjugate())
        zzbar = self.__interleave_first_axis(self.z, self.z.conjugate())

        n = len(zzbar)

        # Left hand (A) and right hand (B) matrices in Eq. (3.11) in [1]
        # of the generalized eigenvalue problem, A@x = \lambda B@x

        A = np.block([
            [ np.zeros((1,1)), wwbar[None, :] ],
            [ np.ones((n, 1)), np.diag(zzbar) ]] )

        B = np.diag(np.concatenate(([0.0], np.ones(n))))

        if scale_and_balance:
            print(f'A =\n{A}\nB =\n{B}')

            s = np.concatenate(([1.], np.sqrt(wwbar)))
            print(f's = {s}')

            SAS = (1./s)[:, None] * A * s[None, :]
            print(f'SAS =\n{SAS}')


            r_norm_sas = np.linalg.norm(SAS[0, :])
            c_norm_sas = np.linalg.norm(SAS[:, 0])

            sl = np.concatenate(([r_norm_sas], np.ones(n)))
            sr = np.concatenate(([c_norm_sas], np.ones(n)))

            SSASS = (1./sl)[:, None] * SAS * (1./sr)[None, :]

            r_norm_ssass = np.linalg.norm(SSASS[0, :])
            c_norm_ssass = np.linalg.norm(SSASS[:, 0])

            print(f'SAS r_norm_sas = {r_norm_sas:2.2E}, c_norm_sas = {c_norm_sas:2.2E}')
            print(f'SAS r_norm_ssass = {r_norm_ssass:2.2E}, c_norm_ssass = {c_norm_ssass:2.2E}')

            r_norm = np.linalg.norm(A[0, :])
            c_norm = np.linalg.norm(A[:, 0])
            print(f'A   r_norm = {r_norm:2.2E}, c_norm = {c_norm:2.2E}')

            #poles_sas = scipy_eigvals(SAS, B, overwrite_a=True)
            poles_ssass = scipy_eigvals(SSASS, B, overwrite_a=True)

        # Obtain the poles as generalized eigenvalues of the pencil (A, B)
        # NB! two eigenvalues are infinite, and needs to be discarded.
        poles = scipy_eigvals(A, B, overwrite_a=True)

        if scale_and_balance:
            #print(f'Poles from SAS   = {poles_sas}')
            print(f'Poles from SSASS = {poles_ssass}')
            print(f'Poles from A     = {poles}')
            poles = poles_ssass

        poles = poles[np.isfinite(poles)]

        # Sort poles by real part, to enable simpler testing...
        sidx = np.argsort(poles.real)
        poles = poles[sidx]

        if False:
            """ Test the accuracy of the poles by evaluation of the 
            denominator of the rational function at the poles, which should be zero."""

            def eval_denominator(Z):
                Zz    = Z[:, None] - self.z[None, :]
                Zzbar = Z[:, None] - self.z[None, :].conjugate()

                wC    = self.w[None, :] / Zz
                wCbar = self.w[None, :].conjugate() / Zzbar

                d = np.sum(wC + wCbar, axis=1)
                return d
            
            zeros = eval_denominator(poles)
            print(f'zeros = {zeros}')
            #exit()

        # Calculate residues by evaluating the function at points close to the poles
        # See the MATLAB code in Fig. 4.1 of Ref [1], 
        # and comment in BarycentricRationalApproximation.poles_and_residues().

        dz = residue_dz * np.exp(2j*np.pi*np.arange(1, 5)/4)
        Z = poles[:, None] + dz[None, :]

        shape = [len(poles), 4] + list(self.f.shape[1:])
        residues = np.einsum(
            'pf...,f->p...', self.fast_eval(Z.flatten()).reshape(shape), dz / 4)

        return poles, residues


    def remove_froissart_doublets(self, Z, F, tol=None, imag_tol=1e-4, verbose=True, prefix=''):
        """ Remove Froissart doublets, i.e. poles with small residues, 
        by putting the closest support point of each pole back to the fitting set.
         
        Optionally, locate poles with imaginary part > imag_tol, 
        and also remove their adjacent support points. """

        if tol is None:
            tol = 1e-13

        poles, residues = self.poles_and_residues()

        # Find indices of small residues
        if residues.ndim == 1: ridxs = np.nonzero(np.abs(residues) < tol)
        elif residues.ndim == 3: ridxs = np.nonzero(np.max(np.abs(residues), axis=(1, 2)) < tol)
        else: raise NotImplementedError("Only scalar and matrix valued functions are supported.")

        # Optionally locate poles with non-negligible imaginary part.
        # In infinite arithmetic, the constrained AAA algorithm only should produce real poles.
        if imag_tol is not None:
            ridxs_im = np.nonzero(np.abs(poles.imag) > imag_tol)
            ridxs = (np.unique(np.concatenate((ridxs[0], ridxs_im[0]))),)

        if len(ridxs[0]) <= 1:
            # Since the conjugated support points produce pole pairs, 
            # do not remove a single pole with small residue.
            # This is likely not a Froissart doublet.
            
            #print(f'AAA: No Froissart doublets found with residues smaller than {tol:2.2E}')
            return 0, Z, F

        if verbose:
            print(f'{prefix}AAA: Found {len(ridxs[0])} residues < {tol}.')

        # Locate the closest support point to each pole with small residue
        zz = np.concatenate((self.z, self.z.conjugate()))
        dists = np.abs(zz[:, None] - poles[ridxs][None, :])
        pidxs = np.argmin(dists, axis=0)
        pidxs = np.unique(np.mod(pidxs, len(self.z)))

        if verbose:
            print(f'{prefix}AAA: Found {len(pidxs)} adjacent support points to remove.')

        assert( len(pidxs) > 0 )

        # Put support points back to the fitting set
        Z = np.concatenate((Z, self.z[pidxs]))
        F = np.concatenate((F, self.f[pidxs]))

        # Remove points in support set
        self.z = np.delete(self.z, pidxs)
        self.f = np.delete(self.f, pidxs, axis=0)

        self.w = self.__fit_weights(Z, F)
        R = F - self.fast_eval(Z) # Recompute residual
        self.residual = np.max(np.abs(R))

        if verbose:
            print(f'{prefix}AAA: After removing {len(pidxs)} support points the residual is {self.residual:2.2E}')

        return len(pidxs), Z, F


    def __interleave_first_axis(self, A, B):
        """ Interleave the first axis of two arrays A and B, i.e. [A0, B0, A1, B1, ...]. """
        assert(A.shape == B.shape)
        shape = list(A.shape)
        shape[0] *= 2
        C = np.empty(shape, dtype=A.dtype)
        C[0::2] = A
        C[1::2] = B
        return C


    def barycentric_rational_interpolant(self):
        """ Return the conjugated barycentric rational interpolant 
        as a standard BarycentricRationalApproximation object, 
        with support points and values interleaved in conjugate pairs."""

        zzbar = self.__interleave_first_axis(self.z, self.z.conjugate())
        ffbar = self.__interleave_first_axis(self.f, self.__fbar())
        wwbar = self.__interleave_first_axis(self.w, self.w.conjugate())

        if zzbar[0].imag < 0:
            zzbar = zzbar.conjugate()
            ffbar = ffbar.conjugate()
            wwbar = wwbar.conjugate()

        return BarycentricRationalApproximation(zzbar, ffbar, wwbar)
        

    def get_sop(self, real_poles=True):
        """ Return the rational approximation as a sum of simple poles. """

        poles, residues = self.poles_and_residues()

        if real_poles:
            poles = poles.real

        sop = SumOfSimplePoles(poles=poles, residues=residues)
        return sop

