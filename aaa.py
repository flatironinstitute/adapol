"""
This code is modified based on https://github.com/c-f-h/aaa/tree/master

Two main modifications:
1. Add function aaa_real for analytic continuation, which gives us real poles

For more information, see the paper

  The AAA Algorithm for Rational Approximation
  Yuji Nakatsukasa, Olivier Sete, and Lloyd N. Trefethen
  SIAM Journal on Scientific Computing 2018 40:3, A1494-A1522

as well as the Chebfun package <http://www.chebfun.org>. This code is an almost
direct port of the Chebfun implementation of aaa to Python.
"""

import numpy as np
import scipy.linalg

class BarycentricRational:
    """A class representing a rational function in barycentric representation.
    """
    def __init__(self, z, f, w):
        """Barycentric representation of rational function with nodes z, values f and weights w.

        The rational function has the interpolation property r(z_j) = f_j.
        """
        self.nodes = z
        self.values = f
        self.weights = w

    def __call__(self, x):
        """Evaluate rational function at all points of `x`"""
        zj,fj,wj = self.nodes, self.values, self.weights
        xv = np.asanyarray(x).ravel()
        if len(fj.shape)==1:
            
            # ignore inf/nan for now
            with np.errstate(divide='ignore', invalid='ignore'):
                C = 1.0 / (xv[:,None] - zj[None,:])
                r = C.dot(wj*fj) / C.dot(wj)

            # for z in zj, the above produces NaN; we check for this
            nans = np.nonzero(np.isnan(r))[0]
            for i in nans:
                # is xv[i] one of our nodes?
                nodeidx = np.nonzero(xv[i] == zj)[0]
                if len(nodeidx) > 0:
                    # then replace the NaN with the value at that node
                    r[i] = fj[nodeidx[0]]

            if np.isscalar(x):
                return r[0]
            else:
                r.shape = x.shape
                return r
        else:
            Norb = fj.shape[1]
            if np.isscalar(xv):
                rr = np.zeros((Norb,Norb),dtype=fj.dtype)
            else:
                rr = np.zeros((len(xv), Norb,Norb),dtype=fj.dtype) 
            for i in range(Norb):
                for j in range(Norb):
                    rij = BarycentricRational(zj, fj[:,i,j], wj)
                    if np.isscalar(xv): 
                        rr[i,j] = rij(xv)
                    else:
                        rr[:,i,j] = rij(xv)
            return rr


    def pol(self):
        """Return the poles and residues of the rational function."""
        zj,fj,wj = self.nodes, self.values, self.weights
        m = len(wj)

        # compute poles
        B = np.eye(m+1)
        B[0,0] = 0
        E = np.block([[0, wj],
                      [np.ones((m,1)), np.diag(zj)]])
        evals = scipy.linalg.eigvals(E, B)
        pol = np.real_if_close(evals[np.isfinite(evals)])

        return pol

    def zeros(self):
        """Return the zeros of the rational function."""
        zj,fj,wj = self.nodes, self.values, self.weights
        m = len(wj)

        B = np.eye(m+1)
        B[0,0] = 0
        E = np.block([[0, wj],
                      [fj[:,None], np.diag(zj)]])
        evals = scipy.linalg.eigvals(E, B)
        return np.real_if_close(evals[np.isfinite(evals)])


################################################################################

def aaa_real(F, Z, tol=1e-13, mmax=100, return_errors=False):
    """Compute a rational approximation of `F` over the points `Z`.

    The nodes `Z` should be given as an array.

    The nodes `F` should be given as an array.

    Returns a `BarycentricRational` instance which can be called to evaluate
    the rational function, and can be queried for the poles, residues, and
    zeros of the function.
    """
    Z = np.asanyarray(Z).ravel()

    F = np.asanyarray(F).ravel()

    #only use input z that are on iR_+. Will map them to iR_- by taking conjugate of function value.
    half_index = np.imag(Z)>0
    Z_half = Z[half_index]
    F_half = F[half_index]
    M_half = len(Z_half)

    Z = np.append(Z_half, np.conjugate(Z_half))
    F = np.append(F_half, np.conjugate(F_half))


    M = M_half*2
    J = list(range(M))
    zj = np.empty(0, dtype=Z.dtype)
    fj = np.empty(0, dtype=F.dtype)
    C = np.empty([M,0], dtype=F.dtype)
    errors = []

    reltol = tol * np.linalg.norm(F, np.inf)

    R = np.mean(F) * np.ones_like(F)

    mlist = range(2, mmax+1,2)


    for m in mlist:
        # find largest residual
        jj = np.argmax(abs(F - R))
        zj = np.append(zj, (Z[jj],))
        fj = np.append(fj, (F[jj],))
        J.remove(jj)

        # Cauchy matrix containing the basis functions as columns

        jj2 = (jj + M_half) % M
        zj = np.append(zj, (Z[jj2],))
        fj = np.append(fj, (F[jj2],))

        J.remove(jj2)

        C = 1.0 / (Z[J,None] - zj[None,:]) 
        

        # Loewner matrix
        Apart = (F[J,None] - fj[None,:]) * C
        
    
        Awidth = np.size(Apart, 1)
        Apart_l = Apart[:, range(0,Awidth,2)]
        Apart_r = Apart[:, range(1, Awidth,2)]
        Anew = np.concatenate((Apart_l+Apart_r, (Apart_l-Apart_r)*1j),axis = 1)
        Anew = np.concatenate((np.real(Anew), np.imag(Anew)),axis =0)

        # compute weights as right singular vector for smallest singular value
        # AM = np.conjugate(np.transpose(Anew))@Anew
        # _,  Vh = np.linalg.eig(AM)
        _, _,  Vh = np.linalg.svd(Anew,full_matrices=False)

        wj_r = Vh[m-1,:]

        wj_r = np.reshape(wj_r,(2,int(m/2)))
        wj_c = np.zeros((2,int(m/2)), dtype = np.complex128)
        wj_c[0,:] = wj_r[0,:]+1j*wj_r[1,:]
        wj_c[1,:] = wj_r[0,:]-1j*wj_r[1,:]

        wj = np.asanyarray(wj_c.T).ravel()

        # approximation: numerator / denominator
        N = C.dot(wj * fj)
        D = C.dot(wj)

        # update residual
        R = F.copy()
        R[J] = N / D

        # check for convergence
        maxerr = np.linalg.norm(F - R, np.inf)
        errors.append(maxerr)
        if errors[-1] <= reltol:
            break
    maxerrAAA = maxerr
    # if  M == 2:
    #     zj = Z
    #     fj = F
    #     wj = np.array([1  -1])       # Only pole at infinity.
    #     wj = wj/np.linalg.norm(wj)   # Impose norm(w) = 1 for consistency.
    #     maxerrAAA = 0
    

    wj0, fj0 = wj, fj #Save params in case Lawson fails
    wt = np.empty((M))
    wt.fill(np.NaN)
    wt_new = np.ones(M)

    maxerrold = maxerrAAA
    maxeerr = maxerrold
    nj = len(zj)
    A = np.empty([M,0],dtype=np.complex128)
    with np.errstate(divide='ignore', invalid='ignore'):
        for j in range(nj): # Cauchy/Loewner matrix
            A = np.hstack([A, 1/(Z[:,None]-zj[j])])
            A = np.hstack([A, F[:,None]/(Z[:,None]-zj[j])])
    for j in range(nj):
        i = np.argwhere(Z == zj[j]) #support pt rows are special
        A[i,:] = 0.0
        A[i, 2*j] = 1.0
        A[i, 2*j+1] = F[i]
    stepno = 0
    
    while (stepno<20) or (maxerr/maxerrold < 0.999 and stepno < 1000):
        stepno = stepno + 1
        wt = wt_new
        W = np.diag(wt)
       
        WA = W@A
        WAM = np.conjugate(np.transpose(WA))@WA
        _,  V = np.linalg.eig(WAM)
        c = V[:,-1]
        denom = np.zeros(M, dtype=np.complex128)
        num = np.zeros(M, dtype = np.complex128)
        with np.errstate(divide='ignore', invalid='ignore'):
            for j in range(nj):
                denom = denom + c[2*j+1]/(Z-zj[j])
                num= num - c[2*j]/(Z-zj[j])
            R = num/denom
        
        for j in range(nj):
            i = np.argwhere(Z == zj[j]) #support pt rows are special
            R[i] = -c[2*j]/c[2*j+1]
        
        err = F - R
        abserr = np.abs(err)
        wt_new = wt*abserr
        wt_new = wt_new/np.linalg.norm(wt_new, np.inf)
        maxerrold = maxerr
        maxerr = np.max(abserr)
    wj = c[range(1,2*nj,2)]
    fj = -c[range(0,2*nj,2)]/wj
    if maxerr > maxerrAAA:
        wj = wj0
        fj = fj0

    #Remove support points with zero weight:
    I = np.argwhere(wj==0)
    zj = np.delete(zj,I)
    wj = np.delete(wj,I)
    fj = np.delete(fj,I)
    

    r = BarycentricRational(zj, fj, wj)
    return (r, errors) if return_errors else r


################################################################################

def aaa_matrix_real(F, Z, tol=1e-13, mmax=100, return_errors=False):
    """Compute a rational approximation of `F` over the points `Z`.

    The nodes `Z` should be given as an array.

    `F` can be given as a function or as an array of function values over `Z`.

    Returns a `BarycentricRational` instance which can be called to evaluate
    the rational function, and can be queried for the poles, residues, and
    zeros of the function.
    """
    Z = np.asanyarray(Z).ravel()
    

    #only use input z that are on iR_+. Will map them to iR_- by taking conjugate of function value.
    half_index = np.imag(Z)>0
    Z_half = Z[half_index]
    F_half = F[half_index,:,:]
    M_half = len(Z_half)

    Z = np.append(Z_half, np.conjugate(Z_half))

    Norb = F.shape[1]
    F_other_half = np.zeros_like(F_half)
    for i in range(M_half):
        F_other_half[i,:,:] = np.conjugate(np.transpose(F_half[i,:,:]))
    F = np.concatenate((F_half, F_other_half),axis=0)

    M = M_half*2 

    F_mat = np.reshape(F, (M, Norb*Norb))    
    
    J = list(range(M))
    zj = np.empty(0, dtype=Z.dtype)
    fj = np.empty((0,Norb*Norb), dtype=F.dtype)
    C = np.empty([M,0], dtype=F.dtype)
    errors = []

    reltol = tol * np.linalg.norm(F_mat, np.inf)

    R = np.mean(F_mat) * np.ones_like(F_mat)

    mlist = range(2, mmax+1,2)


    for m in mlist:
        # find largest residual
        jj = np.argmax(np.sum(abs(F_mat - R)**2,1))
        zj = np.append(zj, (Z[jj],))
        fj = np.concatenate((fj, F_mat[jj:jj+1,:]),axis=0)
        J.remove(jj)

        # Cauchy matrix containing the basis functions as columns

        jj2 = (jj + M_half) % M
        zj = np.append(zj, (Z[jj2],))
        fj = np.concatenate((fj, F_mat[jj2:jj2+1,:]),axis=0)

        J.remove(jj2)

        C = 1.0 / (Z[J,None] - zj[None,:])        

        # Loewner matrix
        Apart = np.zeros(((M-m)*Norb*Norb, m), dtype=F.dtype)
        for i in range(Norb*Norb):
            Fhere = F_mat[:,i]
            fjhere = fj[:,i]

            Apart[range(0 + i, i + (M-m)*Norb*Norb, Norb*Norb),:] = (Fhere[J,None] - fjhere[None,:]) * C 
        
        Awidth = np.size(Apart, 1)
        Apart_l = Apart[:, range(0,Awidth,2)]
        Apart_r = Apart[:, range(1, Awidth,2)]
        Anew = np.concatenate((Apart_l+Apart_r, (Apart_l-Apart_r)*1j),axis = 1)
        Anew = np.concatenate((np.real(Anew), np.imag(Anew)),axis =0)

        

        # compute weights as right singular vector for smallest singular value
        _, _,  Vh = np.linalg.svd(Anew,full_matrices=False)

        wj_r = Vh[-1,:]

        wj_r = np.reshape(wj_r,(2,int(m/2)))
        wj_c = np.zeros((2,int(m/2)), dtype = np.complex128)
        wj_c[0,:] = wj_r[0,:]+1j*wj_r[1,:]
        wj_c[1,:] = wj_r[0,:]-1j*wj_r[1,:]

        wj = np.asanyarray(wj_c.T).ravel()
       

        # approximation: numerator / denominator
        
        D = C.dot(wj)

        # update residual
        R = F_mat.copy()
        

        for i in range(Norb*Norb):
            fjhere = fj[:,i]
            N = C.dot(wj * fjhere) #needs to change N and R
            R[J,i] = N / D
        
        # check for convergence
        errors.append(np.linalg.norm(F_mat - R, np.inf))
        if errors[-1] <= reltol:
            break
    
    fj = fj.reshape(m, Norb, Norb)
    r = BarycentricRational(zj, fj, wj)
    return (r, errors) if return_errors else r

def interpolate_poly(values, nodes):
    """Compute the interpolating polynomial for the given nodes and values in
    barycentric form.
    """
    n = len(nodes)
    if n != len(values):
        raise ValueError('input arrays should have the same length')
    x = nodes
    weights = np.array([
            1.0 / np.prod([x[i] - x[j] for j in range(n) if j != i])
            for i in range(n)
    ])
    return BarycentricRational(nodes, values, weights)

def interpolate_with_poles(values, nodes, poles):
    """Compute a rational function which interpolates the given values at the
    given nodes and which has the given poles.
    """
    n = len(nodes)
    if n != len(values) or n != len(poles) + 1:
        raise ValueError('invalid length of arrays')
    nodes = np.asanyarray(nodes)
    values = np.asanyarray(values)
    poles = np.asanyarray(poles)
    # compute Cauchy matrix
    C = 1.0 / (poles[:,None] - nodes[None,:])
    # compute null space
    _, _, Vh = np.linalg.svd(C)
    weights = Vh[-1, :]
    return BarycentricRational(nodes, values, weights)

def floater_hormann(values, nodes, blending):
    """Compute the Floater-Hormann rational interpolant for the given nodes and
    values. See (Floater, Hormann 2007), DOI 10.1007/s00211-007-0093-y.

    The blending parameter (usually called `d` in the literature) is an integer
    between 0 and n (inclusive), where n+1 is the number of interpolation
    nodes. For functions with higher smoothness, the blending parameter may be
    chosen higher. For d=n, the result is the polynomial interpolant.

    Returns an instance of `BarycentricRational`.
    """
    n = len(values) - 1
    if n != len(nodes) - 1:
        raise ValueError('input arrays should have the same length')
    if not (0 <= blending <= n):
        raise ValueError('blending parameter should be between 0 and n')

    weights = np.zeros(n + 1)
    # abbreviations to match the formulas in the literature
    d = blending
    x = nodes
    for i in range(n + 1):
        Ji = range(max(0, i-d), min(i, n-d) + 1)
        weight = 0.0
        for k in Ji:
            weight += np.prod([1.0 / abs(x[i] - x[j])
                    for j in range(k, k+d+1)
                    if j != i])
        weights[i] = (-1.0)**(i-d) * weight
    return BarycentricRational(nodes, values, weights)
