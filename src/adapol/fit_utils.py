# This is a python implementation for analytic continuation of Fermionic Green's functions/self energy
# using PES (ES) method
# Reference: PhysRevB.107.075151
import numpy as np
import scipy
import scipy.optimize
import cvxpy as cp
from .aaa import aaa_matrix_real


# import mosek
def eval_with_pole(pol, Z, weight):
    pol_t = np.reshape(pol, [pol.size, 1])
    M = 1 / (Z - pol_t)
    M = M.transpose()
    if len(weight.shape) == 1:
        return M @ weight
    else:
        G = M @ np.reshape(weight, (weight.shape[0], weight.shape[1] * weight.shape[2]))
        return np.reshape(G, (G.shape[0], weight.shape[1], weight.shape[2]))


def get_weight(
    pol, Z, G, cleanflag=True, maxiter=1000, complex=True, fast=False, eps=1e-8
):
    pol_t = np.reshape(pol, [pol.size, 1])
    M = 1 / (Z - pol_t)
    M = M.transpose()
    MM = np.concatenate([M.real, M.imag])
    if len(G.shape) == 1:
        GG = np.concatenate([G.real, G.imag])
        if cleanflag:
            R = np.linalg.lstsq(MM, GG, rcond=0)[0]
        else:
            [R, rnorm] = scipy.optimize.nnls(MM, GG, maxiter=maxiter)
        residue = G - M @ R
    else:
        Np = len(pol)
        Norb = G.shape[1]
        R = np.zeros((Np, Norb, Norb), dtype=np.complex128)
        if cleanflag:
            for i in range(Norb):
                GG = np.concatenate([G[:, i, i].real, G[:, i, i].imag])
                R[:, i, i] = np.linalg.lstsq(MM, GG, rcond=0)[0]
                for j in range(i + 1, Norb):
                    g1 = (G[:, j, i] + G[:, i, j]) / 2.0
                    g2 = (G[:, i, j] - G[:, j, i]) / 2.0
                    GG1 = np.concatenate([g1.real, g1.imag])
                    GG2 = np.concatenate([g2.imag, -g2.real])
                    R1 = np.linalg.lstsq(MM, GG1, rcond=0)[0]
                    R2 = np.linalg.lstsq(MM, GG2, rcond=0)[0]
                    R[:, i, j] = R1 + 1j * R2
                    R[:, j, i] = R1 - 1j * R2
        else:
            if not fast:
                Nw = len(Z)

                if complex:
                    X = [cp.Variable((Norb, Norb), hermitian=True) for i in range(Np)]
                    constraints = [X[i] >> 0 for i in range(Np)]
                else:
                    X = [cp.Variable((Norb, Norb), PSD=True) for i in range(Np)]

            
                Gfit = []
                for w in range(Nw):
                    Gfit.append(cp.sum_squares(sum([ M[w,i]*X[i] for i in range(Np)]) - G[w,:,:]))

                if complex:
                    prob = cp.Problem(cp.Minimize(sum(Gfit)), constraints)
                else:
                    prob = cp.Problem(cp.Minimize(sum(Gfit)))
                prob.solve(solver="SCS", verbose=False, eps=eps)
                #  MOSEK parameters
                # mosek_params_dict = {"MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1.e-8,\
                #                     "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1.e-8,
                #                     "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1.e-8,
                #                     "MSK_DPAR_INTPNT_CO_TOL_NEAR_REL": 1000}
                # result = prob.solve(solver = "MOSEK", verbose=False,\
                #                 mosek_params = mosek_params_dict)

                for i in range(Np):
                    R[i] = X[i].value
            else:
                for i in range(Norb):
                    GG = np.concatenate([G[:, i, i].real, G[:, i, i].imag])
                    Rii = scipy.optimize.nnls(MM, GG, maxiter=maxiter)[0]
                    R[:, i, i] = Rii
                MM2 = np.concatenate([M, np.conj(M)])
                for i in range(Norb):
                    for j in range(i + 1, Norb):
                        GG = np.concatenate([G[:, i, j], np.conj(G[:, j, i])])
                        bound = np.sqrt(np.abs(R[:, i, i] * R[:, j, j]))
                        x = cp.Variable(MM.shape[1], complex=True)
                        constraints = [
                            cp.abs(x[k]) <= bound[k] for k in range(MM.shape[1])
                        ]
                        objective = cp.Minimize(cp.sum_squares(MM2 @ x - GG))
                        prob = cp.Problem(objective, constraints)
                        prob.solve(solver="SCS", verbose=False, eps=eps)
                        # result = prob.solve(solver = cp.MOSEK,verbose = False,mosek_params = mosek_params_dict)
                        R[:, i, j] = x.value
                        R[:, j, i] = np.conj(x.value)

        residue = 1.0 * G
        for i in range(Np):
            residue = residue - M[:, i, None, None] * R[i]
    return R, M, residue


def aaa_reduce(pol, R, eps=1e-6):
    Np = R.shape[0]
    Rnorm = np.zeros(Np)
    for i in range(Np):
        Rnorm[i] = np.linalg.norm(R[i])
    nonz_index = Rnorm > eps
    return pol[nonz_index], R[nonz_index]


def erroreval(pol, Z, G, cleanflag=True, maxiter=1000, fast=False, complex=True):
    R, M, residue = get_weight(
        pol, Z, G, cleanflag=cleanflag, maxiter=maxiter, complex=complex, fast=fast
    )
    if len(G.shape) == 1:
        y = np.linalg.norm(residue)
        grad = np.real(np.dot(np.conj(residue), (R * (M**2))))
    else:
        y = np.linalg.norm(residue.flatten())

        Np = len(pol)
        grad = np.zeros(Np)
        Nw = len(Z)
        for k in range(Np):
            for w in range(Nw):
                grad[k] = grad[k] + np.real(
                    np.sum((M[w, k] ** 2) * (np.conj(residue[w, :, :]) * R[k]))
                )

    grad = -grad / y
    return y, grad


def pole_fitting(
    Delta,
    Z,
    tol=None,
    Ns=None,
    mmin=None,
    mmax=50,
    maxiter=50,
    solver="lstsq",
    fast=False,
    disp=False,
    complex=True,
):
    # set cleanflag
    if solver == "lstsq":
        cleanflag = True
    if solver == "sdp":
        cleanflag = False

    # pole estimation
    # tol needs to be fixed
    if Ns is None and tol is None:
        raise Exception(
            "One needs to specify either the number of poles or the fitting error tolerance."
        )
    if Ns is not None and tol is not None:
        raise Exception(
            "One can not specify both the number of poles and the fitting error tolerance. Only specify one of them."
        )
    if Ns is None:
        if mmin is None or mmin < 4:
            mmin = 4
        if mmin % 2 == 1:
            mmin = mmin + 1
        if mmax > 2 * (Z.shape[0] // 2):
            mmax = 2 * (Z.shape[0] // 2)
    else:
        if Ns % 2 == 1:
            Ns = Ns + 1
        mmin, mmax = Ns, Ns
    if len(Delta.shape) == 1:
        Delta = Delta.reshape(Delta.shape[0], 1, 1)

    for m in range(mmin, mmax + 1, 2):
        pol, _, _, _ = aaa_matrix_real(Delta, 1j * Z, mmax=m)
        pol = np.real(pol)
        weight, _, residue = get_weight(
            pol, 1.0j * Z, Delta, cleanflag=cleanflag, complex=complex, fast=fast
        )
        # print(np.max(np.abs(residue)))
        if tol is not None:
            if np.max(np.abs(residue)) > tol * 10:
                continue
        if Ns is None:
            pol, weight = aaa_reduce(pol, weight, 1e-5)
        # print("Number of poles is ", len(pol))
        if cleanflag:
            if maxiter > 0:

                def fhere(pole):
                    return erroreval(
                        pole, 1j * Z, Delta, cleanflag=cleanflag, complex=complex
                    )

                # fhere = lambda pole: erroreval(pole,1j*Z,Delta,cleanflag=cleanflag,complex=complex)
                res = scipy.optimize.minimize(
                    fhere,
                    pol,
                    method="L-BFGS-B",
                    jac=True,
                    options={
                        "disp": disp,
                        "maxiter": maxiter,
                        "gtol": 1e-10,
                        "ftol": 1e-10,
                    },
                )
        else:

            def fhere1(pole):
                return erroreval(pole, 1j * Z, Delta, cleanflag=True, complex=complex)

            # fhere = lambda pole: erroreval(pole,1j*Z,Delta,cleanflag=True,complex=complex)
            res = scipy.optimize.minimize(
                fhere1,
                pol,
                method="L-BFGS-B",
                jac=True,
                options={"disp": False, "gtol": 1e-10, "ftol": 1e-10},
            )
            if maxiter > 0:

                def fhere2(pole):
                    return erroreval(
                        pole, 1j * Z, Delta, cleanflag=False, complex=complex, fast=fast
                    )

                # fhere = lambda pole: erroreval(pole,1j*Z,Delta,cleanflag=False,complex= complex,fast = fast)
                res = scipy.optimize.minimize(
                    fhere2,
                    res.x,
                    method="L-BFGS-B",
                    jac=True,
                    options={
                        "disp": disp,
                        "maxiter": maxiter,
                        "gtol": 1e-10,
                        "ftol": 1e-10,
                    },
                )

        weight, _, residuenew = get_weight(
            res.x, 1j * Z, Delta, cleanflag=cleanflag, fast=fast, complex=complex
        )
        if not check_psd(weight):
            weight, _, residuenew = get_weight(
                res.x, 1j * Z, Delta, cleanflag=False, complex=complex
            )
        err = np.max(np.abs(residuenew))
        if tol is not None:
            if err < tol:
                return res.x, weight, err
        else:
            return res.x, weight, err

    if tol is not None:
        print("Fail to reach desired fitting error!")
    return res.x, weight, np.max(np.abs(residuenew))


def check_psd(weight, atol=1e-6):
    check_psd = True
    for i in range(weight.shape[0]):
        val, _ = np.linalg.eig(weight[i])
        check_psd = check_psd and np.min(val.real) > -atol
    return check_psd

# The following function is used to support applications in ppsc-soe. The main difference is here we evaluate the fitting error on the time domain.
# todo: merge this function with the above pole_fitting function.
def kernel(tau, omega):
    kernel = np.empty((len(tau), len(omega)))

    p, = np.where(omega > 0.)
    m, = np.where(omega <= 0.)
    w_p, w_m = omega[p].T, omega[m].T

    tau = tau[:, None]

    kernel[:, p] = np.exp(-tau*w_p) / (1 + np.exp(-w_p))
    kernel[:, m] = np.exp((1. - tau)*w_m) / (1 + np.exp(w_m))

    return kernel
def get_weight_t(pol, tgrid, Deltat, beta):
    M = -kernel(tgrid/beta, pol*beta)
    shape_iaa = Deltat.shape
    shape_iA = (shape_iaa[0], shape_iaa[1]*shape_iaa[2])
    shape_xaa = (len(pol), shape_iaa[1], shape_iaa[2])
    weight = np.linalg.lstsq(M, Deltat.reshape(shape_iA), rcond=None)[0]
    residue = (Deltat.reshape(shape_iA) - M@weight).reshape(shape_iaa)
    
    weight = weight.reshape(shape_xaa)
    return weight, M, residue
def erroreval_t(pol,  tgrid, Deltat, beta):
    R, M, residue = get_weight_t(pol, tgrid, Deltat, beta)
    if len(Deltat.shape)==1:
        y = np.linalg.norm(residue)
        grad = np.real(np.dot(np.conj(residue) ,(R*(M**2))))
    else:
        y = np.linalg.norm(residue.flatten())

        Np = len(pol)
        grad = np.zeros(Np)
        Nw = len(tgrid)
        for k in range(Np):
            for w in range(Nw):
                grad[k] = grad[k] + np.real(np.sum((M[w,k]**2)*(np.conj(residue[w,:,:]) * R[k])))

    grad = -grad/y
    return y, grad
def polefitting(Deltaiw, Z, Deltat,tgrid, Deltat_dense, tgrid_dense,beta, Np_max=50,eps = 1e-5,Hermitian=True):
    Num_of_nonzero_entries = 0
    for i in range(Deltaiw.shape[1]):
        for j in range(Deltaiw.shape[2]):
            if np.max(np.abs((Deltat[:,i,j])))>1e-12:
                Num_of_nonzero_entries += 1
    
    for mmax in range(4,Np_max,2):
        pol, _, _, _ = aaa_matrix_real(Deltaiw, Z, mmax=mmax)
        pol = np.real(pol)
        weight, _, residue = get_weight_t(pol, tgrid, Deltat,beta)
        pol, weight = aaa_reduce(pol, weight,eps)
        fhere = lambda pole: erroreval_t(pol, tgrid, Deltat,beta)
        res = scipy.optimize.minimize(fhere,pol, method='L-BFGS-B', jac=True,options= {"disp" :False,"gtol":1e-14,"ftol":1e-14})
        weight, _, residue = get_weight_t(res.x, tgrid, Deltat,beta)
        M = -kernel(tgrid_dense/beta, res.x*beta)
        residue_dense = M@weight.reshape((weight.shape[0], weight.shape[1]*weight.shape[2])) - Deltat_dense.reshape((Deltat_dense.shape[0], Deltat_dense.shape[1]*Deltat_dense.shape[2]))
        error = np.linalg.norm(residue_dense.flatten()) / np.sqrt(len(tgrid_dense))
        error =error/Num_of_nonzero_entries
        if error<eps:
            return weight, res.x, error
        
    return weight, res.x, np.linalg.norm(residue)
