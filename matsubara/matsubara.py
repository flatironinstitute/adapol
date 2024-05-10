import numpy as np
from .fit_utils import pole_fitting, eval_with_pole, check_psd


def anacont(
    Delta,
    iwn_vec,
    tol=None,
    Np=None,
    solver="lstsq",
    maxiter=500,
    mmin=4,
    mmax=50,
    verbose=False,
):
    """
    The main fitting function for both hybridization fitting.
    Examples:
    --------

    func = anacont(Np = Np) # analytic continuation with Np poles
    func = anacont(tol = tol) # analytic continuation with fixed error tolerance tol

    Analytic continuation with improved accuracy:
        fitting(tol = tol, flag = flag, cleanflag = False)
        fitting(Np = Np, flag = flag, cleanflag = False)

    Parameters:
    --------
    Delta: np.array (Nw, Norb, Norb)
        The input Matsubara function in Matsubara frequency

    iwn_vec: np.array (Nw)
        The Matsubara frequency vector, complex-valued


    tol: Fitting error tolreance, float
        If tol is specified, the fitting will be conducted with fixed error tolerance tol.
        default: None

    Np: number of poles, integer
        If Np is specified, the fitting will be conducted with fixed number of poles.
        default: None
        Np needs to be an odd integer, and number of supoort points is Np + 1.


    solver: string
        The solver that is used for optimization.
        choices: "lstsq", "sdp"
        default: "lstsq"

    maxiter: int
        maximum number of iterations
        default: 500

    mmin, mmax: number of minimum or maximum poles, integer
        default: mmin = 4, mmax = 50
        if tol is specified, mmin and mmax will be used as the minimum and maximum number of poles.
        if Np is specified, mmin and mmax will not be used.

    verbose: bool
        whether to display optimization details
        default: False




    Returns:
    --------
    func: function
            Analytic continuation function
            func(w) = sum_n weight[n]/(w-pol[n])

    fitting_error: float
        fitting error

    pol: np.array (Np)
        poles obtained from fitting

    weight: np.array (Np, Norb, Norb)
        weights obtained from fitting

    """

    # Check dimensions
    assert len(iwn_vec.shape) == 1 or len(iwn_vec.shape) == 2
    if len(iwn_vec.shape) == 2:
        assert iwn_vec.shape[1] == 1
        iwn_vec = iwn_vec.flatten()
    assert len(Delta.shape) == 3 or len(Delta.shape) == 1
    if len(Delta.shape) == 1:
        assert Delta.shape[0] == iwn_vec.shape[0]
        Delta = Delta[:, None, None]
    if len(Delta.shape) == 3:
        assert Delta.shape[0] == iwn_vec.shape[0]
        assert Delta.shape[1] == Delta.shape[2]

    solver = solver.lower()
    assert solver == "lstsq" or solver == "sdp"

    # Check input tol or Np
    if tol is None and Np is None:
        raise ValueError("Please specify either tol or Np")
    if tol is not None and Np is not None:
        raise ValueError(
            "Please specify either tol or Np. One can not specify both of them."
        )
    
    wn_vec = np.imag(iwn_vec)

    if Np is not None:
        pol, weight, fitting_error = pole_fitting(
            Delta, wn_vec, Ns=Np+1, maxiter=maxiter, solver=solver, disp=verbose
        )
    elif tol is not None:
        pol, weight, fitting_error = pole_fitting(
            Delta,
            wn_vec,
            tol=tol,
            mmin=mmin,
            mmax=mmax,
            maxiter=maxiter,
            solver=solver,
            disp=verbose,
        )
    def func(Z):
        return eval_with_pole(pol, Z, weight)
    return func, fitting_error, pol, weight


def anacont_triqs(
    Delta_triqs,
    tol=None,
    Np=None,
    solver="lstsq",
    maxiter=500,
    mmin=4,
    mmax=50,
    verbose=False,
):
    """
    The triqs interface for analytical continuation.
    The function requires triqs package in python.
    Examples:
    --------

    func = anacont(Np = Np) # analytic continuation with Np poles
    func = anacont(tol = tol) # analytic continuation with fixed error tolerance tol

    Parameters:
    --------
    Delta_triqs: triqs Green's function container
        The input hybridization function in Matsubara frequency

    tol, Np, cleanflag, maxiter, mmin, mmax, disp: see above in anacont


    Returns:
    --------
    func: function
            Analytic continuation function
            func(w) = sum_n weight[n]/(w-pol[n])

    fitting_error: float
        fitting error

    pol: np.array (Np)
        poles obtained from fitting

    weight: np.array (Np, Norb, Norb)
        weights obtained from fitting

    """
    try:
        from triqs.gf.meshes import MeshDLRImFreq
        from triqs.gf import MeshImFreq
    except ImportError:
        raise ImportError("Failed to import the triqs package (https://triqs.github.io/triqs/latest/). "
                          "Please ensure it is installed.")

    if not isinstance(Delta_triqs.mesh, (MeshImFreq, MeshDLRImFreq)):
        raise TypeError("Delta.mesh must be an instance of MeshImFreq or MeshDLRImFreq.")

    delta_data = Delta_triqs.data.copy()
    iwn_vec = np.array([iw.value for iw in Delta_triqs.mesh.values()])

    return anacont(delta_data, iwn_vec, tol, Np, solver, maxiter,
                   mmin, mmax, verbose)


def hybfit(
    Delta,
    iwn_vec,
    tol=None,
    Np=None,
    svdtol=1e-7,
    solver="lstsq",
    maxiter=500,
    mmin=4,
    mmax=50,
    verbose=False,
):
    """
    The main fitting function for both hybridization fitting.
    Examples:
    --------

        hybfit(Delta, iwn_vec, Np = Np) # hybridization fitting with Np poles
        hybfit(Delta, iwn_vec, tol = tol) # hybridization fitting with fixed error tolerance tol

    Parameters:
    --------
    Delta: np.array (Nw, Norb, Norb)
        The input hybridization function in Matsubara frequency

    iwn_vec: np.array (Nw)
        The Matsubara frequency vector, complex-valued

    svdtol: float, optional
        Truncation threshold for bath orbitals while doing SVD of weight matrices in hybridization fitting
        default:1e-7

    tol, Np, cleanflag, maxiter, mmin, mmax, disp: see above in anacont




    Returns:
    --------


    bathenergy: np.array (Nb)
        Bath energy

    bathhyb: np.array (Nb, Norb)
        Bath hybridization

    final_error: float
        final fitting error

    func: function
        Hybridization function evaluator
        func(w) = sum_n bathhyb[n,i]*conj(bathhyb[n,j])/(1j*w-bathenergy[n])


    """

   
    # Check dimensions
    assert len(iwn_vec.shape) == 1 or len(iwn_vec.shape) == 2
    if len(iwn_vec.shape) == 2:
        assert iwn_vec.shape[1] == 1
        iwn_vec = iwn_vec.flatten()
    assert len(Delta.shape) == 3 or len(Delta.shape) == 1
    if len(Delta.shape) == 1:
        assert Delta.shape[0] == iwn_vec.shape[0]
        Delta = Delta[:, None, None]
    if len(Delta.shape) == 3:
        assert Delta.shape[0] == iwn_vec.shape[0]
        assert Delta.shape[1] == Delta.shape[2]

    solver = solver.lower()
    assert solver == "lstsq" or solver == "sdp"

    # Check input tol or Np
    if tol is None and Np is None:
        raise ValueError("Please specify either tol or Np")
    if tol is not None and Np is not None:
        raise ValueError(
            "Please specify either tol or Np. One can not specify both of them."
        )
    
    wn_vec = np.imag(iwn_vec)

    if Np is not None:
        pol, weight, fitting_error = pole_fitting(
            Delta, wn_vec, Ns=Np + 1, maxiter=maxiter, solver=solver, disp=verbose
        )
    elif tol is not None:
        pol, weight, fitting_error = pole_fitting(
            Delta,
            wn_vec,
            tol=tol,
            mmin=mmin,
            mmax=mmax,
            maxiter=maxiter,
            solver=solver,
            disp=verbose,
        )

    bathenergy, bathhyb, bath_mat = obtain_orbitals(pol, weight, svdtol=svdtol)
    def func(Z):
        return eval_with_pole(bathenergy, Z, bath_mat)
    Delta_reconstruct = func(iwn_vec)
    final_error = np.max(np.abs(Delta - Delta_reconstruct))
    return bathenergy, bathhyb, final_error, func


def hybfit_triqs(
    Delta_triqs,
    tol=None,
    Np=None,
    svdtol=1e-7,
    solver="lstsq",
    maxiter=500,
    mmin=4,
    mmax=50,
    verbose=False,
    debug=False
):
    """
    The triqs interface for hybridization fitting.
    The function requires triqs package in python.
    Examples:
    --------

        hybfit_triqs(delta_triqs, Np = Np) # hybridization fitting with Np poles
        hybfit_triqs(delta_triqs, tol = tol) # hybridization fitting with fixed error tolerance tol

    Parameters:
    --------
    Delta_triqs: triqs Green's function container
        The input hybridization function in Matsubara frequency

    tol, Np, cleanflag, maxiter, mmin, mmax, disp: see above in hybfit

    Returns:
    --------

    bathhyb: np.array (Nb, Norb)
        Bath hybridization

    bathenergy: np.array (Nb)
        Bath energy

    Delta_fit: triqs Gf or BlockGf
        Discretized hybridization function
        The input hybridization function in Matsubara frequency

    if debug == True:
        final_error: float
            final fitting error

        weight: np.array (Np, Norb, Norb)
            weights obtained from fitting
    """
    try:
        from triqs.gf.meshes import MeshDLRImFreq
        from triqs.gf import MeshImFreq
        from triqs.gf.gf import Gf
        from triqs.gf.block_gf import BlockGf
    except ImportError:
        raise ImportError("Failed to import the triqs package (https://triqs.github.io/triqs/latest/). "
                          "Please ensure it is installed.")

    if isinstance(Delta_triqs, Gf) and isinstance(Delta_triqs.mesh, (MeshImFreq, MeshDLRImFreq)):
        iwn_vec = np.array([iw.value for iw in Delta_triqs.mesh.values()])
        results = hybfit(Delta_triqs.data, iwn_vec, tol, Np, svdtol, solver, maxiter, mmin, mmax, verbose)
        eps_opt, V_opt, final_error, func,  = hybfit(Delta_triqs.data, iwn_vec, tol, Np,
                                                   svdtol, solver, maxiter, mmin, mmax, verbose)[:4]
        print('optimization finished with fitting error {:.3e}'.format(final_error))

        delta_fit = Gf(mesh=Delta_triqs.mesh, target_shape=Delta_triqs.target_shape)
        delta_fit.data[:] = results[3](iwn_vec)

        if debug:
            # V_opt, eps_opt, delta_fit, error, weight
            return results[1].T.conj(), results[0], delta_fit, results[2], results[5]
        else:
            return results[1].T.conj(), results[0], delta_fit
    elif isinstance(Delta_triqs, BlockGf) and isinstance(Delta_triqs.mesh, (MeshImFreq, MeshDLRImFreq)):
        V_list, eps_list, delta_list, error_list, weight_list = [], [], [], [], []
        for j, (block, delta_blk) in enumerate(Delta_triqs):
            res = hybfit_triqs(delta_blk, tol, Np, svdtol, solver, maxiter, mmin, mmax, verbose, debug)
            V_list.append(res[0])
            eps_list.append(res[1])
            delta_list.append(res[2])
            if debug:
                error_list.append(res[3])
                weight_list.append(res[4])

        if debug:
            return V_list, eps_list, BlockGf(name_list=list(Delta_triqs.indices), block_list=delta_list), error_list, weight_list
        else:
            return V_list, eps_list, BlockGf(name_list=list(Delta_triqs.indices), block_list=delta_list)
    else:
        raise RuntimeError("Error: Delta_triqs.mesh must be an instance of MeshImFreq or MeshDLRImFreq.")


def obtain_orbitals(pol, weight, svdtol=1e-7):
    """
    obtaining bath orbitals through svd
    """
    polelist = []
    veclist = []
    matlist = []
    for i in range(weight.shape[0]):
        eigval, eigvec = np.linalg.eig(weight[i])
        for j in range(eigval.shape[0]):
            if eigval[j] > svdtol:
                polelist.append(pol[i])
                veclist.append(eigvec[:, j] * np.sqrt(eigval[j]))
                matlist.append(
                    (eigvec[:, j, None] * np.conjugate(eigvec[:, j].T)) * (eigval[j])
                )

    return np.array(polelist), np.array(veclist), np.array(matlist)


def check_weight_psd(weight, atol=1e-6):
    """
    check whether the weight matrices are positive semidefinite
    """

    return check_psd(weight, atol=atol)
