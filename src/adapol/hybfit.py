from .fit_utils import pole_fitting, eval_with_pole, check_psd

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
    The function for hybridization fitting.

    Examples:
    ----------

        -  Fitting with :math:`N_p` poles:
            :code:`hybfit(Delta, iwn_vec, Np = Np)` 

        - Fitting with fixed error tolerance tol :
            :code:`hybfit(Delta, iwn_vec, tol = tol)` 

    Parameters:
    ------------
    :code:`Delta`: np.array, :math:`(N_w, N_\mathrm{orb}, N_\mathrm{orb})`
        The input hybridization function in Matsubara frequency.

    :code:`iwn_vec`: np.array, :math:`(N_w)`
        The Matsubara frequency vector, complex-valued 

    :code:`svdtol`: float, optional
        Truncation threshold for bath orbitals while doing SVD of weight matrices in hybridization fitting
        default:1e-7

    :code:`tol`, :code:`Np`, :code:`solver`, :code:`maxiter`, :code:`mmin`, :code:`mmax`, :code:`verbose`: 
        see below in anacont


    Returns:
    ---------


    :code:`bathenergy` :math:`E`: np.array, :math:`(N_b)`
        Bath energy

    :code:`bathhyb` :math:`V`: np.array, :math:`(N_b,N_{\mathrm{orb}})` 
        Bath hybridization

    :code:`final_error`: float
        final fitting error

    :code:`func`: function
        Hybridization function evaluator
        :math:`f(z) = \sum_n V_{ni}V_{nj}^*/(z-E_n).`

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
    ----------

        -  Fitting with :math:`N_p` poles:
            :code:`hybfit_triqs(delta_triqs, Np = Np)` 

        - Fitting with fixed error tolerance tol :
            :code:`hybfit_triqs(delta_triqs, tol = tol)` 


    Parameters:
    ------------
    :code:`Delta_triqs`: triqs Green's function container
        The input hybridization function in Matsubara frequency

    :code:`debug`: bool
        return additional outputs for debugging.
        Default: False

    :code:`svdtol`: float, optional
        Truncation threshold for bath orbitals while doing SVD of weight matrices in hybridization fitting
        default: 1e-7

    :code:`tol`, :code:`Np`, :code:`solver`, :code:`maxiter`, :code:`mmin`, :code:`mmax`, :code:`verbose`: 
        same as in hybfit

    Returns:
    ---------

    :code:`bathhyb`: np.array :math:`(N_b, N_\mathrm{orb})`
        Bath hybridization

    :code:`bathenergy`: np.array :math:`(N_b,)`
        Bath energy

    :code:`Delta_fit`: triqs Gf or BlockGf
        Discretized hybridization function
        The input hybridization function in Matsubara frequency

    if debug is True:
        :code:`final_error`: float
            final fitting error

        :code:`weight`: np.array :math:`(N_p, N_\mathrm{orb}, N_\mathrm{orb})`
            weights obtained from fitting
    """
    try:
        from triqs.gf import Gf, BlockGf, MeshImFreq, MeshDLRImFreq
    except ImportError:
        raise ImportError("Failed to import the triqs package (https://triqs.github.io/triqs/latest/). "
                          "Please ensure it is installed.")

    if isinstance(Delta_triqs, Gf) and isinstance(Delta_triqs.mesh, (MeshImFreq, MeshDLRImFreq)):
        iwn_vec = np.array([iw.value for iw in Delta_triqs.mesh.values()])
        eps_opt, V_opt, final_error, func = hybfit(Delta_triqs.data, iwn_vec, tol, Np,
                                                   svdtol, solver, maxiter, mmin, mmax, verbose)
        print('optimization finished with fitting error {:.3e}'.format(final_error))

        delta_fit = Gf(mesh=Delta_triqs.mesh, target_shape=Delta_triqs.target_shape)
        delta_fit.data[:] = func(iwn_vec)

        if debug:
            return V_opt.T.conj(), eps_opt, delta_fit, final_error
        else:
            return V_opt.T.conj(), eps_opt, delta_fit
    elif isinstance(Delta_triqs, BlockGf) and isinstance(Delta_triqs.mesh, (MeshImFreq, MeshDLRImFreq)):
        V_list, eps_list, delta_list, error_list = [], [], [], []
        for j, (block, delta_blk) in enumerate(Delta_triqs):
            res = hybfit_triqs(delta_blk, tol, Np, svdtol, solver, maxiter, mmin, mmax, verbose, debug)
            V_list.append(res[0])
            eps_list.append(res[1])
            delta_list.append(res[2])
            if debug:
                error_list.append(res[3])

        if debug:
            return V_list, eps_list, BlockGf(name_list=list(Delta_triqs.indices), block_list=delta_list), error_list
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
