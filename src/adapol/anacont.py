import numpy as np
from .fit_utils import pole_fitting, eval_with_pole

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
    statistics="Fermion",
):
    """
    The function for analytical continuation.

    Examples:
    ----------

        -  Analytic continuation with :math:`N_p` poles:
            :code:`func = anacont(Np = Np)` 

        - Fitting with fixed error tolerance tol:
            :code:`func = anacont(tol = tol)` 
            
        - Analytic continuation with improved accuracy:
            :code:`fitting(Np = Np, flag = flag, solver = "sdp")`

    Parameters:
    ------------
    :code:`Delta`: np.array, :math:`(N_w, N_\mathrm{orb}, N_\mathrm{orb})`
        The input hybridization function in Matsubara frequency.

    :code:`iwn_vec`: np.array, :math:`(N_w,)`
        The Matsubara frequency vector, complex-valued 


    :code:`tol`: Fitting error tolreance, float
        If tol is specified, the fitting will be conducted with fixed error tolerance tol.
        default: None

    :code:`Np`: number of poles, integer
        If Np is specified, the fitting will be conducted with fixed number of poles Np.
        default: None
        Np needs to be an odd integer, and number of supoort points is Np + 1.


    :code:`solver`: string
        The solver that is used for optimization.
        choices: "lstsq", "sdp"
        default: "lstsq"

    :code:`maxiter`: int
        maximum number of iterations
        default: 500

    :code:`mmin`, :code:`mmax`: number of minimum or maximum poles, integer
        default: mmin = 4, mmax = 50
        if tol is specified, mmin and mmax will be used as the minimum and maximum number of poles.
        if Np is specified, mmin and mmax will not be used.

    :code:`verbose`: bool
        whether to display optimization details
        default: False

    :code:`statistics`: str
        statistics of the hybridization function. Currently "Fermion" and "Boson" is supported.
        Default: "Fermion"




    Returns:
    ---------
    :code:`func`: function
            Analytic continuation function. In the fermionic case:
            :math:`f(z) = \sum_n \mathrm{Weight}[n]/(z-\mathrm{pol}[n]).`
            In the bosonic case:
            :math:`f(z) = \sum_n \mathrm{Weight}[n]\mathrm{pol}[n]/(z-\mathrm{pol}[n]).`

    :code:`fitting_error`: float
        fitting error

    :code:`pol`: np.array, :math:`(N_p,)`
        poles obtained from fitting

    :code:`weight`: np.array, :math:`(N_p, N_\mathrm{orb}, N_\mathrm{orb})`
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
            Delta, wn_vec, Ns=Np+1, maxiter=maxiter, solver=solver, disp=verbose, statistics=statistics
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
            statistics=statistics
        )
    def func(Z):
        return eval_with_pole(pol, Z, weight, statistics=statistics)
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
    debug=False,
    statistics="Fermion"
):
    """
    The triqs interface for analytical continuation.
    The function requires triqs package in python.

    Parameters:
    ------------
    :code:`Delta_triqs`: triqs Green's function container
        The input hybridization function in Matsubara frequency

    :code:`debug`: bool
        return additional outputs for debugging.
        Default: False

    :code:`tol`, :code:`Np`, :code:`solver`, :code:`maxiter`, :code:`mmin`, :code:`mmax`, :code:`verbose`: 
        same as in anacont

    :code:`statistics`: str
        statistics of the hybridization function. Currently "Fermion" and "Boson" is supported.
        Default: "Fermion"

    Returns:
    ---------
    :code:`func`: function
            Analytic continuation function:
            :math:`f(z) = \sum_n \mathrm{Weight}[n]/(z-\mathrm{pol}[n]).`
    
    if debug == True:
        :code:`fitting_error`: float
            fitting error

        :code:`pol`: np.array, :math:`(N_p,)`
            poles obtained from fitting

        :code:`weight`: np.array, :math:`(N_p, N_\mathrm{orb}, N_\mathrm{orb})`
            weights obtained from fitting

    """
    try:
        from triqs.gf import Gf, BlockGf, MeshImFreq, MeshDLRImFreq
    except ImportError:
        raise ImportError("Failed to import the triqs package (https://triqs.github.io/triqs/latest/). "
                          "Please ensure it is installed.")

    if isinstance(Delta_triqs, Gf) and isinstance(Delta_triqs.mesh, (MeshImFreq, MeshDLRImFreq)):
        iwn_vec = np.array([iw.value for iw in Delta_triqs.mesh.values()])
        func, fit_error, pol, weight = anacont(Delta_triqs.data, iwn_vec, tol, Np, solver, maxiter,
                                               mmin, mmax, verbose, statistics=statistics)
        print('optimization finished with fitting error {:.3e}'.format(fit_error))

        if debug:
            return func, fit_error, pol, weight
        else:
            return func
    elif isinstance(Delta_triqs, BlockGf) and isinstance(Delta_triqs.mesh, (MeshImFreq, MeshDLRImFreq)):
        func_list, error_list, pol_list, weight_list = [], [], [], []
        for j, (block, delta_blk) in enumerate(Delta_triqs):
            func, fit_error, pol, weight = anacont_triqs(delta_blk, tol, Np, solver, maxiter, mmin,
                                                         mmax, verbose, statistics=statistics)
            func_list.append(func)
            if debug:
                error_list.append(fit_error)
                pol_list.append(pol)
                weight_list.append(weight)

        if debug:
            return func_list, error_list, pol_list, weight_list
        else:
            return func_list
    else:
        raise RuntimeError("Error: Delta_triqs.mesh must be an instance of MeshImFreq or MeshDLRImFreq.")
