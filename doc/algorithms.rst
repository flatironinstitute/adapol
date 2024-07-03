.. _algorithms:

Algorithms 
======================

Obtain bath fitting from pole fitting
--------------------------------------------

In bath fitting, given :math:`\Delta(\mathrm i\nu_k)` evaluated on :math:`\{\mathrm i\nu_k\}_{k=1}^{N_{w}}`, we wish to find 
$V_p, E_p$ such that

.. math::

    \begin{equation}
    \Delta(\mathrm i\nu_k) = \sum_{j=1}^{N_b} \frac{V_jV_j^{\dagger}}{\mathrm i\nu_k - E_j}.
    \end{equation}

This is achieved by the following strategy:
 
- Find pole fitting with semidefinite constraints:

.. math::
    
        \begin{equation}
        \Delta(\mathrm i\nu_k) = \sum_{p=1}^{N_p} \frac{M_p}{\mathrm i\nu_k - \lambda_p}, M_p\geq 0, \tag{1} \label{polefit}
        \end{equation}
    
Here :math:`M_p` are :math:`N\times N` positive semidefinite matrices.

- Compute eigenvalue decomposition of :math:`M_p`:

.. math::

    M_p = \sum_{j=1}^{N} V_{j}^{(p)} (V_{j}^{(p)})^{\dagger}. \tag{2} \label{eigdecomp}

- Combining :math:`\eqref{polefit}` and :math:`\eqref{eigdecomp}`, we obtain the desired bath fitting:

.. math::

    \Delta(\mathrm i\nu_k) = \sum_{pj} \frac{V_{j}^{(p)}(V_{j}^{(p)})^{\dagger}}{\mathrm i\nu_k - \lambda_p}.



Rational approximation via (modified) AAA algorithm
------------------------------------------------------------------------------------------------------------

To find the poles :math:`\lambda_p` in :math:`\eqref{polefit}`, we use the `AAA algorithm <https://epubs.siam.org/doi/10.1137/16M1106122>`_, which is a rational approximation algorithm based on the Barycentric interpolant:

.. math::

    \begin{equation}
    f(z) = \frac{p(z)}{q(z)} = \frac{\sum_{j=1}^{m} \frac{c_jf_j}{z - z_j}}{\sum_{j=1}^{m} \frac{c_j}{z - z_j}.}.
    \tag{3} \label{bary}
    \end{equation}

The AAA algorithm is an iterative procedure. It selects the next support point in a greedy fashion.
Suppose we have obtained an approximant :math:`\widetilde f` from the :math:`(k-1)`-th iteration, using support point :math:`z_1,\cdots z_{k-1}`.
At the :math:`k`-th iteration, we do the following:

#. Select the next support point :math:`z_k` at which the previous approximant :math:`\widetilde f` has the largest error.

#. Find :math:`c_k` in :math:`\eqref{bary}` by solving the following linear square problem:

    .. math::

        \begin{equation}
        \min_{\{c_k\}} \sum_{z\neq z_1,\cdots z_k} \left| f(z) q(z) - p(z) \right|^2. \quad \text{s.t.} \|c\|_2= 1.
        \end{equation}

    This is a linear problem and amounts to solving a SVD problem. (See details in `paper <https://epubs.siam.org/doi/10.1137/16M1106122>`_).

#. If the new approximant has reached desired accuracy, stop the iteration. Otherwise, repeat the above steps.

The poles of :math:`f(z)` are the zeros of :math:`q(z)`, which can be found by solving the following generalized eigenvalue problem:

.. math::

    \begin{equation}
    \left(\begin{array}{ccccc}
    0 & c_1 & c_2 & \cdots & c_m \\
    1 & z_1 & & & \\
    1 & & z_2 & & \\
    \vdots & & & \ddots & \\
    1 & & & & z_m
    \end{array}\right)=\lambda\left(\begin{array}{lllll}
    0 & & & & \\
    & 1 & & & \\
    & & 1 & & \\
    & & & \ddots & \\
    & & & & 1
    \end{array}\right)
    \end{equation}

For our application, we modify the AAA algorithm to deal with matrix-valued functions by replacing :math:`f_j` with matrices :math:`F_j`.

Semidefinite programming
--------------------------------------------

After obtaining :math:`\lambda_p`, we need to find the weight matrices :math:`M_p` in :math:`\eqref{polefit}`.
We are solving the following problem:

.. math::

    \begin{equation}
    \min_{\{M_p\}} \sum_{k=1}^{N_w} \left\| \Delta(\mathrm i\nu_k) - \sum_{p=1}^{N_p} \frac{M_p}{\mathrm i\nu_k - \lambda_p} \right\|_F^2, \quad \text{s.t. } M_p\geq 0.
    \end{equation}

This is a linear problem with respect to :math:`M_p`, and has semidefinite constraints, therefore could be solved efficiently via standard semidefinite programming (SDP) solvers.

Bi-level optimization
--------------------------------------------
With :math:`\lambda_p` and :math:`M_p` obtained, we can further refine the poles and weights by solving the following bi-level optimization.
Let us define the error function as 

.. math::

    \begin{equation}
    \text{Err}(\lambda_p, M_p) =  \sum_{k=1}^{N_w} \left\| \Delta(\mathrm i\nu_k) - \sum_{p=1}^{N_p} \frac{M_p}{\mathrm i\nu_k - \lambda_p} \right\|_F^2.
    \end{equation}

Note that :math:`\text{Err}` is linear in :math:`M_p` but nonlinear in :math:`\lambda_p`. As we have mentioned, optimization in :math:`M_p` is  a SDP problem and therefore is robust, while optimization in :math:`\lambda_p` is a nonlinear problem and could be very challenging.
This motivates us to define :math:`\text{Err}(\lambda_1,\cdots, \lambda_{N_p})` as a function of :math:`\{\lambda_p\}` only:

.. math::

    \begin{equation}
    \text{Err}(\lambda_1,\cdots, \lambda_{N_p}) = \min_{\{M_p\}}\text{Err}(\lambda_p, M_p) 
    \end{equation}

The value of :math:`\text{Err}(\lambda_1,\cdots, \lambda_{N_p})` is obtained by solving a SDP problem.
The gradient of :math:`\text{Err}(\lambda_1,\cdots, \lambda_{N_p})` could also be obtained analytically.
(For details, see eq. 28 `here <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.107.075151>`_.)
And thus we could use a gradient-based optimization algorithm (L-BFGS) to minimize :math:`\text{Err}(\lambda_1,\cdots, \lambda_{N_p})` with respect to :math:`\{\lambda_p\}`.

For performances, robustness and other details of this bi-level optimization framework, see again `our original paper <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.107.075151>`_.