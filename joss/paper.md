---
title: 'adapol: Adaptive pole fitting for Matsubara functions'
tags:
  - python
  - quantum many-body systems
  - imaginary time Green's function
  - Matsubara Green's function
  - many-body Green's function methods
  - pole fitting
  - hybridization fitting
  - analytic continuation
authors:
  - name: Zhen Huang
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1
  - name: Chia-Nan Yeh
    affiliation: 2
  - name: Jason Kaye
    orcid: 0000-0001-8045-6179
    affiliation: "2, 3" # (Multiple affiliations must be quoted)
  - name: Nils Wentzell
    orcid: 0000-0003-3613-007X
    affiliation: 2
  - name: Lin Lin
    affiliation: "1, 4"
affiliations:
 - name: Department of Mathematics, University of California, Berkeley, CA 94720, USA
   index: 1
 - name: Center for Computational Quantum Physics, Flatiron Institute, New York, NY 10010, USA
   index: 2
 - name: Center for Computational Mathematics, Flatiron Institute, New York, NY 10010, USA
   index: 3
 - name: Applied Mathematics and Computational Research Division, Lawrence Berkeley National Laboratory, Berkeley, CA 94720, USA
   index: 4
date: 10 October 2024
bibliography: paper.bib
link-citations: true

---

# Summary

The Green's function approach to quantum many-body physics aims to replace high-dimensional wavefunctions with correlation functions more closely related to experimental observables of interest, such as the spectral function and response functions. Within this framework, real time quantities, such as the Green's function, self-energy, and hybridization functions, are often represented in the discrete "Matsubara" domain on the imaginary frequency axis. A class of physical observables can be recovered directly from the Matsubara Green's function [JK: make more specific], and many quantities of interest can be calculated more efficiently in the Matsubara formalism.

Within this framework, a common computational task is the decomposition of a Matsubara function into a sum of simple poles:
$$G(\mathrm{i} \omega_n) \approx \sum_{p=1}^{N_p} \frac{v_pv_p^\dagger}{\mathrm{i} \omega_n-E_p}.$$
Here, $G$ is a matrix-valued function of the Matsubara frequency point $\mathrm i \omega_n = (2n+1) \pi\mathrm i / \beta$ for fermionic functions, and $i \omega_n = 2 n \pi\mathrm i / \beta$ for bosonic functions, with $\beta$ the inverse temperature and $n \in \mathbb{Z}$, and $(E_p,v_p)$ can be thought of as a collection of eigenpairs for an effective non-interacting Hamiltonian model.
This "pole-fitting" problem arises as a natural step in various numerical methods, such as in hybridization fitting as an input to quantum impurity solvers [@georges1996dynamical; @mejuto2020efficient; @huang2024_3], and analytic continuation of Matsubara Green's functions [@fei2021nevanlinna; @huang2023] [JK: review and add references].
As a consequence of the rank-one positive-semidefiniteness constraint implied by form of the model, the problem of obtaining a best fit given Matsubara frequency data cannot be treated component-wise, and has been found to have a highly non-convex optimization landscape (see, e.g., Fig. 3 in [@huang2023]). Furthermore, in certain applications, the given Matsubara data might be noisy, leading to instability in methods based solely on standard rational approximation techniques [@schott2016analytic;  @fei2021nevanlinna].

Our Python package `adapol` ("add a pole") implements an adaptive pole fitting procedure introduced in [@huang2023,@huang2024_3].
Briefly, the method first uses the AAA rational approximation algorithm [@nakatsukasa2018] to find an initial guess for the pole locations $E_p$, and then use optimization and singular value decomposition to refine $E_p$ and obtain $v_p$.
It has been shown that this procedure provides an accurate fit of the Matsubara data in a black-box and noise-robust manner. It has led to recent algorithmic advances in dynamical mean-field theory [@mejuto2020efficient], as well as Feynman diagram evaluation [@kaye2024], for which it typically provides a more efficient low-rank decomposition than the discrete Lehmann representation [@kaye2022discrete;@kaye2022libdlr] when the function to be fitted is fixed [@huang2024_3]. [JK: This sentence to be reviewed.] [JK: Mention/cite IR work on hybridization fitting.] [JK: In general, references should be reviewed and expanded.]

# Statement of need

`adapol` is a simple and self-contained package which can be incorporated into codes requiring Matsubara pole-fitting. It includes a specialized API for the common tasks of hybridization fitting and analytic continuation, as well as a user-friendly interface to the TRIQS package [@parcollet2015triqs], enabling TRIQS users to use `adapol` with only slight modifications of their existing code.

`adapol` is distributed under the Apache License Version 2.0 and made available on Github [@huang2024]. Its documentation [@huang2024_2] contains  background on the physics of Matsubara functions and the mathematics of pole fitting, as well as a detailed user guide describing example applications, the TRIQS interface, and API reference documentation for all functions. 

# Acknowledgements

The work by Z.H. is supported by the Simons Targeted Grants in Mathematics and Physical Sciences on Moiré Materials Magic. The Flatiron Institute is a division of the Simons Foundation.