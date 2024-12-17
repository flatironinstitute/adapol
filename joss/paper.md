---
title: 'adapol: Adaptive pole-fitting for Matsubara functions'
tags:
  - python
  - quantum many-body systems
  - imaginary time Green's function
  - Matsubara Green's function
  - many-body Green's function methods
  - pole-fitting
  - hybridization fitting
  - analytic continuation
authors:
  - name: Zhen Huang
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1
  - name: Chia-Nan Yeh
    orcid: 0000-0002-4166-0764
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

The Green's function approach to quantum many-body physics aims to replace high-dimensional wavefunctions with correlation functions which are more closely related to experimental observables of interest, such as spectral and response functions.
Within this framework, real-time quantities, such as the Green's function, self-energy, and hybridization functions, are often represented in the discrete "Matsubara" domain on the imaginary frequency axis. A variety of physical observables can be directly recovered from the Matsubara Green's function, and many quantities of interest can be calculated more efficiently in this formalism.

A common computational task within this framework is decomposing a Matsubara function into a sum of simple poles:
$$G_{ab}(\mathrm{i} \omega_n) \approx \sum_{p=1}^{N_p} \frac{v_{ap}v_{bp}^{*}}{\mathrm{i} \omega_n-E_p}.$$
Here, $\mathbf{G}(i\omega_{n})$ is a matrix-valued function of the Matsubara frequency point $\mathrm i \omega_n = (2n+1) \pi\mathrm{i} / \beta$ for fermionic functions, and $i \omega_n = 2 n \pi\mathrm{i} / \beta$ for bosonic functions, with $\beta$ representing the inverse temperature and $n \in \mathbb{Z}$. The pair $(E_p,\mathbf{v}_p)$ can be interpreted as a set of eigenpairs for an effective non-interacting Hamiltonian model.
This "pole-fitting" problem is a crucial step in various numerical methods, such as hybridization fitting for quantum impurity solvers [@georges1996dynamical; @mejuto2020efficient; @shinaoka21; @kaye24; @gazizova24], and the analytic continuation of Matsubara Green's functions [@fei2021nevanlinna; @fei21_2; @ying22; @huang2023; @zhang24; @zhang24_2].
As a consequence of the rank-one positive-semidefiniteness constraint implied by the model, a best fit from Matsubara frequency data cannot be obtained component-wise, leading to a highly non-convex optimization landscape (see, e.g., Fig. 3 in [@huang2023]). Additionally, given that Matsubara data may be noisy in certain applications, instabilities can arise in methods which rely solely on standard rational approximation techniques [@schott2016analytic; @fei2021nevanlinna].

Our Python package `adapol` ("add a pole") implements an adaptive pole-fitting procedure introduced in [@huang2023; @huang2024_3].
The method first uses the AAA rational approximation algorithm [@nakatsukasa2018] to find an initial guess for the pole locations $E_p$. It then uses non-convex optimization and singular value decomposition to refine $E_p$ and obtain $v_p$.
Variants of this procedure have been shown to provide an accurate and compact fit for Matsubara data in a black-box and noise-robust manner, enabling new algorithms for dynamical mean-field theory [@mejuto2020efficient] and Feynman diagram evaluation [huang2024_3]. For example, [huang2024_3] demonstrates that the procedure yields a more compact pole approximation than the generic discrete Lehmann representation [@kaye2022discrete] for fixed objective functions. 

# Statement of Need

`adapol` is a simple and self-contained package which can be incorporated into codes requiring Matsubara pole-fitting. It includes a specialized API for common tasks such as hybridization fitting and analytic continuation, along with a user-friendly interface to the TRIQS package [@parcollet2015triqs], enabling TRIQS users to utilize `adapol` with minimal modifications to their existing code.

`adapol` is distributed under the Apache License Version 2.0 and is available on GitHub [@huang2024]. The documentation [@huang2024_2] provides background on the physics of Matsubara functions and the mathematics of pole-fitting, along with a detailed user guide, example applications, the TRIQS interface, and API reference documentation for all functions.

# Acknowledgements

The work by Z.H. is supported by the Simons Targeted Grants in Mathematics and Physical Sciences on Moiré Materials Magic. The Flatiron Institute is a division of the Simons Foundation.
