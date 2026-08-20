---
title: 'adapol: Adaptive pole-fitting for quantum many-body physics'
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
    orcid: 0000-0002-4801-8635
    affiliation: 1
  - name: Chia-Nan Yeh
    orcid: 0000-0002-4166-0764
    affiliation: 2
  - name: Lin Lin
    orcid: 0000-0001-6860-9566
    affiliation: "1, 4, 5"
  - name: Nils Wentzell
    orcid: 0000-0003-3613-007X
    affiliation: 2
  - name: Jason Kaye
    corresponding: true
    orcid: 0000-0001-8045-6179
    affiliation: "2, 3"
  - name: Hugo U. R. Strand
    orcid: 0000-0002-7263-4403
    affiliation: 6
affiliations:
 - name: Department of Mathematics, University of California, Berkeley, CA 94720, USA
   index: 1
 - name: Center for Computational Quantum Physics, Flatiron Institute, New York, NY 10010, USA
   index: 2
 - name: Center for Computational Mathematics, Flatiron Institute, New York, NY 10010, USA
   index: 3
 - name: Applied Mathematics and Computational Research Division, Lawrence Berkeley National Laboratory, Berkeley, CA 94720, USA
   index: 4
 - name: Department of Computing and Mathematical Sciences, California Institute of Technology
   index: 5
 - name: School of Science and Technology, Örebro University, SE-70182 Örebro, Sweden
   index: 6
   
date: 20 August 2026
bibliography: paper.bib
link-citations: true

---

# Summary

The Green's function approach to quantum many-body physics aims to replace high-dimensional wavefunctions with correlation functions which are more closely related to experimental observables of interest, such as spectral and response functions.
Within this framework, real-time quantities, such as the Green's function, self-energy, and hybridization functions, are often represented in the discrete "Matsubara" domain on the imaginary frequency axis. A variety of physical observables can be directly recovered from the Matsubara Green's function, and many quantities of interest can be calculated more efficiently in this formalism.

A common computational task within this framework is decomposing a Matsubara function into a sum of simple poles:
$$G(\mathrm{i} \nu_n) \approx \sum_{k=1}^{M} \frac{R_k}{\mathrm{i} \nu_n-p_k}.$$
Here, $G(\mathrm{i}\nu_{n})$ is in general an $m \times m$ matrix-valued function of the Matsubara frequency point $\mathrm{i} \nu_n = (2n+1) \pi\mathrm{i} / \beta$ for fermionic functions, and $\mathrm{i} \nu_n = 2 n \pi\mathrm{i} / \beta$ for bosonic functions, with $\beta$ representing the inverse temperature, $n \in \mathbb{Z}$, and $m$ the number of quantum states or spin-orbitals. The $p_k$ are real pole locations, and the $R_k$ are the corresponding matrix-valued residues. In applications such as hybridization fitting, the poles and residues define an effective non-interacting model, with the $p_k$ playing the role of energy levels, and it is often desirable to obtain an accurate fit with as few poles as possible.

Since the pole locations enter the approximation nonlinearly and are shared by all components of a matrix-valued function, a best fit from Matsubara frequency data cannot be obtained component-wise, leading to a highly non-convex optimization landscape.
`adapol` ("add a pole") is a Python package implementing the adaptive pole-fitting procedure outlined in [@huang25; @huang2023].
The method uses a modified version of the AAA rational approximation algorithm [@nakatsukasa2018] to obtain a guess of the pole locations $p_k$, which can optionally be refined by non-convex optimization. The residues $R_k$ are then obtained by a linear least-squares fit. This procedure has been shown to provide an accurate and compact fit of Matsubara data in a black-box and noise-robust manner [@huang25; @zima26]. 

# Statement of Need

The "pole-fitting" problem described above is a crucial step in various numerical methods, such as hybridization fitting for quantum impurity solvers [@georges1996dynamical] based on exact diagonalization [@caffarel94; @liebsch11; @mejuto2020efficient], perturbation theory [@kaye24; @huang25], and time evolution of matrix product states [@wolf15; @zima26], as well as certain approaches to analytic continuation of Matsubara Green's functions [@fei2021nevanlinna; @fei21_2; @ying22; @ying22_2; @huang2023; @zhang24; @zhang24_2], and other perturbation theory-based diagrammatic methods [@gazizova24; @gazizova25]. In many applications (e.g., dynamical mean-field theory), the pole-fitting step appears inside a self-consistent loop, requiring a black-box algorithm delivering results with controlled accuracy.

Although significant progress has been made in the past several years on developing algorithms to solve the pole-fitting problem [@mejuto2020efficient; @shinaoka21; @huang2023; @huang25; @ying22; @ying22_2; @zhang24; @zhang24_2], a lack of widely-deployed and user-friendly software has limited the adoption of these methods. Practitioners often still rely on ad-hoc or older, easy-to-implement methods: for example, brute force optimization methods for pole-fitting, or Padé approximants for analytic continuation [@vidberg77]. `adapol` addresses this gap by providing a simple, self-contained interface with few user parameters, tailored for common applications.

# State of the field

We note two primary approaches which have recently been pursued in the literature on the pole-fitting problem: methods based on (i) Prony's method and its variants [@ying22; @ying22_2; @zhang24; @zhang24_2], and (ii) AAA rational approximation [@nakatsukasa2018] followed by non-convex optimization [@huang2023; @huang25]. The MiniPole Python package [@minipole] implements the Prony's method-based approach described in [@zhang24; @zhang24_2], while `adapol` implements the AAA-based approach described in [@huang2023; @huang25]. These methods are distinct, and the availability of both packages will allow users to compare the two approaches.

# Software design

`adapol` is a simple and self-contained package which can be incorporated into codes requiring Matsubara pole-fitting. Users can provide Matsubara data, or an existing pole expansion to be compressed; for example, a discrete Lehmann representation (DLR) [@kaye2022discrete]. They can choose to perform the fit with or without optimization-based post-processing of the AAA result, and can specify either a maximum number of poles or a target error tolerance. `adapol` functions are documented extensively both within the API reference documentation, and in example notebooks demonstrating various use cases and modes of operation. An interface to the TRIQS package [@parcollet2015triqs] is also provided. 

# Research impact statement

In the context of analytic continuation, the AAA-based approach described in [@huang2023] is considered as one of the state-of-the-art methods, and has been cited extensively. For many other pole-fitting applications, reducing the number of poles required to achieve a given accuracy as much as possible is often crucial, as computational costs often scale exponentially with the number of poles. This is the case, for example, in exact diagonalization quantum impurity solvers [@caffarel94; @liebsch11; @mejuto2020efficient] and diagrammatic evaluation methods [@kaye24; @huang25; @gazizova24; @gazizova25]. Recent developments in tensor network-based quantum impurity solvers [@zima26] also benefit substantially from compact pole approximations of the hybridization function, with a significant increase in computational cost observed as the size of this approximation grows. The `adapol` algorithm was shown in [@huang25] to consistently yield a more compact pole approximation of a fixed, given Green's function than the generic DLR approach [@kaye2022discrete], which is itself an exponential-in-$\beta$ improvement over naive uniform frequency grid approaches.

# AI usage disclosure

Generative AI tools such as Claude and Codex were used to assist in writing code, tests, documentation, and examples in the `adapol` package. The content produced by these tools was reviewed and edited by the authors.

# Acknowledgements

This work is partially supported by the Simons Targeted Grants in Mathematics and Physical Sciences on Moiré Materials Magic (Z.H., L.L.). The Flatiron Institute is a division of the Simons Foundation.

# References
