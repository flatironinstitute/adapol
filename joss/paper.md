---
title: 'Adapol: Adaptive pole fitting for Matsubara Green’s functions'
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
  - name: Chia-nan Yeh
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
 - name: Department of Mathematics, University of California, Berkeley, California, 94720, USA
   index: 1
 - name: Center for Computational Quantum Physics, Flatiron Institute, New York, NY 10010, USA
   index: 2
 - name: Center for Computational Mathematics, Flatiron Institute, New York, NY 10010, USA
   index: 3
 - name: Applied Mathematics and Computational Research Division, Lawrence Berkeley National Laboratory, Berkeley, California 94720, USA
   index: 4
date: 10 October 2024
bibliography: paper.bib

---

# Summary

Matsubara Green's functions encode thermal properties of a quantum many-body system and are directly related to experimentally measurable quantities such as  spectral density and linear response.
As a result, the decomposition of Matsubara Green's functions often appears as a crucial subroutine in various quantum many-body calculation frameworks.
In the frequency domain, such decompositions are known as the following pole fitting problem:
$$G({i} \nu_k) \approx \sum_{p=1}^{N_p} \frac{v_pv_p^\dagger}{{i} \nu_k-E_p}.$$
Such pole fitting problems arise naturally in various applications, such as in hybridization fitting [@georges1996dynamical; @mejuto2020efficient; @HuangKayeStrand2024etal] and analytic continuation of Matsubara Green's functions \cite{fei2021nevanlinna,HuangGullLin2023}.
The Matsubara data usually has non-zero non-diagonal components, hindering the problem from being treated in a component-wise fashion.
The fitting problem \cref{eq:pole_fitting} has been found to have a highly nonconvex loss landscape (see \cite[Fig 3]{HuangGullLin2023}). The residue matrix $R_p$ corresponding to the pole $E_p$, i.e., $R_p = v_pv_p^\dagger$, is constrained to be a rank-1 positive semidefinite matrix. Furthermore, the given Matsubara data could be potentially noisy, making methods based solely on rational interpolations possibly unstable \cite{schott2016analytic, fei2021nevanlinna}.

We implemented an adaptive pole fitting (Adapol) procedure to circument the above challenges \cite{HuangGullLin2023,HuangKayeStrand2024etal}.
We first use the AAA algorithm \cite{NakatsukasaSeteTrefethen2018}, which is a state-of-the-art rational approximation algorithm, to find an initial guess for the poles $E_p$, then use linear fitting, optimization and singular value decomposition to obtain the final values of $E_p$ and $v_p$.
The adapol procedure can provide an accurate fitting to the Matsubara data, is noise-robust and also gives  results guranteed to satisfy causality.

Adapol could be regarded as an efficient approach for obtaining a low-rank compression of the Matsubara data, where the number of terms $N_p$ is comparable (and usually smaller) than the DLR representation \cite{HuangKayeStrand2024etal}, which is known to achieve the optimal scaling \cite{kaye2022discrete,kaye2022libdlr}. This has enabled recent algorithmic advances in dynamical mean-field theory \cite{mejuto2020efficient} and fast Feynman diagram evaluations \cite{kayeHuangStrandetat2024}.

# Statement of need

Adapol is a python library which conducts the adaptive pole fitting for Matsubara Green's functions. 
The analytic continuation of Green's functions using pole fitting was implemented in Matlab \cite{HuangGullLin2023}.
Nevertheless, Adapol provides a very crucial framework for future code developments. On the one hand, it incorporates more application scenarios by providing application interface (API) for both hybridization fitting and analytic continuations. On the other hand, 
adapol is implemented in python, and provides a user-friendly interface for TRIQS \cite{parcollet2015triqs}, enabling TRIQS users to use Adapol with very slight modifications of their existing routines.

Adapol is distributed under the Apache License Version 2.0 through a public Git repository
\cite{Huang2024}. The online project documentation \cite{Huang2024_2} contains  background on the physics of Matsubara Green's functions and the mathematics of pole fitting, a detailed user guide describing several physical examples and the TRIQS interface, and also the API reference documentation for all functions. 

# Acknowledgements

The work by Z.H. is supported by the Simons Targeted Grants in Mathematics and Physical Sciences on Moir\'e Materials Magic.
