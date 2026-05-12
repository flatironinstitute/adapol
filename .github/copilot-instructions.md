# adapol Copilot instructions

## Build, test, and lint commands

- Install the package in editable mode before running tests against the local source tree: `python3 -m pip install -e .`
- Default CI test command: `python3 -m pytest -m "not triqs" -q`
- Run a single test: `python3 -m pytest test/test_adapol.py::test_freq_tol -q`
- Run only the pure-Python core tests when TRIQS is unavailable: `python3 -m pytest test/test_adapol.py test/test_sop.py test/test_imtime.py -q`
- Lint: `python3 -m ruff check .`
- Build docs: `make -C doc html`

## High-level architecture

- `src/adapol/adapol.py` is the main user-facing API for two entry points: approximating sampled frequency data directly, and compressing an existing sum-of-simple-poles representation to fewer poles.
- `src/adapol/aaa.py` drives the adaptive AAA iteration. It builds either a standard or conjugate-constrained barycentric rational approximation and optionally removes Froissart doublets before returning a fitted object.
- `src/adapol/bra.py` holds the barycentric rational approximation classes. This is where support points, weights, residual updates, and pole/residue extraction are implemented.
- `src/adapol/sop.py` is the canonical post-AAA representation. After AAA, the code converts to `SumOfSimplePoles`, refits residues against the original frequency samples, and uses this representation for evaluation and optimization.
- `src/adapol/imtime.py` is the imaginary-time optimization layer. It builds the dyadically refined quadrature and provides the L2-norm and nonlinear optimization routines used when compressing pole sets in imaginary time.
- `src/adapol/triqs.py` and `src/adapol/triqs_xca.py` are wrappers around the same core pipeline for TRIQS Green's-function objects and DLR-based compression workflows.

## Key conventions

- Import from submodules, not from `adapol` itself. `src/adapol/__init__.py` does not re-export the package API.
- The core data shape convention is `samples x ...`: the first axis is always the frequency or quadrature sample axis, and matrix-valued targets stay in trailing axes. The implementation relies heavily on `numpy.einsum`, reshaping, and broadcasting with that layout.
- The high-level drivers in `adapol.py`, `triqs.py`, and `triqs_xca.py` all use conjugate-constrained AAA for Matsubara data, then convert to `SumOfSimplePoles` and refit residues after pole extraction.
- Naming is consistent across the pipeline: raw sample points are usually `Z`, sampled values are `F`, pole locations are `poles`/`p`, and residues are `residues`/`R`.
- Tests are split between `test/` and `benchmarks/`, and `pytest` will discover both. Some benchmark and TRIQS-oriented tests import TRIQS directly even though CI uses `-m "not triqs"`, so file-targeted pytest runs are the safest choice when TRIQS is not installed.
- The Sphinx docs still reference older modules such as `adapol.hybfit` and `adapol.anacont`; `make -C doc html` currently succeeds with warnings, so prefer the current `src/adapol/*.py` layout over stale doc references when reconciling behavior.
