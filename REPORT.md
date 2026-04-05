# ISSUE-095 REPORT

## What Changed

- Enabled 1D triangulation on the Rust side by removing the `dim == 1` rejection in `Triangulation::validate_coords`.
- Updated `PyTriangulation::__new__` to bypass SciPy for 1D inputs and construct the triangulation directly with `Triangulation::new`.
- Fixed `simplex_volume_in_embedding` so 2-point simplices return Euclidean segment length in any embedding dimension, while identical endpoints still raise `DegenerateSimplex`.
- Replaced the hard Bowyer-Watson flatness cutoff for 1D with a scale-aware relative-volume check, while keeping the existing absolute cutoff for `dim >= 2`.
- Added Rust unit tests and Python tests covering 1D construction, geometry helpers, faces/hull, inside/outside insertion, circumsphere behavior, duplicate rejection, random 1D validation, and tiny intervals.

## Test Results

- `cargo test`: passed (7 tests).
- `maturin develop --release`: passed.
- `python -m pytest tests/`: passed (50 tests).

## Notes

- `cargo` and `maturin` were not on the default `PATH` in this environment, so the build/test commands were run through `nix shell`.
- `pytest` reported 2 existing `RuntimeWarning`s from the upstream `adaptive` reference implementation in the degenerate 4D circumsphere test; the test suite still passed cleanly.
