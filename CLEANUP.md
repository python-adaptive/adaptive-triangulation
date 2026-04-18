# ISSUE-095 Cleanup Summary

## Rust cleanup

- Renamed the 1D simplex filtering helpers in `src/triangulation.rs` so they describe the normalized volume check more accurately.
- Added a brief comment explaining why 1D uses a normalized check during Bowyer-Watson: tiny non-zero intervals are valid and should not be rejected as flat.
- Extracted `scipy_delaunay_simplices` to flatten `PyTriangulation::new` and keep the 1D SciPy bypass in one place.

## Test cleanup

- Added shared 1D test helpers in `tests/test_triangulation.py` for sorted-point order, expected hull vertices, triangulation-state assertions, and midpoint location checks.
- Consolidated the repeated 1D `add_point` scenarios into one parametrized test covering inside-hull insertion, outside-hull insertion, and tiny intervals.
- Reused the shared helpers in the 1D construction, hull, and incremental cross-validation tests to reduce duplicated setup and assertions.

## Verification

- `nix shell nixpkgs#cargo nixpkgs#rustc -c cargo test`
- `nix shell nixpkgs#cargo nixpkgs#rustc -c zsh -lc 'source .venv/bin/activate && maturin develop --release && python -m pytest tests/'`
