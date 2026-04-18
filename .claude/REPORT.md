# ISSUE-096 Cleanup Report

## Line counts

- `src/learner1d/mod.rs`: `833 -> 783` (`-50`)
- `src/learner1d/loss.rs`: `448 -> 428` (`-20`)
- `src/learner1d/python.rs`: `239 -> 229` (`-10`)
- `tests/test_learner1d.py`: `600 -> 399` (`-201`)
- Total: `2120 -> 1839` (`-281`)

## Specific simplifications made

- `src/learner1d/mod.rs`
  - Removed the dedicated `missing_bounds` cache and derive missing bounds directly from `bounds`, `data`, and `pending`.
  - Collapsed duplicate `vdim` detection and y-scale recomputation logic.
  - Simplified the `tell_many` rebuild path by removing repeated interval checks and trimming dead bookkeeping in `remove_unfinished`.
  - Replaced a few manual padding loops with `resize`, which made the neighborhood-building code shorter and easier to scan.

- `src/learner1d/loss.rs`
  - Removed unused `LossManager` surface (`contains`, `len`, `is_empty`, `clear`).
  - Centralized the interval lookup used by `peek_max_loss` and `iter_by_priority`.
  - Simplified curvature loss to reuse the existing interval slices instead of constructing temporary arrays.

- `src/learner1d/python.rs`
  - Removed unused `Python` handles from `tell` and `tell_many`.
  - Collapsed the `run` evaluation loop into a single `collect::<PyResult<_>>()`.
  - Simplified vector `to_numpy` conversion by building rows directly instead of flattening and rechunking.

- `tests/test_learner1d.py`
  - Replaced repeated learner setup with small helpers.
  - Merged the obvious near-duplicates into parametrized tests.
  - Kept the focused suite at `56` collected cases while cutting `201` lines.

- Verification plumbing
  - Moved `pyo3/extension-module` out of the default Cargo dependency features so `cargo test --release` links a runnable test binary.
  - Added a tiny `build.rs` rpath hook so the Rust test binary can find `libpython` in this workspace without changing the `maturin` build path.

## Considered cutting, but didn't

- Replacing the explicit `LossFunction::Clone` impl with `#[derive(Clone)]`.
  - Not viable with the current PyO3 version here because `PyObject` does not implement `Clone`.

- Pushing scalar/vector handling behind a larger `YValue` helper API.
  - It would have spread more abstraction across the core learner code and made the files longer.

- Removing the small Rust unit tests in `src/learner1d/mod.rs`.
  - They are already tiny and still provide direct coverage for the simplest ask/linspace behavior.

## Verification

- `uv tool run maturin develop --release`
- `uv run pytest tests/test_learner1d.py -x --tb=short` -> `56 passed`
- `cargo test --release` -> `4 passed`, `0 failed`
