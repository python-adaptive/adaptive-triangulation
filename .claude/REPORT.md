# Code Review Fix Report: Rust Learner1D Implementation

**Branch:** `issue-096-impl`
**Status:** All review findings addressed, all tests passing (56/56)

---

## Fixes Applied

### Fix #1 (CRITICAL) — `python_callback_loss` panic on exceptions
**File:** `src/learner1d/loss.rs:282-289`

Replaced `.expect()` calls with `match` + `unwrap_or_else` error handling. When a Python
callback raises an exception or returns a non-float, the error is printed to stderr via
`err.print(py)` and `f64::INFINITY` is returned as a fallback loss. This prevents process
crashes and preserves the Python traceback for debugging.

### Fix #2 (HIGH) — Out-of-bounds data polluting neighbor queries
**File:** `src/learner1d/mod.rs`

Added `out_of_bounds_data: HashMap<OF64, YValue>` field to `Learner1D`. In `tell()` and
`tell_many()`, points outside bounds are stored in `out_of_bounds_data` instead of `data`
(BTreeMap). This prevents OOB points from appearing in `find_real_neighbors()` queries,
which would have created incorrect loss intervals extending beyond bounds. The `data`
Python property and `npoints()` include OOB data for API compatibility.

### Fix #3 (HIGH) — `x_scale` stale after `tell_many(force=True)` rebuild
**File:** `src/learner1d/mod.rs`

Changed `rebuild_scale()` to always set `x_scale = bounds.1 - bounds.0` (the constant
domain width) instead of computing it from the span of combined points. This matches the
Python behavior and ensures `x_scale` is never stale after a rebuild followed by
incremental tells.

### Fix #4 (MEDIUM) — `loss` property doesn't expose `real` parameter
**File:** `src/learner1d/python.rs:147-149`

Changed `loss` from a `#[getter]` property to a method with
`#[pyo3(signature = (real=true))]`. Users can now call `loss(real=False)` to get the
combined loss accounting for pending points. Updated all test references from `l.loss` to
`l.loss()`.

### Fix #5 (MEDIUM) — Empty learner `ask(n>2)` linspace missing right endpoint
**File:** `src/learner1d/mod.rs`

Changed the linspace formula from `a + step * i` (which misses the right endpoint) to
`a + (b - a) * i / (n - 1)` which matches `np.linspace(a, b, n)` and includes both
endpoints.

### Fix #6 (MEDIUM) — Ask tiebreaking differs from Python
**File:** `src/learner1d/mod.rs`

Updated the `prefer_combined` comparison in `ask()` to use lexicographic `(loss, interval)`
comparison when finite losses are equal, matching Python's tuple comparison behavior.

### Fix #7 (LOW) — `has_missing_bounds` allocates Vec unnecessarily
**File:** `src/learner1d/mod.rs`

Replaced `self.get_missing_bounds_list().iter().any(|_| true)` with a direct iterator
check on `self.missing_bounds` using `.any()`, eliminating the Vec allocation and sort.

### Fix #8 (LOW) — `rebuild_scale` clones all data values
**File:** `src/learner1d/mod.rs`

Extracted y-bounds tracking into a free function `update_y_bounds()` that takes mutable
references to `y_min`/`y_max` vectors. This allows `rebuild_scale()` to iterate
`self.data.values()` by reference without cloning, resolving the borrow checker issue
that originally required the clone.

---

## Test Results

```
56 passed in 0.15s
```

### New Tests Added (12)
- `TestOutOfBounds` (5 tests): OOB tells don't affect loss, are stored in data, counted
  in npoints, don't appear as neighbors, duplicates ignored
- `TestCallbackException` (2 tests): Exception returns infinity, wrong return type
  returns infinity
- `TestLossRealParam` (3 tests): `loss(real=True)`, `loss(real=False)` with pending,
  default is real
- `TestLinspaceEndpoint` (2 tests): Empty ask includes right endpoint for n=3 and n=5
