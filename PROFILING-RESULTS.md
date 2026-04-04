# Adaptive Triangulation Profiling Results

Benchmark command used for every step:

```bash
maturin develop --release 2>&1 | tail -1
python -u -c "
import time, numpy as np
import adaptive_triangulation as at
rng = np.random.default_rng(42)
for dim, n in [(2, 1000), (2, 5000), (3, 500)]:
    pts = rng.random((n, dim))
    init = pts[:dim+1]; add = pts[dim+1:]
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        tr = at.Triangulation(init.tolist())
        for p in add:
            try: tr.add_point(tuple(p))
            except: pass
        times.append(time.perf_counter() - t0)
    print(f'{dim}D {n}: {min(times)*1000:.1f}ms')
"
```

Baseline before any optimization:

| Case | Time |
| --- | ---: |
| 2D 1000 | 73.9 ms |
| 2D 5000 | 1165.3 ms |
| 3D 500 | 402.8 ms |

## Accepted

### 1. `FxHashMap` / `FxHashSet`

| Case | Before | After | Delta |
| --- | ---: | ---: | ---: |
| 2D 1000 | 73.9 ms | 63.3 ms | 14.3% faster |
| 2D 5000 | 1165.3 ms | 1111.5 ms | 4.6% faster |
| 3D 500 | 402.8 ms | 343.6 ms | 14.7% faster |

Outcome: kept. Geometric-mean speedup across the three cases was 12.7%.

### 2. Simplex walking for point location

| Case | Before | After | Delta |
| --- | ---: | ---: | ---: |
| 2D 1000 | 63.3 ms | 39.0 ms | 38.4% faster |
| 2D 5000 | 1111.5 ms | 279.2 ms | 74.9% faster |
| 3D 500 | 343.6 ms | 132.3 ms | 61.5% faster |

Outcome: kept. Geometric-mean speedup across the three cases was 58.3%.

## Rejected

### 3. Reuse `bowyer_watson` BFS scratch buffers

| Case | Before | After | Delta |
| --- | ---: | ---: | ---: |
| 2D 1000 | 39.0 ms | 39.4 ms | 1.0% slower |
| 2D 5000 | 279.2 ms | 262.6 ms | 5.9% faster |
| 3D 500 | 132.3 ms | 124.9 ms | 5.6% faster |

Outcome: reverted. Improvement was below the 10% threshold.

### 4. Extra 2D/3D no-transform fast paths in `point_in_circumcircle` / `volume`

| Case | Before | After | Delta |
| --- | ---: | ---: | ---: |
| 2D 1000 | 39.0 ms | 57.6 ms | 47.7% slower |
| 2D 5000 | 279.2 ms | 339.5 ms | 21.6% slower |
| 3D 500 | 132.3 ms | 171.2 ms | 29.4% slower |

Outcome: reverted. This was a clear regression.

## Final state

Current branch state includes the two accepted optimizations only.

| Case | Baseline | Final | Total delta |
| --- | ---: | ---: | ---: |
| 2D 1000 | 73.9 ms | 39.0 ms | 47.2% faster |
| 2D 5000 | 1165.3 ms | 279.2 ms | 76.0% faster |
| 3D 500 | 402.8 ms | 132.3 ms | 67.2% faster |

The flat vertex-storage refactor was not kept because the remaining low-complexity experiments after the algorithmic win were already below threshold or regressive, and the branch goal was to avoid adding complexity for marginal gains.
