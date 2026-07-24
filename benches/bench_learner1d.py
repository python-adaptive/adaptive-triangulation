"""Benchmarks for the Rust-powered Learner1D."""

from __future__ import annotations

import math
import time

from adaptive_triangulation import Learner1D


def bench(name: str, fn, *, n_iter: int = 10):
    """Run a benchmark and print results."""
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    median = times[len(times) // 2]
    print(f"  {name}: {median * 1000:.2f} ms (median of {n_iter})")
    return median


def bench_tell_single():
    print("=== tell_single ===")
    for n in [1_000, 10_000, 100_000]:
        step = 2.0 / n

        def run(n=n, step=step):
            l = Learner1D(bounds=(-1.0, 1.0))
            for i in range(n):
                x = -1.0 + step * i
                l.tell(x, math.sin(x * 10))

        bench(f"tell {n:,} points", run, n_iter=5 if n >= 100_000 else 10)


def bench_tell_many_batch():
    print("\n=== tell_many (force rebuild) ===")
    for n in [1_000, 10_000]:
        step = 2.0 / n
        xs = [-1.0 + step * i for i in range(n)]
        ys = [math.sin(x * 10) for x in xs]

        def run(xs=xs, ys=ys):
            l = Learner1D(bounds=(-1.0, 1.0))
            l.tell_many(xs, ys, force=True)

        bench(f"tell_many {n:,} points", run)


def bench_ask():
    print("\n=== ask 100 points ===")
    for n_existing in [100, 1_000, 10_000]:
        step = 2.0 / n_existing
        xs = [-1.0 + step * i for i in range(n_existing)]
        ys = [math.sin(x * 10) for x in xs]

        def run(xs=xs, ys=ys):
            l = Learner1D(bounds=(-1.0, 1.0))
            l.tell_many(xs, ys, force=True)
            l.ask(100, tell_pending=False)

        bench(f"ask 100 (from {n_existing:,} pts)", run)


def bench_full_loop():
    print("\n=== full loop 10K points ===")
    f = lambda x: math.sin(x * 10)

    def run_serial():
        l = Learner1D(bounds=(-1.0, 1.0))
        l.run(f, n_points=10_000, batch_size=1)

    def run_batched():
        l = Learner1D(bounds=(-1.0, 1.0))
        l.run(f, n_points=10_000, batch_size=100)

    bench("serial (batch=1)", run_serial, n_iter=3)
    bench("batched (batch=100)", run_batched, n_iter=3)


if __name__ == "__main__":
    print("Learner1D Benchmarks")
    print("=" * 50)
    bench_tell_single()
    bench_tell_many_batch()
    bench_ask()
    bench_full_loop()
