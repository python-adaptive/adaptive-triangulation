use std::hint::black_box;
use std::time::{Duration, Instant};

use adaptive_triangulation::geometry;
use adaptive_triangulation::triangulation::{Simplex, Triangulation};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn next_unit_f64(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 11) as f64) / ((1_u64 << 53) as f64)
}

fn generate_points(dim: usize, count: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut state = seed;
    let mut points = Vec::with_capacity(count);
    for index in 0..count {
        let mut point = Vec::with_capacity(dim);
        for axis in 0..dim {
            let low_discrepancy =
                ((index + 1) as f64 * (axis + 2) as f64 * 0.618_033_988_749_894_9).fract();
            let jitter = next_unit_f64(&mut state);
            point.push(
                0.7 * low_discrepancy + 0.3 * jitter + axis as f64 * 1e-4 + index as f64 * 1e-12,
            );
        }
        points.push(point);
    }
    points
}

fn first_simplex(triangulation: &Triangulation) -> Simplex {
    triangulation
        .simplices
        .iter()
        .next()
        .cloned()
        .expect("triangulation should not be empty")
}

fn simplex_centroid(triangulation: &Triangulation, simplex: &[usize]) -> Vec<f64> {
    let mut center = vec![0.0; triangulation.dim];
    for &vertex in simplex {
        for (coord, value) in center.iter_mut().zip(&triangulation.vertices[vertex]) {
            *coord += *value;
        }
    }
    let scale = 1.0 / simplex.len() as f64;
    for coord in &mut center {
        *coord *= scale;
    }
    center
}

fn outside_point(triangulation: &Triangulation) -> Vec<f64> {
    let mut min = vec![f64::INFINITY; triangulation.dim];
    let mut max = vec![f64::NEG_INFINITY; triangulation.dim];
    for point in &triangulation.vertices {
        for axis in 0..triangulation.dim {
            min[axis] = min[axis].min(point[axis]);
            max[axis] = max[axis].max(point[axis]);
        }
    }

    let mut point = vec![0.0; triangulation.dim];
    point[0] = max[0] + (max[0] - min[0]).abs() + 0.5;
    for axis in 1..triangulation.dim {
        point[axis] = (min[axis] + max[axis]) * 0.5;
    }
    point
}

fn insert_pending_vertex(triangulation: &mut Triangulation, point: Vec<f64>) -> usize {
    triangulation.vertex_to_simplices.push(Default::default());
    triangulation.vertices.push(point);
    triangulation.vertices.len() - 1
}

fn constructor_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("constructor");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(8));

    for &(dim, count, seed) in &[
        (2, 100, 0xA11CE),
        (2, 500, 0xA11CF),
        (2, 1000, 0xA11D0),
        (3, 100, 0xBEEF1),
        (3, 500, 0xBEEF2),
        (3, 1000, 0xBEEF3),
    ] {
        let points = generate_points(dim, count, seed);
        group.bench_with_input(
            BenchmarkId::new(format!("{dim}d"), count),
            &points,
            |b, points| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let coords = points.clone();
                        let start = Instant::now();
                        let triangulation = Triangulation::new(coords).unwrap();
                        black_box(triangulation.simplices.len());
                        total += start.elapsed();
                    }
                    total
                });
            },
        );
    }

    group.finish();
}

fn add_point_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("add_point");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    let points = generate_points(2, 5000, 0xDEADBEEF);
    let seed = points[..32].to_vec();
    let to_insert = points[32..].to_vec();
    group.bench_function("2d_incremental_5000", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let mut triangulation = Triangulation::new(seed.clone()).unwrap();
                let start = Instant::now();
                for point in &to_insert {
                    triangulation.add_point(point.clone(), None, None).unwrap();
                }
                black_box(triangulation.simplices.len());
                total += start.elapsed();
            }
            total
        });
    });

    group.finish();
}

fn circumsphere_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("circumsphere");
    let simplex_2d = vec![vec![0.0, 0.0], vec![0.9, 0.1], vec![0.2, 0.8]];
    let simplex_3d = vec![
        vec![0.0, 0.0, 0.0],
        vec![1.0, 0.1, 0.0],
        vec![0.0, 1.1, 0.1],
        vec![0.1, 0.0, 0.9],
    ];

    group.bench_function("2d", |b| {
        b.iter(|| black_box(geometry::circumsphere(black_box(&simplex_2d)).unwrap()))
    });
    group.bench_function("3d", |b| {
        b.iter(|| black_box(geometry::circumsphere(black_box(&simplex_3d)).unwrap()))
    });
    group.finish();
}

fn point_in_circumcircle_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("point_in_circumcircle");
    let triangulation = Triangulation::new(generate_points(2, 256, 0xCAFE)).unwrap();
    let simplex = first_simplex(&triangulation);
    let point = simplex_centroid(&triangulation, &simplex);

    group.bench_function("2d", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let mut current = triangulation.clone();
                let pt_index = insert_pending_vertex(&mut current, point.clone());
                let start = Instant::now();
                let inside = current
                    .point_in_circumcircle(pt_index, &simplex, &None)
                    .unwrap();
                black_box(inside);
                total += start.elapsed();
            }
            total
        });
    });

    group.finish();
}

fn locate_and_containing_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("point_location");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(8));

    let triangulation = Triangulation::new(generate_points(2, 2048, 0xFACE)).unwrap();
    let simplices: Vec<Simplex> = triangulation.simplices.iter().take(256).cloned().collect();
    let queries: Vec<Vec<f64>> = simplices
        .iter()
        .map(|simplex| simplex_centroid(&triangulation, simplex))
        .collect();
    let faces: Vec<Simplex> = simplices
        .iter()
        .map(|simplex| simplex[..triangulation.dim].to_vec())
        .collect();

    group.bench_function("locate_point", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for index in 0..iters {
                let query = &queries[index as usize % queries.len()];
                let start = Instant::now();
                let simplex = triangulation.locate_point(query).unwrap();
                black_box(simplex);
                total += start.elapsed();
            }
            total
        });
    });

    group.bench_function("containing", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for index in 0..iters {
                let face = &faces[index as usize % faces.len()];
                let start = Instant::now();
                let containing = triangulation.containing(face).unwrap();
                black_box(containing);
                total += start.elapsed();
            }
            total
        });
    });

    group.finish();
}

fn bowyer_watson_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("bowyer_watson");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(8));

    let base = Triangulation::new(generate_points(2, 512, 0x1234_5678)).unwrap();
    let simplex = first_simplex(&base);
    let point = simplex_centroid(&base, &simplex);

    group.bench_function("2d_cavity", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let mut triangulation = base.clone();
                let pt_index = insert_pending_vertex(&mut triangulation, point.clone());
                let start = Instant::now();
                let result = triangulation
                    .bowyer_watson(pt_index, Some(simplex.clone()), &None)
                    .unwrap();
                black_box(result);
                total += start.elapsed();
            }
            total
        });
    });

    group.finish();
}

fn extend_hull_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("extend_hull");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(8));

    let base = Triangulation::new(generate_points(2, 512, 0xFEED_BEEF)).unwrap();
    let point = outside_point(&base);

    group.bench_function("2d", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let mut triangulation = base.clone();
                let pt_index = insert_pending_vertex(&mut triangulation, point.clone());
                let start = Instant::now();
                let result = triangulation.extend_hull(pt_index).unwrap();
                black_box(result);
                total += start.elapsed();
            }
            total
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    constructor_benches,
    add_point_bench,
    circumsphere_benches,
    point_in_circumcircle_bench,
    locate_and_containing_benches,
    bowyer_watson_bench,
    extend_hull_bench
);
criterion_main!(benches);
