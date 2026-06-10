//! Numerical tolerance policy.
//!
//! Every fuzzy comparison in this crate goes through one of the constants
//! below. The values intentionally match the Python reference implementation
//! in `adaptive.learner.triangulation` so that both backends produce
//! identical triangulations from identical input; do not change one without
//! cross-validating against the reference (see `tests/test_triangulation.py`).
//!
//! When touching any of these, first check whether the comparison is
//! *scale-invariant* (relative to the simplex being tested) or *absolute*
//! (silently assumes coordinates of order 1). Mixing the two up is how small
//! simplices far from the origin get misclassified — see the regression tests
//! added for exactly that failure in PR #3.
//!
//! # Degenerate-input policy (deliberate divergence from the reference)
//!
//! Point sets that mix widely separated coordinate scales in one
//! triangulation (e.g. a unit-sized cloud plus a 1e-5-sized cluster 100
//! away) force sliver simplices with aspect ratios around 1e7 into the mesh.
//! Floating-point circumsphere tests on such slivers are unreliable, so the
//! Bowyer-Watson cascade can assemble a cavity that is not re-triangulable
//! (it has voids or protrusions, misses the simplex containing the point,
//! or would disconnect the point). The Python reference detects part of
//! this with a volume-conservation `assert` only *after* mutating, so a
//! failing insertion corrupts its state; failures it cannot see (cavities
//! whose total volume sits below the check's absolute tolerance) silently
//! orphan vertices.
//!
//! This implementation instead validates the cavity *before* mutating
//! (volume conservation, computed with adaptive-precision determinants, and
//! point connectivity). When validation fails it repairs the cavity — in 2D
//! and 3D by rebuilding it with Shewchuk's exact insphere predicates (the
//! `robust` crate; see `geometry::robust_in_circumsphere`), in higher
//! dimensions by shrinking it until star-shaped — and re-validates. If even
//! the repaired cavity fails, the insertion is rejected (the same
//! `AssertionError`, or `ValueError` for a point that cannot be connected)
//! with the triangulation untouched, so callers can skip the point and
//! continue. Well-conditioned insertions never enter the repair path and
//! behave bit-identically to the reference.
//!
//! Homogeneous-scale inputs — at any magnitude and any offset from the
//! origin — never trigger any of this; see
//! `test_random_insertions_small_scale_far_from_origin`.

/// Relative slack on barycentric coordinates when deciding whether a point
/// lies inside a simplex, on one of its faces, or coincides with a vertex.
///
/// Scale-invariant: barycentric coordinates are already normalized by the
/// simplex size. Also exposed to Python as the default `eps` argument of
/// `point_in_simplex` and `get_reduced_simplex`.
pub const BARYCENTRIC_EPS: f64 = 1e-8;

/// Relative slack on the circumsphere radius in the Bowyer-Watson
/// in-circumcircle test: a point is "inside" when
/// `distance < radius * (1.0 + CIRCUMCIRCLE_RTOL)`.
///
/// Scale-invariant, but note it is *proportional to the neighbouring
/// simplex's size*, not the inserted point's distance to it — a point can be
/// within this slack of a large neighbour while being a meaningful fraction
/// of a small simplex away from it. Two guards in
/// `Triangulation::bowyer_watson` keep that asymmetry from corrupting the
/// triangulation: 1D insertion skips the circumcircle cascade entirely, and
/// in higher dimensions an insertion that would strand a cavity vertex is
/// rejected as a duplicate before anything is mutated.
pub const CIRCUMCIRCLE_RTOL: f64 = 1e-8;

/// Threshold below which a candidate simplex is discarded as numerically
/// degenerate during Bowyer-Watson re-triangulation. A candidate is dropped
/// only when BOTH its raw volume (absolute; guarantees the omission cannot
/// leave a material hole) and its normalized volume (volume divided by
/// characteristic length to the dim-th power, i.e. scale-invariant
/// *flatness*) fall below this constant.
///
/// The flatness condition is a deliberate divergence from the Python
/// reference, which uses the absolute test alone and therefore silently
/// deletes well-shaped simplices once a mesh is refined below ~1e-8 volume
/// (in 2D that is edge lengths of ~1e-4) — emptying whole regions of the
/// triangulation. See `Triangulation::simplex_is_numerically_degenerate`.
pub const DEGENERATE_VOLUME_EPS: f64 = 1e-8;

/// Absolute and relative tolerances of the volume-conservation check after
/// Bowyer-Watson (the sum of deleted simplex volumes must equal the sum of
/// added ones). Same constants as `numpy.isclose`, but applied symmetrically
/// (scaled by `max(|a|, |b|)` rather than `|b|`) so the check cannot depend
/// on argument order.
pub const VOLUME_CONSERVATION_ATOL: f64 = 1e-8;
/// See [`VOLUME_CONSERVATION_ATOL`].
pub const VOLUME_CONSERVATION_RTOL: f64 = 1e-5;

/// `orientation` treats a face as coplanar with the origin point when the
/// log-determinant of the difference matrix is below this cutoff
/// (`|det| < exp(-50) ~= 2e-22`).
///
/// NOT scale-invariant: the determinant scales with the dim-th power of the
/// face's edge lengths, so faces with edges below ~`exp(-50/dim)` are always
/// reported as degenerate. Matches the Python reference.
pub const ORIENTATION_LOG_DET_CUTOFF: f64 = -50.0;
