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
/// of a small simplex away from it. This is why 1D insertion skips the
/// circumcircle cascade entirely (see `Triangulation::bowyer_watson`).
pub const CIRCUMCIRCLE_RTOL: f64 = 1e-8;

/// Absolute volume below which a candidate simplex is discarded as
/// degenerate during Bowyer-Watson re-triangulation (dim >= 2).
///
/// NOT scale-invariant: this assumes coordinates of order 1, matching the
/// Python reference. In 1D, where adaptive sampling routinely produces
/// intervals far below this threshold, a normalized (scale-invariant) volume
/// is used instead — see `Triangulation::simplex_is_numerically_degenerate`.
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
