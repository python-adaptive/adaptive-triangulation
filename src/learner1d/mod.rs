pub mod loss;
pub mod python;

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ops::Bound;

use ordered_float::OrderedFloat;

use self::loss::{
    compute_loss, finite_loss_value, finite_loss_with_n, LossFunction, LossManager,
};

pub type OF64 = OrderedFloat<f64>;

/// Represents a function output value (scalar or vector).
#[derive(Clone, Debug)]
pub enum YValue {
    Scalar(f64),
    Vector(Vec<f64>),
}

/// Adaptive 1D learner backed by `BTreeMap` internals.
pub struct Learner1D {
    // Core data
    pub(crate) data: BTreeMap<OF64, YValue>,
    /// Data points outside bounds — stored separately so they don't affect neighbor queries.
    pub(crate) out_of_bounds_data: HashMap<OF64, YValue>,
    pub(crate) pending: BTreeSet<OF64>,
    /// Union of `data` keys and `pending`.
    combined_points: BTreeSet<OF64>,

    // Loss tracking
    pub(crate) losses: LossManager,
    pub(crate) losses_combined: LossManager,

    // Scaling
    pub(crate) bounds: (f64, f64),
    x_scale: f64,
    y_scale: f64,
    old_y_scale: f64,
    y_min: Vec<f64>,
    y_max: Vec<f64>,

    // Config
    pub(crate) loss_fn: LossFunction,
    nth_neighbors: usize,
    pub(crate) vdim: Option<usize>,
    dx_eps: f64,
}

// ---- BTreeMap / BTreeSet neighbour helpers ----

fn predecessor_map<V>(map: &BTreeMap<OF64, V>, x: OF64) -> Option<OF64> {
    map.range(..x).next_back().map(|(&k, _)| k)
}

fn successor_map<V>(map: &BTreeMap<OF64, V>, x: OF64) -> Option<OF64> {
    map.range((Bound::Excluded(x), Bound::Unbounded))
        .next()
        .map(|(&k, _)| k)
}

fn predecessor_set(set: &BTreeSet<OF64>, x: OF64) -> Option<OF64> {
    set.range(..x).next_back().copied()
}

fn successor_set(set: &BTreeSet<OF64>, x: OF64) -> Option<OF64> {
    set.range((Bound::Excluded(x), Bound::Unbounded))
        .next()
        .copied()
}

/// Return n-1 interior points evenly spaced inside `(left, right)`.
fn linspace_interior(left: f64, right: f64, n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![];
    }
    let step = (right - left) / n as f64;
    (1..n).map(|i| left + step * i as f64).collect()
}

/// Update y-min/y-max bounds from a single data point.
fn update_y_bounds(y_min: &mut Vec<f64>, y_max: &mut Vec<f64>, y: &YValue) {
    match y {
        YValue::Scalar(v) => {
            if y_min.is_empty() {
                *y_min = vec![*v];
                *y_max = vec![*v];
            } else {
                y_min[0] = y_min[0].min(*v);
                y_max[0] = y_max[0].max(*v);
            }
        }
        YValue::Vector(v) => {
            if y_min.is_empty() {
                *y_min = v.clone();
                *y_max = v.clone();
            } else {
                for (i, val) in v.iter().enumerate() {
                    if i < y_min.len() {
                        y_min[i] = y_min[i].min(*val);
                        y_max[i] = y_max[i].max(*val);
                    }
                }
            }
        }
    }
}

impl Learner1D {
    // ----------------------------------------------------------------
    //  Construction
    // ----------------------------------------------------------------

    pub fn new(bounds: (f64, f64), loss_fn: LossFunction) -> Self {
        let x_scale = bounds.1 - bounds.0;
        let nth_neighbors = loss_fn.nth_neighbors();
        let dx_eps = 2.0 * f64::max(bounds.0.abs(), bounds.1.abs()) * f64::EPSILON;
        Self {
            data: BTreeMap::new(),
            out_of_bounds_data: HashMap::new(),
            pending: BTreeSet::new(),
            combined_points: BTreeSet::new(),
            losses: LossManager::new(x_scale),
            losses_combined: LossManager::new(x_scale),
            bounds,
            x_scale,
            y_scale: 0.0,
            old_y_scale: 0.0,
            y_min: Vec::new(),
            y_max: Vec::new(),
            loss_fn,
            nth_neighbors,
            vdim: None,
            dx_eps,
        }
    }

    // ----------------------------------------------------------------
    //  Public API
    // ----------------------------------------------------------------

    pub fn tell(&mut self, x: f64, y: YValue) {
        let xo = OF64::from(x);
        if self.data.contains_key(&xo) || self.out_of_bounds_data.contains_key(&xo) {
            return;
        }

        self.set_vdim_if_unknown(&y);

        self.pending.remove(&xo);

        if !(self.bounds.0 <= x && x <= self.bounds.1) {
            self.out_of_bounds_data.insert(xo, y);
            return;
        }

        self.data.insert(xo, y.clone());
        self.combined_points.insert(xo);

        self.update_scale(&y);
        self.update_losses(xo, true);

        let should_recompute = (self.old_y_scale == 0.0 && self.y_scale > 0.0)
            || (self.old_y_scale > 0.0 && self.y_scale > 2.0 * self.old_y_scale);
        if should_recompute {
            let ivals: Vec<(OF64, OF64)> = self.losses.all_intervals();
            for (l, r) in ivals {
                self.update_interpolated_loss_in_interval(l, r);
            }
            self.old_y_scale = self.y_scale;
        }
    }

    pub fn tell_pending(&mut self, x: f64) {
        let xo = OF64::from(x);
        if self.data.contains_key(&xo) {
            return;
        }
        self.pending.insert(xo);
        self.combined_points.insert(xo);
        self.update_losses(xo, false);
    }

    pub fn tell_many(&mut self, xs: &[f64], ys: &[YValue], force: bool) {
        if xs.is_empty() {
            return;
        }
        if !force && !(xs.len() > 2 && xs.len() as f64 > 0.5 * self.data.len() as f64) {
            // Incremental path
            for (x, y) in xs.iter().zip(ys.iter()) {
                self.tell(*x, y.clone());
            }
            return;
        }
        // Fast rebuild path
        for (x, y) in xs.iter().zip(ys.iter()) {
            let xo = OF64::from(*x);
            self.pending.remove(&xo);
            self.set_vdim_if_unknown(y);
            if !(self.bounds.0 <= *x && *x <= self.bounds.1) {
                self.out_of_bounds_data.insert(xo, y.clone());
            } else {
                self.data.insert(xo, y.clone());
            }
        }

        // Rebuild combined_points
        self.combined_points = self
            .data
            .keys()
            .copied()
            .chain(self.pending.iter().copied())
            .collect();

        // Rebuild scale from all data
        self.rebuild_scale();

        // Rebuild losses
        let real_pts: Vec<OF64> = self.data.keys().copied().collect();
        let comb_pts: Vec<OF64> = self.combined_points.iter().copied().collect();
        let intervals: Vec<(OF64, OF64)> =
            real_pts.windows(2).map(|w| (w[0], w[1])).collect();
        let intervals_combined: Vec<(OF64, OF64)> =
            comb_pts.windows(2).map(|w| (w[0], w[1])).collect();

        self.losses = LossManager::new(self.x_scale);
        for &(l, r) in &intervals {
            let loss = self.get_loss_in_interval(l, r);
            self.losses.insert(l, r, loss);
        }

        self.losses_combined = LossManager::new(self.x_scale);
        let mut to_interpolate: Vec<(OF64, OF64)> = Vec::new();
        for &(l, r) in &intervals_combined {
            match self.losses.get(l, r) {
                Some(loss) => self.losses_combined.insert(l, r, loss),
                None => {
                    self.losses_combined.insert(l, r, f64::INFINITY);
                    if matches!(to_interpolate.last(), Some((_, last_right)) if *last_right == l) {
                        to_interpolate.last_mut().unwrap().1 = r;
                    } else {
                        to_interpolate.push((l, r));
                    }
                }
            }
        }
        for (l, r) in to_interpolate {
            if self.losses.get(l, r).is_some() {
                self.update_interpolated_loss_in_interval(l, r);
            }
        }
    }

    /// Return `n` points that are expected to maximally reduce the loss,
    /// plus the corresponding loss improvements.
    pub fn ask(&mut self, n: usize, do_tell_pending: bool) -> (Vec<f64>, Vec<f64>) {
        let (points, improvements) = self.ask_points_without_adding(n);
        if do_tell_pending {
            for &p in &points {
                self.tell_pending(p);
            }
        }
        (points, improvements)
    }

    pub fn loss(&self, real: bool) -> f64 {
        if self.is_missing_bound(self.bounds.0) || self.is_missing_bound(self.bounds.1) {
            return f64::INFINITY;
        }
        let mgr = if real {
            &self.losses
        } else {
            &self.losses_combined
        };
        mgr.peek_max_loss().unwrap_or(f64::INFINITY)
    }

    pub fn npoints(&self) -> usize {
        self.data.len() + self.out_of_bounds_data.len()
    }

    pub fn remove_unfinished(&mut self) {
        self.pending.clear();
        self.losses_combined = self.losses.clone();
        self.combined_points = self.data.keys().copied().collect();
    }

    /// Sorted `(x, y)` data.
    pub fn to_sorted_data(&self) -> Vec<(f64, YValue)> {
        self.data
            .iter()
            .map(|(&x, y)| (x.into_inner(), y.clone()))
            .collect()
    }

    /// `(x_left, x_right, loss)` for all real intervals.
    pub fn intervals_with_loss(&self) -> Vec<(f64, f64, f64)> {
        let pts: Vec<OF64> = self.data.keys().copied().collect();
        pts.windows(2)
            .map(|w| {
                let l = w[0];
                let r = w[1];
                let loss = self.losses.get(l, r).unwrap_or(0.0);
                (l.into_inner(), r.into_inner(), loss)
            })
            .collect()
    }

    // ----------------------------------------------------------------
    //  Internal: scale management
    // ----------------------------------------------------------------

    fn set_vdim_if_unknown(&mut self, y: &YValue) {
        if self.vdim.is_none() {
            self.vdim = Some(match y {
                YValue::Scalar(_) => 1,
                YValue::Vector(values) => values.len(),
            });
        }
    }

    fn refresh_y_scale(&mut self) {
        self.y_scale = self
            .y_min
            .iter()
            .zip(self.y_max.iter())
            .map(|(lo, hi)| hi - lo)
            .fold(0.0_f64, f64::max);
    }

    fn update_scale(&mut self, y: &YValue) {
        update_y_bounds(&mut self.y_min, &mut self.y_max, y);
        self.refresh_y_scale();
    }

    fn rebuild_scale(&mut self) {
        self.y_min.clear();
        self.y_max.clear();
        for y in self.data.values() {
            update_y_bounds(&mut self.y_min, &mut self.y_max, y);
        }
        self.refresh_y_scale();
        // x_scale is always the domain width (matches Python where x_scale = bounds[1] - bounds[0])
        self.x_scale = self.bounds.1 - self.bounds.0;
        self.old_y_scale = self.y_scale;
    }

    fn scale_y(&self, y: &YValue) -> YValue {
        let ys = if self.y_scale != 0.0 {
            self.y_scale
        } else {
            1.0
        };
        match y {
            YValue::Scalar(v) => YValue::Scalar(v / ys),
            YValue::Vector(v) => YValue::Vector(v.iter().map(|x| x / ys).collect()),
        }
    }

    // ----------------------------------------------------------------
    //  Internal: neighbour queries
    // ----------------------------------------------------------------

    fn find_real_neighbors(&self, x: OF64) -> (Option<OF64>, Option<OF64>) {
        (predecessor_map(&self.data, x), successor_map(&self.data, x))
    }

    fn find_combined_neighbors(&self, x: OF64) -> (Option<OF64>, Option<OF64>) {
        (
            predecessor_set(&self.combined_points, x),
            successor_set(&self.combined_points, x),
        )
    }

    /// Intervals affected by adding `x` to the real set.
    /// With `nth_neighbors = 0`: `[(x_left, x), (x, x_right)]`.
    /// With `nth_neighbors = 1`: extends one further on each side.
    fn get_affected_intervals(&self, x: OF64) -> Vec<(OF64, OF64)> {
        let nn = self.nth_neighbors;
        let mut pts: Vec<OF64> = Vec::new();

        // Collect up to nn+1 predecessors (including x_left, x_{left-1}, …)
        let mut cur = x;
        let mut before = Vec::new();
        for _ in 0..(nn + 1) {
            if let Some(prev) = predecessor_map(&self.data, cur) {
                before.push(prev);
                cur = prev;
            } else {
                break;
            }
        }
        before.reverse();
        pts.extend(before);

        pts.push(x);

        // Collect up to nn+1 successors
        cur = x;
        for _ in 0..(nn + 1) {
            if let Some(next) = successor_map(&self.data, cur) {
                pts.push(next);
                cur = next;
            } else {
                break;
            }
        }

        pts.windows(2).map(|w| (w[0], w[1])).collect()
    }

    /// Gather the neighbourhood points (with optional None padding)
    /// for loss computation on interval `[x_left, x_right]`.
    fn get_neighborhood(
        &self,
        x_left: OF64,
        x_right: OF64,
    ) -> (Vec<Option<f64>>, Vec<Option<&YValue>>) {
        let nn = self.nth_neighbors;
        let mut xs: Vec<Option<f64>> = Vec::with_capacity(2 + 2 * nn);
        let mut ys: Vec<Option<&YValue>> = Vec::with_capacity(2 + 2 * nn);

        // Predecessors of x_left
        let mut before_x: Vec<Option<OF64>> = Vec::new();
        let mut cur = x_left;
        for _ in 0..nn {
            let prev = predecessor_map(&self.data, cur);
            before_x.push(prev);
            if let Some(p) = prev {
                cur = p;
            } else {
                break;
            }
        }
        before_x.resize(nn, None);
        before_x.reverse();

        for p in &before_x {
            xs.push(p.map(|v| v.into_inner()));
            ys.push(p.and_then(|v| self.data.get(&v)));
        }

        // The interval endpoints
        xs.push(Some(x_left.into_inner()));
        ys.push(self.data.get(&x_left));
        xs.push(Some(x_right.into_inner()));
        ys.push(self.data.get(&x_right));

        // Successors of x_right
        cur = x_right;
        for _ in 0..nn {
            let next = successor_map(&self.data, cur);
            xs.push(next.map(|v| v.into_inner()));
            ys.push(next.and_then(|v| self.data.get(&v)));
            if let Some(n) = next {
                cur = n;
            } else {
                break;
            }
        }
        xs.resize(2 + 2 * nn, None);
        ys.resize(2 + 2 * nn, None);

        (xs, ys)
    }

    // ----------------------------------------------------------------
    //  Internal: loss computation
    // ----------------------------------------------------------------

    fn get_loss_in_interval(&self, x_left: OF64, x_right: OF64) -> f64 {
        let dx = x_right.into_inner() - x_left.into_inner();
        if dx < self.dx_eps {
            return 0.0;
        }
        let (xs_raw, ys_raw) = self.get_neighborhood(x_left, x_right);
        let xs_sc: Vec<Option<f64>> = xs_raw
            .iter()
            .map(|x| x.map(|v| v / self.x_scale))
            .collect();
        let ys_scaled: Vec<Option<YValue>> =
            ys_raw.iter().map(|y| y.map(|v| self.scale_y(v))).collect();
        let ys_refs: Vec<Option<&YValue>> = ys_scaled.iter().map(|y| y.as_ref()).collect();
        compute_loss(&self.loss_fn, &xs_sc, &ys_refs)
    }

    fn update_interpolated_loss_in_interval(&mut self, x_left: OF64, x_right: OF64) {
        let loss = self.get_loss_in_interval(x_left, x_right);
        self.losses.insert(x_left, x_right, loss);

        // Walk combined points between x_left and x_right, setting
        // interpolated losses proportional to sub-interval width.
        let dx = (x_right - x_left).into_inner();
        if dx == 0.0 {
            return;
        }
        let mut a = x_left;
        while a != x_right {
            if let Some(b) = successor_set(&self.combined_points, a) {
                if b > x_right {
                    break;
                }
                let sub_loss = (b - a).into_inner() * loss / dx;
                self.losses_combined.insert(a, b, sub_loss);
                a = b;
            } else {
                break;
            }
        }
    }

    fn update_losses(&mut self, x: OF64, real: bool) {
        let (x_left, x_right) = self.find_real_neighbors(x);
        let (a, b) = self.find_combined_neighbors(x);

        // Remove the old combined interval that x now splits.
        if let (Some(a_val), Some(b_val)) = (a, b) {
            self.losses_combined.remove(a_val, b_val);
        }

        if real {
            // Recompute losses for all affected intervals
            let affected = self.get_affected_intervals(x);
            for (il, ir) in affected {
                self.update_interpolated_loss_in_interval(il, ir);
            }
            // Remove the old real interval
            if let (Some(xl), Some(xr)) = (x_left, x_right) {
                self.losses.remove(xl, xr);
                self.losses_combined.remove(xl, xr);
            }
        } else if let (Some(xl), Some(xr)) = (x_left, x_right) {
            // Interpolate from the real interval
            if let Some(loss) = self.losses.get(xl, xr) {
                let dx = (xr - xl).into_inner();
                if let Some(a_val) = a {
                    let sub = (x - a_val).into_inner() * loss / dx;
                    self.losses_combined.insert(a_val, x, sub);
                }
                if let Some(b_val) = b {
                    let sub = (b_val - x).into_inner() * loss / dx;
                    self.losses_combined.insert(x, b_val, sub);
                }
            }
        }

        // Handle unknown-loss edges (pending point with no real neighbour on a side).
        let left_unknown =
            x_left.is_none() || (!real && x_right.is_none());
        let right_unknown =
            x_right.is_none() || (!real && x_left.is_none());

        if let Some(a_val) = a {
            if left_unknown {
                self.losses_combined.insert(a_val, x, f64::INFINITY);
            }
        }
        if let Some(b_val) = b {
            if right_unknown {
                self.losses_combined.insert(x, b_val, f64::INFINITY);
            }
        }
    }

    // ----------------------------------------------------------------
    //  Internal: ask algorithm
    // ----------------------------------------------------------------

    fn is_missing_bound(&self, bound: f64) -> bool {
        let bound = OF64::from(bound);
        !self.data.contains_key(&bound) && !self.pending.contains(&bound)
    }

    fn missing_bounds(&self) -> Vec<f64> {
        [self.bounds.0, self.bounds.1]
            .into_iter()
            .filter(|&bound| self.is_missing_bound(bound))
            .collect()
    }

    fn ask_points_without_adding(&self, n: usize) -> (Vec<f64>, Vec<f64>) {
        if n == 0 {
            return (vec![], vec![]);
        }

        let missing = self.missing_bounds();
        if missing.len() >= n {
            return (missing[..n].to_vec(), vec![f64::INFINITY; n]);
        }

        if self.data.is_empty() && self.pending.is_empty() {
            let (a, b) = self.bounds;
            let pts: Vec<f64> = if n == 1 {
                vec![a]
            } else {
                (0..n).map(|i| a + (b - a) * i as f64 / (n - 1) as f64).collect()
            };
            return (pts, vec![f64::INFINITY; n]);
        }

        // --- Build quals via merge of losses_combined + quals priority queue ---

        // Quals: sorted by (neg_finite_loss, left, right) → first() = max loss
        // Each entry stores the interval, n (number of splits), and raw loss.
        #[derive(Clone, Copy)]
        struct Qual {
            left: f64,
            right: f64,
            n: usize,
            loss: f64,
        }

        // We store quals in a BTreeMap keyed by (neg_fl, left, right, n)
        // to ensure uniqueness and efficient max access.
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
        struct QKey {
            neg_fl: OF64,
            left: OF64,
            right: OF64,
            n: u64,
        }

        let x_scale = self.x_scale;
        let mk_key = |q: &Qual| -> QKey {
            let fl = finite_loss_with_n(q.loss, q.left, q.right, q.n, x_scale);
            QKey {
                neg_fl: OF64::from(-fl),
                left: OF64::from(q.left),
                right: OF64::from(q.right),
                n: q.n as u64,
            }
        };

        let mut quals: BTreeMap<QKey, Qual> = BTreeMap::new();
        let insert_qual = |quals: &mut BTreeMap<QKey, Qual>, q: Qual| {
            let key = mk_key(&q);
            quals.insert(key, q);
        };

        // Add missing-bound intervals to quals
        if !missing.is_empty() {
            let all_pts: Vec<f64> = self
                .data
                .keys()
                .chain(self.pending.iter())
                .map(|x| x.into_inner())
                .collect();
            let min_pt = all_pts.iter().copied().fold(f64::INFINITY, f64::min);
            let max_pt = all_pts
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let bound_intervals = [
                (self.bounds.0, min_pt),
                (max_pt, self.bounds.1),
            ];
            for (ival, &bound) in bound_intervals.iter().zip(&[self.bounds.0, self.bounds.1]) {
                if self.is_missing_bound(bound) {
                    insert_qual(
                        &mut quals,
                        Qual {
                            left: ival.0,
                            right: ival.1,
                            n: 1,
                            loss: f64::INFINITY,
                        },
                    );
                }
            }
        }

        let points_to_go = n - missing.len();

        // Collect combined losses in priority order
        let combined_sorted: Vec<(f64, f64, f64)> = self
            .losses_combined
            .iter_by_priority()
            .map(|(l, r, loss)| (l.into_inner(), r.into_inner(), loss))
            .collect();
        let i_max = combined_sorted.len();
        let mut i = 0;

        for _ in 0..points_to_go {
            // Peek at best qual
            let qual_top = quals.first_key_value().map(|(k, q)| (*k, *q));
            // Peek at next from combined
            let ival_entry = if i < i_max {
                Some(combined_sorted[i])
            } else {
                None
            };

            let prefer_combined = match (&ival_entry, &qual_top) {
                (Some((il, ir, loss_c)), Some((qk, _qq))) => {
                    let fl_c = finite_loss_value(*loss_c, *il, *ir, x_scale);
                    let fl_q = -qk.neg_fl.into_inner();
                    // Python compares (loss, interval) tuples lexicographically
                    if fl_c != fl_q {
                        fl_c >= fl_q
                    } else {
                        (*il, *ir) >= (qk.left.into_inner(), qk.right.into_inner())
                    }
                }
                (Some(_), None) => true,
                (None, Some(_)) => false,
                (None, None) => false,
            };

            if prefer_combined {
                let (il, ir, loss_c) = ival_entry.unwrap();
                i += 1;
                insert_qual(
                    &mut quals,
                    Qual {
                        left: il,
                        right: ir,
                        n: 2,
                        loss: loss_c / 2.0,
                    },
                );
            } else {
                // Pop from quals and re-insert with n+1
                let (qk, qq) = qual_top.unwrap();
                quals.remove(&qk);
                let new_n = qq.n + 1;
                let new_loss = qq.loss * qq.n as f64 / new_n as f64;
                insert_qual(
                    &mut quals,
                    Qual {
                        left: qq.left,
                        right: qq.right,
                        n: new_n,
                        loss: new_loss,
                    },
                );
            }
        }

        // Generate points and loss_improvements from quals
        // Quals are iterated in sorted order (highest loss first),
        // which determines the order of points.
        let mut points: Vec<f64> = Vec::with_capacity(n);
        let mut improvements: Vec<f64> = Vec::with_capacity(n);

        points.extend_from_slice(&missing);
        improvements.extend(std::iter::repeat(f64::INFINITY).take(missing.len()));

        for (_key, q) in &quals {
            let interior = linspace_interior(q.left, q.right, q.n);
            improvements.extend(std::iter::repeat(q.loss).take(interior.len()));
            points.extend(interior);
        }

        (points, improvements)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tell_and_ask() {
        let mut learner = Learner1D::new((0.0, 1.0), LossFunction::Default);
        learner.tell(0.0, YValue::Scalar(0.0));
        learner.tell(1.0, YValue::Scalar(1.0));
        let (pts, _) = learner.ask(1, false);
        assert_eq!(pts.len(), 1);
        assert!((pts[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_linspace_interior() {
        assert_eq!(linspace_interior(0.0, 1.0, 1), Vec::<f64>::new());
        assert_eq!(linspace_interior(0.0, 1.0, 2), vec![0.5]);
        let pts = linspace_interior(0.0, 1.0, 4);
        assert_eq!(pts.len(), 3);
        assert!((pts[0] - 0.25).abs() < 1e-10);
        assert!((pts[1] - 0.5).abs() < 1e-10);
        assert!((pts[2] - 0.75).abs() < 1e-10);
    }
}
