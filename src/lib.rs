//! The P-Square (P2) algorithm for dynamic calculation of quantiles without
//! storing observations.
//!
//! An implementation based on the algorithm described in this [paper].
//!
//! This algorithm calculates estimates for percentiles of observations sets
//! dynamically, with a O(1) space complexity.
//!
//! # Examples
//!
//! ```
//! # use p_square::P2;
//! #
//! let mut p2 = P2::new(0.3);
//!
//! for n in 1..=100 {
//!     p2.feed(n as f64);
//! }
//!
//! assert_eq!(p2.estimate(), 30.0);
//! ```
//!
//! [paper]: https://www.cse.wustl.edu/~jain/papers/ftp/psqr.pdf

const MARKERS_COUNT: usize = 5;

/// Marker indices for `q`.
mod marker_index {
    /// Minimum of the observations so far.
    pub(super) const MINIMUM: usize = 0;
    /// Current estimate of the `p/2`-quantile.
    pub(super) const LOWER_MEDIAN: usize = 1;
    /// Current estimate of the `p`-quantile.
    pub(super) const QUANTILE: usize = 2;
    /// Current estimate of the `(1+p)/2`-quantile.
    pub(super) const UPPER_MEDIAN: usize = 3;
    /// Current estimate of the `(1+p)/2`-quantile.
    pub(super) const MAXIMUM: usize = 4;
}

#[derive(Clone)]
pub struct P2 {
    quantile: f64,

    // q
    heights: [f64; MARKERS_COUNT],
    // n
    positions: [f64; MARKERS_COUNT],
    // n'
    desired_positions: [f64; MARKERS_COUNT],
    // dn'
    increments: [f64; MARKERS_COUNT],

    observations_counter: usize,
}

impl P2 {
    /// Construct a new P2 state for estimating the `quantile` of the
    /// observations.
    ///
    /// See the [crate](crate) documentation for more.
    ///
    /// # Panics
    ///
    /// Will panic if `quantile` is not in the range `0.0..=1.0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use p_square::P2;
    ///
    /// // A P-Square state to estimate a median.
    /// let mut p2 = P2::new(0.5);
    /// ```
    pub const fn new(quantile: f64) -> Self {
        assert!(
            0.0 <= quantile && quantile <= 1.0,
            "quantile must be in the range 0.0..=1.0"
        );

        let heights = [0.0; MARKERS_COUNT];
        let positions = [0.0, 1.0, 2.0, 3.0, 4.0];
        let desired_positions = [
            0.0,
            2.0 * quantile,
            4.0 * quantile,
            2.0 + 2.0 * quantile,
            4.0,
        ];
        let increments = [0.0, quantile / 2.0, quantile, (1.0 + quantile) / 2.0, 1.0];

        Self {
            quantile,
            heights,
            positions,
            desired_positions,
            increments,
            observations_counter: 0,
        }
    }

    /// Current estimate of the desired quantile.
    pub fn estimate(&self) -> f64 {
        // PERF: this might be a good use case for hint::unlikely (but it's not
        // stable).
        if self.observations_counter <= MARKERS_COUNT {
            // XXX: is this the best way to handle this situation?

            let len = self.observations_counter;
            let Some(max_idx) = len.checked_sub(1) else {
                return 0.0;
            };

            let index = (max_idx as f64 * self.quantile).round() as usize;
            debug_assert!(index <= max_idx, "quantile <= 1");

            let initialized_heights = &mut self.heights.clone()[..len];

            let (_lesser, v, _greater) =
                initialized_heights.select_nth_unstable_by(index, |a, b| a.total_cmp(b));

            *v
        } else {
            self.q3()
        }
    }

    /// Feed a new observation.
    pub fn feed(&mut self, observation: f64) {
        let j = self.observations_counter;
        self.observations_counter = self.observations_counter.saturating_add(1);

        if j < MARKERS_COUNT {
            self.heights[j] = observation;

            if j + 1 == MARKERS_COUNT {
                self.heights.sort_unstable_by(|a, b| a.total_cmp(b));
            }

            return;
        }

        // B.1.
        let k: usize = if observation < self.q1() {
            self.heights[marker_index::MINIMUM] = observation;
            0
        } else if self.q1() <= observation && observation < self.q2() {
            0
        } else if self.q2() <= observation && observation < self.q3() {
            1
        } else if self.q3() <= observation && observation < self.q4() {
            2
        } else if self.q4() <= observation && observation <= self.q5() {
            3
        } else if self.q5() < observation {
            self.heights[marker_index::MAXIMUM] = observation;
            3
        } else {
            unreachable!();
        };

        // B.2.
        for n in self.positions.iter_mut().skip(k + 1) {
            *n += 1.0;
        }

        for (n, d) in self.desired_positions.iter_mut().zip(self.increments) {
            *n += d;
        }

        // B.3.
        for i in marker_index::LOWER_MEDIAN..=marker_index::UPPER_MEDIAN {
            let d = self.np(i) - self.n(i);

            if (d >= 1.0 && self.n(i + 1) - self.n(i) > 1.0)
                || (d <= -1.0 && self.n(i - 1) - self.n(i) < -1.0)
            {
                let d_sign = d.signum();
                let qp = self.parabolic(i, d_sign);

                self.heights[i] = if self.q(i - 1) < qp && qp < self.q(i + 1) {
                    qp
                } else {
                    self.linear(i, d_sign)
                };

                self.positions[i] += d_sign;
            }
        }
    }

    const fn parabolic(&self, i: usize, d: f64) -> f64 {
        self.q(i)
            + (d / (self.n(i + 1) - self.n(i - 1)))
                * ((self.n(i) - self.n(i - 1) + d)
                    * ((self.q(i + 1) - self.q(i)) / (self.n(i + 1) - self.n(i)))
                    + (self.n(i + 1) - self.n(i) - d)
                        * ((self.q(i) - self.q(i - 1)) / (self.n(i) - self.n(i - 1))))
    }

    const fn linear(&self, i: usize, d: f64) -> f64 {
        let i_plus_d = (i as i64 + d as i64) as usize;

        self.q(i) + d * ((self.q(i_plus_d) - self.q(i)) / (self.n(i_plus_d) - self.n(i)))
    }

    // Helper getters

    #[inline]
    const fn q(&self, i: usize) -> f64 {
        self.heights[i]
    }

    #[inline]
    const fn n(&self, i: usize) -> f64 {
        self.positions[i]
    }

    #[inline]
    const fn np(&self, i: usize) -> f64 {
        self.desired_positions[i]
    }

    #[inline]
    fn q1(&self) -> f64 {
        self.heights[marker_index::MINIMUM]
    }

    #[inline]
    fn q2(&self) -> f64 {
        self.heights[marker_index::LOWER_MEDIAN]
    }

    #[inline]
    fn q3(&self) -> f64 {
        self.heights[marker_index::QUANTILE]
    }

    #[inline]
    fn q4(&self) -> f64 {
        self.heights[marker_index::UPPER_MEDIAN]
    }

    #[inline]
    fn q5(&self) -> f64 {
        self.heights[marker_index::MINIMUM]
    }
}

pub fn from_iter<I>(quantile: f64, iter: I) -> f64
where
    I: Iterator<Item = f64>,
{
    let mut state = P2::new(quantile);

    iter.for_each(|observation| state.feed(observation));

    state.estimate()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn before_initialization() {
        let n = from_iter(0.5, (1..=1).map(|n| n as f64));
        assert_eq!(n, 1.0);

        let n = from_iter(0.7, (1..=2).map(|n| n as f64));
        assert_eq!(n, 2.0);

        let n = from_iter(0.5, (1..=3).map(|n| n as f64));
        assert_eq!(n, 2.0);

        let n = from_iter(0.25, (1..=4).map(|n| n as f64));
        assert_eq!(n, 2.0);

        let n = from_iter(0.6, (1..=5).map(|n| n as f64));
        assert_eq!(n, 3.0);
    }

    #[test]
    #[should_panic(expected = "quantile must be in the range 0.0..=1.0")]
    fn invalid_quantile() {
        P2::new(1.2);
    }
}
