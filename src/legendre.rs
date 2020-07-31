// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Code for Legendre polynomials.
 */

/// Evaluates the Legendre polynomial Pm(n,m) at x. x must satisfy |x| <= 1.
///
/// # Arguments
///
/// `n` - the maximum first index of the Legendre function, which must be at
/// least 0.
/// `m` - the second index of the Legendre function, which must be at least 0,
/// and no greater than N.
/// `x` - the points at which the function is to be evaluated.
///
// This code is a re-write of the C code here:
// https://people.sc.fsu.edu/~jburkardt/c_src/legendre_polynomial/legendre_polynomial.html
// The C code distributed under the GNU LGPL license, and thus this function is
// also licensed under the LGPL. A copy of the LGPL license can be found at
// https://www.gnu.org/licenses/lgpl-3.0.en.html
// TODO: Benchmark. Does this need to go faster?
pub(crate) fn legendre_values(n: usize, m: usize, x: &[f64]) -> Vec<f64> {
    let mm = x.len();
    let mut v = vec![0.0; mm * (n + 1)];

    // J = M is the first non-zero function.
    if m <= n {
        for i in 0..mm {
            v[i + m * mm] = 1.0;
        }

        let mut fact = 1.0;
        for _ in 0..m {
            for i in 0..mm {
                v[i + m * mm] = -v[i + m * mm] * fact * (1.0 - x[i] * x[i]);
            }
            fact += 2.0;
        }
    }

    // J = M + 1 is the second nonzero function.
    if m + 1 <= n {
        for i in 0..mm {
            v[i + (m + 1) * mm] = x[i] * (2 * m + 1) as f64 * v[i + m * mm];
        }
    }

    for j in (m + 2)..=n {
        for i in 0..mm {
            let ji = j as isize;
            let mi = m as isize;
            v[i + j * mm] = ((2 * j - 1) as f64 * x[i] * v[i + (j - 1) * mm]
                + (-ji - mi + 1) as f64 * v[i + (j - 2) * mm])
                / (ji - mi) as f64;
        }
    }

    v
}

/// Returns list of Legendre polynomial values calculated up to order n_max.
// This function is a re-write of P1SIN within the RTS file mwa_tile.c.
pub(crate) fn p1sin(n_max: usize, theta: f64) -> (Vec<f64>, Vec<f64>) {
    let mut all_vals = vec![0.0; (n_max + 1) * (n_max + 2) / 2];

    let size = n_max * n_max + 2 * n_max;
    let mut p1sin_out = vec![0.0; size];
    let mut p1_out = vec![0.0; size];

    let (s_theta, u) = theta.sin_cos();
    let delta_u = 1e-6;

    let mut pm_in = [u];
    let mut m_incr = 0;
    for m in 0..=n_max {
        let pm_vals = legendre_values(n_max, m, &pm_in);
        for i in m..n_max {
            if !(i == 0 && m == 0) {
                all_vals[(i - m) + m_incr] = pm_vals[i];
            }
        }
        m_incr += n_max - m + 1;
    }
    for p in &all_vals {
        println!("{}", p);
    }

    for n in 1..n_max {
        let mut p = vec![0.0; n + 1];
        let mut pm1 = vec![0.0; n + 1];
        let mut pm_sin = vec![0.0; n + 1];
        pm_in[0] = u;

        m_incr = 0;
        for order in 0..=n {
            let index = n + m_incr;
            p[order] = all_vals[index];
            if order > 0 {
                pm1[order - 1] = all_vals[index];
            }
            if order == n {
                pm1[order] = 0.0;
            }
            pm_sin[order] = 0.0;
            m_incr += n_max - order;
        }

        if u == 1.0 {
            pm_in[0] = u - delta_u;
            let pm_vals = legendre_values(n, 0, &pm_in);
            pm_sin[1] = -(p[0] - pm_vals[n]) / delta_u;
        } else if u == -1.0 {
            pm_in[0] = u - delta_u;
            let pm_vals = legendre_values(n, 0, &pm_in);
            pm_sin[1] = -(pm_vals[n] - p[0]) / delta_u;
        } else {
            for order in 0..=n {
                pm_sin[order] = p[order] / s_theta;
            }
        }

        let ind_start = (n - 1) * (n - 1) + 2 * (n - 1);
        let ind_stop = n * n + 2 * n;
        for i in ind_start..ind_stop {
            let index = i - ind_start;
            let j = if index < n { n - index } else { index - n };
            p1sin_out[i] = pm_sin[j];
            p1_out[i] = pm1[j];
        }
    }

    // The C++ version of this code takes two vector pointers in for `p1sin_out`
    // and `p1_out`. But, the same code never allocates memory against those
    // pointers. So, we just make the vectors here.

    // let mut p = vec![0.0; n_max + 1];
    // let mut pm1 = vec![0.0; n_max + 1];
    // let mut pm_sin = vec![0.0; n_max + 1];
    // // pm_in[0] = u;
    // // let mut pu_mdelu = vec![0.0; n_max + 1];
    // // let mut pm_sin_merged = vec![0.0; 2 * n_max + 1];
    // // let mut pm1_merged = vec![0.0; 2 * n_max + 1];

    // // Create a look-up table for the legendre polynomials
    // // Such that legendre_table[ m * nmax + (n-1) ] = legendre(n, m, u)
    // let mut legendre_table = vec![0.0; n_max * (n_max + 1)];
    // for m in 0..n_max + 1 {
    //     // let mut leg0 = legendre_polynomial()
    //     // for n in 2..n_max + 1 {
    //     //     match m.cmp(&n) {
    //     //         Ordering::Less =>
    //     //     }
    //     //     if (m < n) {
    //     //     }
    //     // }
    // }

    (p1sin_out, p1_out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;

    #[test]
    fn legendre_values_n5m0() {
        let x = vec![-1.0, 0.0, 1.0];
        let result = legendre_values(5, 0, &x);
        let expected = vec![
            1.000000, 1.000000, 1.000000, -1.000000, 0.000000, 1.000000, 1.000000, -0.500000,
            1.000000, -1.000000, -0.000000, 1.000000, 1.000000, 0.375000, 1.000000, -1.000000,
            0.000000, 1.000000,
        ];
        assert_eq!(result.len(), expected.len());
        for (r, e) in result.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(r, e, epsilon = 1e-6);
        }
    }

    #[test]
    fn legendre_values_n5m1() {
        let x = vec![-1.0, 0.0, 1.0];
        let result = legendre_values(5, 1, &x);
        let expected = vec![
            0.000000, 0.000000, 0.000000, -0.000000, -1.000000, -0.000000, 0.000000, -0.000000,
            -0.000000, 0.000000, 1.500000, 0.000000, -0.000000, 0.000000, 0.000000, 0.000000,
            -1.875000, 0.000000,
        ];
        assert_eq!(result.len(), expected.len());
        for (r, e) in result.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(r, e, epsilon = 1e-6);
        }
    }

    #[test]
    fn p1sin_16_0() {
        let (p1sin_out, p1_out) = p1sin(16, 0.0);
        for (i, &p) in p1sin_out.iter().enumerate() {
            match i {
                0 | 2 => assert_abs_diff_eq!(p, -1.000000, epsilon = 1e-6),
                4 | 6 => assert_abs_diff_eq!(p, -2.999999, epsilon = 1e-6),
                10 | 12 => assert_abs_diff_eq!(p, -5.999993, epsilon = 1e-6),
                18 | 20 => assert_abs_diff_eq!(p, -9.999978, epsilon = 1e-6),
                28 | 30 => assert_abs_diff_eq!(p, -14.999948, epsilon = 1e-6),
                40 | 42 => assert_abs_diff_eq!(p, -20.999895, epsilon = 1e-6),
                // Because we've matched specific numbers above, this range will
                // only match numbers that have not been already matched. Use
                // the range of the biggest number above, as not all indices
                // will be 0.
                0..=40 => assert_abs_diff_eq!(p, 0.000000, epsilon = 1e-6),
                _ => (),
            }
        }
        for (i, &p) in p1_out.iter().enumerate() {
            match i {
                1 | 5 | 9 | 13 | 17 | 21 | 25 | 33 | 37 | 45 => {
                    assert_abs_diff_eq!(p, -0.000000, epsilon = 1e-6)
                }
                0..=45 => assert_abs_diff_eq!(p, 0.000000, epsilon = 1e-6),
                _ => (),
            }
        }
    }

    #[test]
    fn p1sin_5_0() {
        let (p1sin_out, p1_out) = p1sin(5, 0.0);
        for (i, &p) in p1sin_out.iter().enumerate() {
            println!("p1sin_out: {}: {}", i, p);
            match i {
                0 | 2 => assert_abs_diff_eq!(p, -1.000000, epsilon = 1e-6),
                4 | 6 => assert_abs_diff_eq!(p, -2.999999, epsilon = 1e-6),
                10 | 12 => assert_abs_diff_eq!(p, -5.999993, epsilon = 1e-6),
                18 | 20 => assert_abs_diff_eq!(p, -9.999978, epsilon = 1e-6),
                28 | 30 => assert_abs_diff_eq!(p, -14.999948, epsilon = 1e-6),
                _ => (),
            }
        }
        for (i, &p) in p1_out.iter().enumerate() {
            println!("p1_out: {}: {}", i, p);
            match i {
                1 | 5 | 9 | 13 | 17 | 21 | 25 | 33 => {
                    assert_abs_diff_eq!(p, -0.000000, epsilon = 1e-6)
                }
                0..=45 => assert_abs_diff_eq!(p, 0.000000, epsilon = 1e-6),
                _ => (),
            }
        }
    }

    #[test]
    fn p1sin_5_0785398160() {
        let (p1sin_out, p1_out) = p1sin(5, 0.78539816);
        for (i, &p) in p1sin_out.iter().enumerate() {
            println!("p1sin_out: {}: {}", i, p);
            match i {
                0 | 2 => assert_abs_diff_eq!(p, -1.000000, epsilon = 1e-6),
                1 => assert_abs_diff_eq!(p, -2.999999, epsilon = 1e-6),
                3 | 5 | 7 => assert_abs_diff_eq!(p, 2.121320, epsilon = 1e-6),
                4 | 6 => assert_abs_diff_eq!(p, -2.121320, epsilon = 1e-6),
                8 | 14 => assert_abs_diff_eq!(p, -7.500000, epsilon = 1e-6),
                9 | 13 => assert_abs_diff_eq!(p, -7.500000, epsilon = 1e-6),
                _ => (),
            }
        }
        for (i, &p) in p1_out.iter().enumerate() {
            println!("p1_out: {}: {}", i, p);
            match i {
                // 1 => assert_abs_diff_eq!(p, -0.707107, epsilon = 1e-6),
                4 => assert_abs_diff_eq!(p, 1.500000, epsilon = 1e-6),
                5 => assert_abs_diff_eq!(p, -1.500000, epsilon = 1e-6),
                9 | 13 => assert_abs_diff_eq!(p, -5.303301, epsilon = 1e-6),
                10 | 12 => assert_abs_diff_eq!(p, 5.303301, epsilon = 1e-6),
                0..=13 => assert_abs_diff_eq!(p, 0.000000, epsilon = 1e-6),
                _ => (),
            }
        }
    }
}
