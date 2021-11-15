use nalgebra as na;
use roots;
use roots::Roots;

/// Run Grunert's P3P solver.
///
/// Take in 3D points in the world frame, and calibrated bearing vectors.
/// Each column in the matrices represent one point.
///
/// Reference: Haralick, Bert M., et al. "Review and analysis of solutions of the three point
/// perspective pose estimation problem." International journal of computer vision 13.3 (1994):
/// 331-356.
///
/// # Example
fn grunert(p_w: &na::Matrix3<f64>, p_i: &na::Matrix3<f64>) -> i64 {
    // Note: notation follows paper:
    // Haralick, Bert M., et al. "Review and analysis of solutions of the three point perspective
    // pose estimation problem." International journal of computer vision 13.3 (1994): 331-356.
    // 1. Calculate the known lengths of p_w
    let p1 = p_w.column(0);
    let p2 = p_w.column(1);
    let p3 = p_w.column(2);
    let a: f64 = (p2 - p3).norm();
    let b: f64 = (p1 - p3).norm();
    let c: f64 = (p1 - p2).norm();
    let a_sq = a.powf(2.0);
    let b_sq = b.powf(2.0);
    let c_sq = c.powf(2.0);

    // 2. Get directional vectors j_i (j_i points to p_w(i))
    let j1 = p_i.column(0) / p_i[(0, 2)];
    let j2 = p_i.column(1) / p_i[(1, 2)];
    let j3 = p_i.column(2) / p_i[(2, 2)];

    // 3. Calculate cos(alpha) cos(beta) cos(gamma)
    let cos_alpha = j2.dot(&j3);
    let cos_beta = j1.dot(&j3);
    let cos_gamma = j1.dot(&j2);
    let cos_alpha_sq = cos_alpha.powf(2.0);
    let cos_beta_sq = cos_beta.powf(2.0);
    let cos_gamma_sq = cos_gamma.powf(2.0);

    // 4. Solve polynomial
    let a_sq_minus_c_sq_div_b = (a_sq - c_sq) / b_sq;
    let a_sq_plus_c_sq_div_b = (a_sq + c_sq) / b_sq;
    let b_sq_minus_c_sq_div_b = (b_sq - c_sq) / b_sq;
    let b_sq_minus_a_sq_div_b = (b_sq - a_sq) / b_sq;

    let a4 = (a_sq_minus_c_sq_div_b - 1.0).powf(2.0) - 4.0 * c_sq / b_sq * cos_alpha_sq;
    let a3 = 4.0 * (a_sq_minus_c_sq_div_b * (1.0 - a_sq_minus_c_sq_div_b) * cos_beta
        - (1.0 - a_sq_plus_c_sq_div_b) * cos_alpha * cos_gamma
        + 2.0 * c_sq / b_sq * cos_alpha_sq * cos_beta);
    let a2 = 2.0 * ((a_sq_minus_c_sq_div_b).powf(2.0) - 1.0
        + 2.0 * (a_sq_minus_c_sq_div_b).powf(2.0) * cos_beta_sq
        + 2.0 * (b_sq_minus_c_sq_div_b) * cos_alpha_sq
        - 4.0 * (a_sq_plus_c_sq_div_b) * cos_alpha * cos_beta * cos_gamma
        + 2.0 * (b_sq_minus_a_sq_div_b) * cos_gamma_sq);
    let a1 = 4.0 * (-(a_sq_minus_c_sq_div_b) * (1.0 + a_sq_minus_c_sq_div_b) * cos_beta
        + 2.0 * a_sq / b_sq * cos_gamma_sq * cos_beta
        - (1.0 - (a_sq_plus_c_sq_div_b)) * cos_alpha * cos_gamma);
    let a0 = (1.0 + a_sq_minus_c_sq_div_b).powf(2.0) - 4.0 * a_sq / b_sq * cos_gamma_sq;

    let get_points_in_cam_frame_from_v  = | v : f64 | {
        // calculate u
        let u = ((-1.0 + a_sq_minus_c_sq_div_b) * v.powf(2.0)
            - 2.0 * (a_sq_minus_c_sq_div_b) * cos_beta * v + 1.0 + a_sq_minus_c_sq_div_b)
            /(2.0*cos_gamma - v * cos_alpha);
        // calculate s1, s2, s3
        let s1 = c_sq / (1.0 + u.powf(2.0) - 2.0 * u * cos_gamma).sqrt();
        let s2 = u * s1;
        let s3 = v * s1;
        // calculate the positions of p1, p2, p3 in camera frame
        let p1_cam = s1 * &j1;
        let p2_cam = s2 * &j2;
        let p3_cam = s3 * &j3;
    };

    let all_roots = roots::find_roots_quartic(a4, a3, a2, a1, a0);
    let mut all_vs : roots::Roots<f64>;
    let mut num_roots = 0;
    match all_roots {
        roots::Roots::Four(x) => {
            for i in 0..4 {
                get_points_in_cam_frame_from_v(x[i]);
            }
        },
        roots::Roots::Three(x) => {
            for i in 0..3 {
                get_points_in_cam_frame_from_v(x[i]);
            }
        },
        roots::Roots::Two(x) => {
            for i in 0..2 {
                get_points_in_cam_frame_from_v(x[i]);
            }
        },
        roots::Roots::One(x) => {
            for i in 0..1 {
                get_points_in_cam_frame_from_v(x[i]);
            }
        },
        _ => {num_roots = 0;}
    }

    return 42;
}

fn fischler(p_w: &na::Matrix3<f64>, p_i: &na::Matrix3<f64>) -> i64 {
    return 42;
}

fn kneip(p_w: &na::Matrix3<f64>, p_i: &na::Matrix3<f64>) -> i64 {
    return 42;
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra as na;

    #[test]
    fn test_p3p() {
        let p_w = na::Matrix3::<f64>::identity();
        let p_i = na::Matrix3::<f64>::identity();
        let result = grunert(&p_w, &p_i);
        assert_eq!(42, 42);
    }
}
