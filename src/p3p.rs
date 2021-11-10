use nalgebra as na;
use roots;

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
    let a_sq = a.checked_pow(2);
    let b_sq = b.checked_pow(2);
    let c_sq = c.checked_pow(2);

    // 2. Get directional vectors j_i (j_i points to p_w(i))
    let j1 = p_i.column(0) / p_i[(0, 2)];
    let j2 = p_i.column(1) / p_i[(1, 2)];
    let j3 = p_i.column(2) / p_i[(2, 2)];

    // 3. Calculate cos(alpha) cos(beta) cos(gamma)
    let cos_alpha = j2.dot(&j3);
    let cos_beta = j1.dot(&j3);
    let cos_gamma = j1.dot(&j2);
    let cos_alpha_sq = cos_alpha.checked_pow(2);
    let cos_beta_sq = cos_beta.checked_pow(2);
    let cos_gamma_sq = cos_gamma.checked_pow(2);

    // 4. Solve polynomial
    let a_sq_minus_c_sq_div_b = (a_sq - c_sq) / b_sq;
    let a_sq_plus_c_sq_div_b = (a_sq + c_sq) / b_sq;
    let b_sq_minus_c_sq_div_b = (b_sq - c_sq) / b_sq;
    let b_sq_minus_a_sq_div_b = (b_sq - a_sq) / b_sq;

    let a4 = (a_sq_minus_c_sq_div_b - 1).checked_pow(2) - 4 * c_sq / b_sq * cos_alpha_sq;
    let a3 = 4 * (a_sq_minus_c_sq_div_b * (1 - a_sq_minus_c_sq_div_b) * cos_beta
        - (1 - a_sq_plus_c_sq_div_b) * cos_alpha * cos_gamma
        + 2 * c_sq / b_sq * cos_alpha_sq * cos_beta);
    let a2 = 2 * ((a_sq_minus_c_sq_div_b).checked_pow(2) - 1 + 2 * (a_sq_minus_c_sq_div_b).checked_pow(2) * cos_beta_sq);

    println!("a: {}", a);
    println!("b: {}", b);
    println!("c: {}", c);
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
