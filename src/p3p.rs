use nalgebra as na;
use roots;
use roots::Roots;
use super::solution::Solution;

/// Run Arun's method to solve for rigid body transformation:
/// p_prime = R * p + t
/// where p and p_prime are 3x3 matrices with each column represent one 3D point
fn arun(p: &na::Matrix3<f64>, p_prime: &na::Matrix3<f64>) -> (na::Matrix3<f64>, na::Vector3<f64>) {
    // find centroids
    let p_centroid = p.column_mean();
    let p_prime_centroid = p_prime.column_mean();

    // calculate the vectors from centroids
    let mut q = na::Matrix3::<f64>::zeros();
    q.copy_from(p);
    for (i, mut column) in q.column_iter_mut().enumerate() {
        column -= &p_centroid;
    }
    let mut q_prime = na::Matrix3::<f64>::zeros();
    q_prime.copy_from(p_prime);
    for (i, mut column) in q_prime.column_iter_mut().enumerate() {
        column -= &p_prime_centroid;
    }

    // rotation estimation
    let mut H = na::Matrix3::<f64>::zeros();
    for i in 0..3 {
        let q_i = q.column(i);
        let q_prime_i = q_prime.column(i);
        H += q_i * q_prime_i.transpose();
    };
    let H_svd = H.svd(true, true);
    match (H_svd.v_t, H_svd.u) {
        (Some(v_t), Some(u)) => {
            let V = v_t.transpose();
            let U_transpose = u.transpose();
            let diagonal = na::Matrix3::<f64>::from_diagonal(
                &na::Vector3::<f64>::new(1.0, 1.0, V.determinant() * U_transpose.determinant()));
            let R: na::Matrix3<f64> = V * diagonal * U_transpose;

            // translation estimation
            let t = p_prime_centroid - R * p_centroid;

            return (R, t);
        }
        _ => panic!("SVD failed in Arun's method.")
    }
}


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
fn grunert(p_w: &na::Matrix3<f64>, p_i: &na::Matrix3<f64>) -> Vec<Solution> {
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
    let a_sq = a.powi(2);
    let b_sq = b.powi(2);
    let c_sq = c.powi(2);

    // 2. Get directional vectors j_i (j_i points to p_w(i))
    let j1 = p_i.column(0).normalize();
    let j2 = p_i.column(1).normalize();
    let j3 = p_i.column(2).normalize();

    // 3. Calculate cos(alpha) cos(beta) cos(gamma)
    // note: cosines need to be within [-1, 1]
    let cos_alpha = j2.dot(&j3);
    let cos_beta = j1.dot(&j3);
    let cos_gamma = j1.dot(&j2);
    let cos_alpha_sq = cos_alpha.powi(2);
    let cos_beta_sq = cos_beta.powi(2);
    let cos_gamma_sq = cos_gamma.powi(2);

    // 4. Solve polynomial
    let a_sq_minus_c_sq_div_b_sq = (a_sq - c_sq) / b_sq;
    let a_sq_plus_c_sq_div_b_sq = (a_sq + c_sq) / b_sq;
    let b_sq_minus_c_sq_div_b_sq = (b_sq - c_sq) / b_sq;
    let b_sq_minus_a_sq_div_b_sq = (b_sq - a_sq) / b_sq;

    let a4 = (a_sq_minus_c_sq_div_b_sq - 1.0).powi(2) - 4.0 * c_sq / b_sq * cos_alpha_sq;
    let a3 = 4.0 * (a_sq_minus_c_sq_div_b_sq * (1.0 - a_sq_minus_c_sq_div_b_sq) * cos_beta
        - (1.0 - a_sq_plus_c_sq_div_b_sq) * cos_alpha * cos_gamma
        + 2.0 * c_sq / b_sq * cos_alpha_sq * cos_beta);
    let a2 = 2.0 * ((a_sq_minus_c_sq_div_b_sq).powi(2) - 1.0
        + 2.0 * (a_sq_minus_c_sq_div_b_sq).powi(2) * cos_beta_sq
        + 2.0 * (b_sq_minus_c_sq_div_b_sq) * cos_alpha_sq
        - 4.0 * (a_sq_plus_c_sq_div_b_sq) * cos_alpha * cos_beta * cos_gamma
        + 2.0 * (b_sq_minus_a_sq_div_b_sq) * cos_gamma_sq);
    let a1 = 4.0 * (-(a_sq_minus_c_sq_div_b_sq) * (1.0 + a_sq_minus_c_sq_div_b_sq) * cos_beta
        + 2.0 * a_sq / b_sq * cos_gamma_sq * cos_beta
        - (1.0 - (a_sq_plus_c_sq_div_b_sq)) * cos_alpha * cos_gamma);
    let a0 = (1.0 + a_sq_minus_c_sq_div_b_sq).powi(2) - 4.0 * a_sq / b_sq * cos_gamma_sq;

    let get_points_in_cam_frame_from_v = |v: f64| -> na::Matrix3<f64>{
        // return a 3x3 matrix, with each column represent 1 point
        // calculate u
        let u = ((-1.0 + a_sq_minus_c_sq_div_b_sq) * v.powi(2)
            - 2.0 * (a_sq_minus_c_sq_div_b_sq) * cos_beta * v + 1.0 + a_sq_minus_c_sq_div_b_sq)
            / (2.0 * (cos_gamma - v * cos_alpha));
        // calculate s1, s2, s3
        let s1 = (c_sq / (1.0 + u.powi(2) - 2.0 * u * cos_gamma)).sqrt();
        let s2 = u * s1;
        let s3 = v * s1;
        // calculate the positions of p1, p2, p3 in camera frame
        let p_cam = na::Matrix3::<f64>::from_columns(&[s1 * &j1, s2 * &j2, s3 * &j3]);
        return p_cam;
    };

    let all_roots = roots::find_roots_quartic(a4, a3, a2, a1, a0);
    let num_roots = all_roots.as_ref().len();
    let mut results = Vec::<Solution>::new();
    for i in 0..num_roots {
        let p_cam = get_points_in_cam_frame_from_v(all_roots.as_ref()[i]);
        // calculate the rotation and translation using Arun's method
        let (rotation_est, t_est) = arun(p_w, &p_cam);
        results.push(Solution { rotation: rotation_est, translation: t_est });
    }

    return results;
}

/// Run Fischler's P3P solver.
///
/// Take in 3D points in the world frame, and calibrated bearing vectors.
/// Each column in the matrices represent one point.
///
/// Reference: Haralick, Bert M., et al. "Review and analysis of solutions of the three point
/// perspective pose estimation problem." International journal of computer vision 13.3 (1994):
/// 331-356.
///
/// # Example
fn fischler(p_w: &na::Matrix3<f64>, p_i: &na::Matrix3<f64>) -> Vec<Solution> {
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
    let a_sq = a.powi(2);
    let b_sq = b.powi(2);
    let c_sq = c.powi(2);

    // 2. Get directional vectors j_i (j_i points to p_w(i))
    let j1 = p_i.column(0).normalize();
    let j2 = p_i.column(1).normalize();
    let j3 = p_i.column(2).normalize();

    // 3. Calculate cos(alpha) cos(beta) cos(gamma)
    // note: cosines need to be within [-1, 1]
    let cos_alpha = j2.dot(&j3);
    let cos_beta = j1.dot(&j3);
    let cos_gamma = j1.dot(&j2);
    let cos_alpha_sq = cos_alpha.powi(2);
    let cos_beta_sq = cos_beta.powi(2);
    let cos_gamma_sq = cos_gamma.powi(2);

    // 4. Solve polynomial
    let a_sq_minus_c_sq_div_b_sq = (a_sq - c_sq) / b_sq;
    let a_sq_plus_c_sq_div_b_sq = (a_sq + c_sq) / b_sq;
    let b_sq_minus_c_sq_div_b_sq = (b_sq - c_sq) / b_sq;
    let b_sq_minus_a_sq_div_b_sq = (b_sq - a_sq) / b_sq;

    let d4 = 4f64 * b_sq * c_sq * cos_alpha_sq - (a_sq - b_sq - c_sq).powi(2);
    let d3 = -4f64 * c_sq * (a_sq + b_sq - c_sq) * cos_alpha * cos_beta
        - 8f64 * b_sq * c_sq * cos_alpha_sq * cos_gamma
        + 4f64 * (a_sq - b_sq - c_sq) * (a_sq - b_sq) * cos_gamma;
    let d2 = 4f64 * c_sq * (a_sq - c_sq) * cos_beta_sq
        + 8f64 * c_sq * (a_sq + b_sq) * cos_alpha * cos_beta * cos_gamma
        + 4f64 * c_sq * (b_sq - c_sq) * cos_alpha_sq
        - 2f64 * (a_sq - b_sq - c_sq) * (a_sq - b_sq + c_sq)
        - 4f64 * (a_sq - b_sq).powi(2) * cos_gamma_sq;
    let d1 = -8f64 * a_sq * c_sq * cos_beta_sq * cos_gamma
        - 4f64 * c_sq * (b_sq - c_sq) * cos_alpha * cos_beta
        - 4f64 * a_sq * c_sq * cos_alpha * cos_beta
        + 4f64 * (a_sq - b_sq) * (a_sq - b_sq + c_sq) * cos_gamma;
    let d0 = 4f64 * a_sq * c_sq * cos_beta_sq - (a_sq - b_sq + c_sq).powi(2);

    let get_points_in_cam_frame_from_u = |u: f64| -> na::Matrix3<f64>{
        // return a 3x3 matrix, with each column represent 1 point
        // calculate v
        let v = ( -(a_sq - b_sq - c_sq) * u.powi(2) - 2f64 * (b_sq - a_sq) * cos_gamma * u
            - a_sq + b_sq - c_sq ) / ( 2f64 * c_sq * (cos_alpha * u - cos_beta) );
        // calculate s1, s2, s3
        let s1 = (c_sq / (1.0 + u.powi(2) - 2.0 * u * cos_gamma)).sqrt();
        let s2 = u * s1;
        let s3 = v * s1;
        // calculate the positions of p1, p2, p3 in camera frame
        let p_cam = na::Matrix3::<f64>::from_columns(&[s1 * &j1, s2 * &j2, s3 * &j3]);
        return p_cam;
    };

    let all_roots = roots::find_roots_quartic(d4, d3, d2, d1, d0);
    let num_roots = all_roots.as_ref().len();
    let mut results = Vec::<Solution>::new();
    for i in 0..num_roots {
        let p_cam = get_points_in_cam_frame_from_v(all_roots.as_ref()[i]);
        // calculate the rotation and translation using Arun's method
        let (rotation_est, t_est) = arun(p_w, &p_cam);
        results.push(Solution { rotation: rotation_est, translation: t_est });
    }

    return results;
}

fn kneip(p_w: &na::Matrix3<f64>, p_i: &na::Matrix3<f64>) -> Vec<Solution> {
    unimplemented!();
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra as na;
    use rand::Rng;
    use rand::distributions::Standard;

    #[test]
    fn test_p3p() {
        let p_w = na::Matrix3::<f64>::identity();
        let p_i = na::Matrix3::<f64>::identity();
        let result = grunert(&p_w, &p_i);
        assert_eq!(42, 42);
    }

    #[test]
    fn test_arun() {
        // generate random rotation and translation
        let mut rng = rand::thread_rng();
        let t_gt: na::Vector3<f64> = rng.gen();
        let r_gt: na::Rotation3<f64> = rng.gen();
        let rotation_gt = r_gt.matrix();

        let p_src = na::Matrix3::<f64>::new_random();
        let mut p_tgt = rotation_gt * p_src;
        for (i, mut col) in p_tgt.column_iter_mut().enumerate() {
            col += t_gt;
        }
        let (rotation_est, t_est) = arun(&p_src, &p_tgt);
        assert!(rotation_est.is_special_orthogonal(1e-7));
        // check estimated against gt
        assert!(rotation_est.relative_eq(&rotation_gt, 1e-7, 1e-7));
        assert!(t_est.relative_eq(&t_gt, 1e-7, 1e-7));
    }

    #[test]
    fn test_grunert() {
        // generate camera points (force positive z)
        let p_cam = na::Matrix3::<f64>::new(
            0.7145222331218005, 0.6997616555794328, 2.7801634549912415,
            0.7253251306266671, 1.1639214982781518, 0.238599168957371,
            0.43484773318930925, 0.3052619752472596, 0.29437234778903254,
        );

        // generate random rotation and translation
        let t_gt: na::Vector3<f64> = na::Vector3::<f64>::new(
            0.17216231844648533,
            0.8968470516910476,
            0.7639868514400336,
        );
        let rotation_gt = na::Matrix3::<f64>::new(
            -0.5871671330204742, 0.5523730621371405, -0.5917083387326525,
            -0.6943436357845219, 0.032045330116620696, 0.7189297686584192,
            0.4160789268228421, 0.8329808503458861, 0.364720755662461,
        );

        // calculate gt world points
        // p_cam = R p_world + t
        // p_world = R^T * (p_cam - t)
        let mut p_world: na::Matrix3<f64> = p_cam;
        for (i, mut column) in p_world.column_iter_mut().enumerate() {
            column -= &t_gt;
        }
        p_world = rotation_gt.transpose() * p_world;

        // get bearing vectors for the points in camera frame
        let mut p_i = p_cam;
        p_i.column_iter_mut().for_each(|mut c| c /= c[2]);

        // run grunert's p3p
        let solutions = grunert(&p_world, &p_i);
        // ensure at least one solution is consistent with gt
        let mut flag: bool = false;
        for i in 0..solutions.len() {
            let t_flag = solutions[i].translation.relative_eq(&t_gt, 1e-7, 1e-7);
            let r_flag = solutions[i].rotation.relative_eq(&rotation_gt, 1e-7, 1e-7);
            if t_flag & r_flag {
                flag = true;
                break;
            }
        }
        assert!(flag)
    }
}
