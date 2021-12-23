use nalgebra as na;
use roots;
use super::model::Model;

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
    for (_i, mut column) in q.column_iter_mut().enumerate() {
        column -= &p_centroid;
    }
    let mut q_prime = na::Matrix3::<f64>::zeros();
    q_prime.copy_from(p_prime);
    for (_i, mut column) in q_prime.column_iter_mut().enumerate() {
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
fn grunert(p_w: &na::Matrix3<f64>, p_i: &na::Matrix3<f64>) -> Vec<Model> {
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
    let mut results = Vec::<Model>::new();
    for i in 0..num_roots {
        let p_cam = get_points_in_cam_frame_from_v(all_roots.as_ref()[i]);
        // calculate the rotation and translation using Arun's method
        let (rotation_est, t_est) = arun(p_w, &p_cam);
        results.push(Model { rotation: rotation_est, translation: t_est });
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
fn fischler(p_w: &na::Matrix3<f64>, p_i: &na::Matrix3<f64>) -> Vec<Model> {
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
        let v = (-(a_sq - b_sq - c_sq) * u.powi(2) - 2f64 * (b_sq - a_sq) * cos_gamma * u
            - a_sq + b_sq - c_sq) / (2f64 * c_sq * (cos_alpha * u - cos_beta));
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
    let mut results = Vec::<Model>::new();
    for i in 0..num_roots {
        let p_cam = get_points_in_cam_frame_from_u(all_roots.as_ref()[i]);
        // calculate the rotation and translation using Arun's method
        let (rotation_est, t_est) = arun(p_w, &p_cam);
        results.push(Model { rotation: rotation_est, translation: t_est });
    }

    return results;
}

/// Implementation of Kneip's P3P algorithm.
///
/// Reference: Kneip, Laurent, Davide Scaramuzza, and Roland Siegwart.
/// "A novel parametrization of the perspective-three-point problem for a direct computation of
/// absolute camera position and orientation." CVPR 2011. IEEE, 2011.
///
fn kneip(p_w: &na::Matrix3<f64>, p_i: &na::Matrix3<f64>) -> Vec<Model> {
    // check for degenerate case
    let mut p1: na::Vector3<f64> = p_w.column(0).into_owned();
    let mut p2: na::Vector3<f64> = p_w.column(1).into_owned();
    let mut p3: na::Vector3<f64> = p_w.column(2).into_owned();
    if (p2 - p1).cross(&(p3 - p1)).norm() == 0.0 {
        return Vec::<Model>::new();
    }

    // get the bearing vectors
    let f1_og = p_i.column(0).normalize();
    let f2_og = p_i.column(1).normalize();
    let f3_og = p_i.column(2).normalize();
    let mut f1 = f1_og.clone();
    let mut f2 = f2_og.clone();
    let mut f3 = f3_og.clone();

    // compute transformation matrix T and feature vector f3
    let mut e1 = f1;
    let mut e3 = f1.cross(&f2).normalize();
    let mut e2 = e3.cross(&e1);
    let mut tt = na::Matrix3::<f64>::from_rows(&[e1.transpose(), e2.transpose(), e3.transpose()]);

    f3 = tt * f3;
    // enforce theta within [0, pi]
    // see paper page 4
    if f3[2] > 0.0f64 {
        // flip f1 and f2 to change the sign of e3
        f1 = f2_og;
        f2 = f1_og;
        f3 = f3_og;

        e1 = f1;
        e3 = f1.cross(&f2).normalize();
        e2 = e3.cross(&e1);
        tt = na::Matrix3::<f64>::from_rows(&[e1.transpose(), e2.transpose(), e3.transpose()]);

        f3 = tt * f3;

        p1 = p_w.column(1).into_owned();
        p2 = p_w.column(0).into_owned();
        p3 = p_w.column(2).into_owned();
    }

    let n1 = (p2 - p1).normalize();
    let n3 = n1.cross(&(p3 - p1)).normalize();
    let n2 = n3.cross(&n1);

    // compute transformation matrix N and the world point p3_eta
    // see paper equation (2)
    let nn = na::Matrix3::<f64>::from_rows(&[n1.transpose(), n2.transpose(), n3.transpose()]);
    p3 = nn * (p3 - p1);

    let d_12: f64 = (p2 - p1).norm();
    let f_1: f64 = f3[0] / f3[2];
    let f_2: f64 = f3[1] / f3[2];
    let p_1: f64 = p3[0];
    let p_2: f64 = p3[1];

    let cos_beta: f64 = f1.dot(&f2);
    let mut b: f64 = 1.0 / (1.0 - cos_beta * cos_beta) - 1.0;
    if cos_beta < 0.0 {
        b = -b.sqrt();
    } else {
        b = b.sqrt();
    }

    let f_1_pw2: f64 = f_1 * f_1;
    let f_2_pw2: f64 = f_2 * f_2;
    let p_1_pw2: f64 = p_1 * p_1;
    let p_1_pw3: f64 = p_1_pw2 * p_1;
    let p_1_pw4: f64 = p_1_pw3 * p_1;
    let p_2_pw2: f64 = p_2 * p_2;
    let p_2_pw3: f64 = p_2_pw2 * p_2;
    let p_2_pw4: f64 = p_2_pw3 * p_2;
    let d_12_pw2: f64 = d_12 * d_12;
    let b_pw2: f64 = b * b;

    // polynomial coefficients
    // see eq (11) in paper
    let a4 = -f_2_pw2 * p_2_pw4 - p_2_pw4 * f_1_pw2 - p_2_pw4;
    let a3 = 2.0 * p_2_pw3 * d_12 * b + 2.0 * f_2_pw2 * p_2_pw3 * d_12 * b
        - 2.0 * f_2 * p_2_pw3 * f_1 * d_12;
    let a2 = -f_2_pw2 * p_2_pw2 * p_1_pw2
        - f_2_pw2 * p_2_pw2 * d_12_pw2 * b_pw2
        - f_2_pw2 * p_2_pw2 * d_12_pw2
        + f_2_pw2 * p_2_pw4
        + p_2_pw4 * f_1_pw2
        + 2.0 * p_1 * p_2_pw2 * d_12
        + 2.0 * f_1 * f_2 * p_1 * p_2_pw2 * d_12 * b
        - p_2_pw2 * p_1_pw2 * f_1_pw2
        + 2.0 * p_1 * p_2_pw2 * f_2_pw2 * d_12
        - p_2_pw2 * d_12_pw2 * b_pw2
        - 2.0 * p_1_pw2 * p_2_pw2;
    let a1 = 2.0 * p_1_pw2 * p_2 * d_12 * b
        + 2.0 * f_2 * p_2_pw3 * f_1 * d_12
        - 2.0 * f_2_pw2 * p_2_pw3 * d_12 * b
        - 2.0 * p_1 * p_2 * d_12_pw2 * b;
    let a0 = -2.0 * f_2 * p_2_pw2 * f_1 * p_1 * d_12 * b
        + f_2_pw2 * p_2_pw2 * d_12_pw2
        + 2.0 * p_1_pw3 * d_12
        - p_1_pw2 * d_12_pw2
        + f_2_pw2 * p_2_pw2 * p_1_pw2
        - p_1_pw4
        - 2.0 * f_2_pw2 * p_2_pw2 * p_1 * d_12
        + p_2_pw2 * f_1_pw2 * p_1_pw2
        + f_2_pw2 * p_2_pw2 * d_12_pw2 * b_pw2;

    let all_roots = roots::find_roots_quartic(a4, a3, a2, a1, a0);
    let num_roots = all_roots.as_ref().len();

    let mut results = Vec::<Model>::new();
    for i in 0..num_roots {
        let cos_theta = all_roots.as_ref()[i];
        let cot_alpha =
            (-f_1 * p_1 / f_2 - cos_theta * p_2 + d_12 * b) /
                (-f_1 * cos_theta * p_2 / f_2 + p_1 - d_12);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        let sin_alpha = (1.0 / (cot_alpha * cot_alpha + 1.0)).sqrt();
        let cos_alpha = {
            let mut cos_alpha = (1.0 - sin_alpha * sin_alpha).sqrt();
            if cot_alpha < 0.0 {
                cos_alpha = -cos_alpha;
            }
            cos_alpha
        };

        // In paper, the R and t are defined as below:
        // p_world = R p_cam + t
        //
        // We modify it to get the ones we want:
        // R^T p_world - R^T t = p_cam
        // rotation_est = R^T
        let rotation_est: na::Matrix3<f64> = {
            let mut rr = na::Matrix3::<f64>::new(
                -cos_alpha, -sin_alpha * cos_theta, -sin_alpha * sin_theta,
                sin_alpha, -cos_alpha * cos_theta, -cos_alpha * sin_theta,
                0.0, -sin_theta, cos_theta,
            );
            tt.transpose() * rr * nn
        };

        // t_est = - R^T t
        let t_est: na::Vector3<f64> = {
            let mut t = na::Vector3::<f64>::new(
                d_12 * cos_alpha * (sin_alpha * b + cos_alpha),
                cos_theta * d_12 * sin_alpha * (sin_alpha * b + cos_alpha),
                sin_theta * d_12 * sin_alpha * (sin_alpha * b + cos_alpha),
            );
            t = p1 + nn.transpose() * t;
            -rotation_est * t
        };

        results.push(Model { rotation: rotation_est, translation: t_est });
    }

    return results;
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra as na;
    use rand::Rng;

    /// Return a tuple of p_cam, p_world, t_gt, rotation_gt
    fn easy_test_case() -> (na::Matrix3<f64>, na::Matrix3<f64>, na::Vector3<f64>, na::Matrix3<f64>) {
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
        // we assume:
        // p_cam = R_gt p_world + t_gt
        //
        // So to get world points from camera points, we have
        // p_world = R_gt^T * (p_cam - t_gt)
        let mut p_world: na::Matrix3<f64> = p_cam;
        for (i, mut column) in p_world.column_iter_mut().enumerate() {
            column -= &t_gt;
        }
        p_world = rotation_gt.transpose() * p_world;

        return (p_cam, p_world, t_gt, rotation_gt);
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
        let (p_cam, p_world, t_gt, rotation_gt) = easy_test_case();

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

    #[test]
    fn test_fischler() {
        let (p_cam, p_world, t_gt, rotation_gt) = easy_test_case();

        // get bearing vectors for the points in camera frame
        let mut p_i = p_cam;
        p_i.column_iter_mut().for_each(|mut c| c /= c[2]);

        // run fischler's p3p
        let solutions = fischler(&p_world, &p_i);
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

    #[test]
    fn test_kneip() {
        let (p_cam, p_world, t_gt, rotation_gt) = easy_test_case();

        // get bearing vectors for the points in camera frame
        let mut p_i = p_cam;
        p_i.column_iter_mut().for_each(|mut c| c /= c[2]);

        // run kneip's p3p
        let solutions = kneip(&p_world, &p_i);
        // ensure at least one solution is consistent with gt
        let mut flag: bool = false;
        for i in 0..solutions.len() {
            println!("t_gt: \n {}", t_gt);
            println!("t_est: \n {}", solutions[i].translation);
            println!("R_gt: \n {}", rotation_gt);
            println!("R_est: \n {}", solutions[i].rotation);
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
