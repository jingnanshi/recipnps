use recipnps::p3p::{fischler, grunert, kneip};
use nalgebra as na;
use rand::Rng;

fn easy_test_case() -> (na::Matrix3<f64>, na::Matrix3<f64>, na::Vector3<f64>, na::Matrix3<f64>) {
    // generate camera points (force positive z)
    let p_cam = {
        let mut p_cam = na::Matrix3::<f64>::new(
            0.7145222331218005, 0.6997616555794328, 2.7801634549912415,
            0.7253251306266671, 1.1639214982781518, 0.238599168957371,
            0.43484773318930925, 0.3052619752472596, 0.29437234778903254,
        );
        p_cam.set_column(0, &p_cam.column(0).normalize());
        p_cam.set_column(1, &p_cam.column(1).normalize());
        p_cam.set_column(2, &p_cam.column(2).normalize());
        p_cam
    };

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
fn test_p3p() {

    let (p_cam, p_world, t_gt, rotation_gt) = easy_test_case();

    // get bearing vectors for the points in camera frame
    let mut p_i = p_cam;
    p_i.column_iter_mut().for_each(|mut c| c /= c.norm());

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