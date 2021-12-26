use nalgebra as na;
use crate::{p3p, model, sac};

/// PnP with RANSAC + Fischler's P3P
pub fn pnp_ransac_fischler(world_points: &na::Matrix3xX<f64>, bearing_vectors: &na::Matrix3xX<f64>,
                           max_iterations: u32,
                           inlier_dist_threshold: f64,
                           probability: f64) -> Option<model::Model> {
    sac::ransac(world_points, bearing_vectors, &p3p::fischler, max_iterations, inlier_dist_threshold, probability)
}

/// PnP with RANSAC + Grunert's P3P
pub fn pnp_ransac_grunert(world_points: &na::Matrix3xX<f64>, bearing_vectors: &na::Matrix3xX<f64>,
                          max_iterations: u32,
                          inlier_dist_threshold: f64,
                          probability: f64) -> Option<model::Model> {
    sac::ransac(world_points, bearing_vectors, &p3p::grunert, max_iterations, inlier_dist_threshold, probability)
}

/// PnP with RANSAC + Kneip's P3P
pub fn pnp_ransac_kneip(world_points: &na::Matrix3xX<f64>, bearing_vectors: &na::Matrix3xX<f64>,
                        max_iterations: u32,
                        inlier_dist_threshold: f64,
                        probability: f64) -> Option<model::Model> {
    sac::ransac(world_points, bearing_vectors, &p3p::kneip, max_iterations, inlier_dist_threshold, probability)
}
